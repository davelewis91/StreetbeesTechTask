"""
Embed and Classify

Idea with this one is to build a positional word embedding,
but based on the letters rather than the words (because
there aren't enough words in each entry and each individual
word is not repeated enough to build a regular word2vec 
embedding) and feed that embedding into a classifier. 

Pros:
- architecture here is well understood, relatively
simple to implement
- letter-by-letter encoding may be more powerful than
a word-level embedding
- can include punctuation in the embedding

Cons:
- Embedding space may not learn a powerful enough 
representation to be able to distinguish certain classes
- Not safe against unknown/unseen classes

Results:
- Still think this is the best approach, but I can't get the
positional encoding to work - code works without it, but doesn't
actually seem to learn. Not sure if this is just a parameter tuning
problem or a more fundamental problem with how I have structured the
network. 
"""
import math
import copy
import string
import random
import joblib
import argparse
import pandas as pd
from typing import Optional
from sklearn.preprocessing import OneHotEncoder

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split

from utils import make_class_map

class PositionalEncoder(nn.Module):
    """Encodes a word vector with positional information. 
    If word position is even, use sin, if odd use cosine.

    Parameters
    ----------
    n_dim: int
        Number of input dimensions
    dropout: float, default=0.1
        Dropout rate of layer
    max_len: int, default=500
        Max size of attention
    
    Returns
    -------
    None
    """
    def __init__(self, n_dim: int, dropout: Optional[float] = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        ## build the positional encoding function
        ## I'm just using the typical transformer form here
        pos = torch.arange(0, n_dim)
        pe = torch.zeros(n_dim)
        div_term = torch.exp(torch.arange(0, n_dim).float() 
                             * (-math.log(10000.0) / n_dim))
        for i in range(n_dim):
            if i % 2 == 0:
                pe[i] = torch.sin(pos[i] * div_term)
            else:
                pe[i] = torch.cos(pos[i] * div_term)
        print(pe.shape)
        print(pe)
        #pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.prod(x, self.pe, dim=2)
        return self.dropout(x)

class NounClassifier(nn.Module):
    """Classify a noun into N classes based on the letters in the noun.
    Structure of the classifier is:
    - Embedding layer (n_tokens, n_embedding_dim)
    - Fully-connected input layer (n_embedding_dim, n_hidden)
    - Hidden layer (n_hidden, n_hidden)
    - Output layer (n_hidden, n_classes)

    No softmax activation is applied to the final layer, for evaluation
    with CrossEntropyLoss - when running in eval mode, you'll have to
    scale the output appropriately yourself.

    Parameters
    ----------
    n_tokens: int
        Size of the vocabulary of the corpus
    n_classes: int
        Number of classes to classify into
    n_dim: int, default=300
        Size of the word embedding space
    n_hidden: int, default=30
        Size of the hidden layer
    dropout: float, default=0.1
        Dropout rate during training (only used when positional
        encoding is enabled)
    
    Returns
    -------
    None
    """
    def __init__(self, 
                 n_tokens: int,
                 n_classes: int,
                 n_dim: Optional[int] = 300,
                 n_hidden: Optional[int] = 30,
                 dropout: Optional[float] = 0.1) -> None:
        super().__init__()
        self.n_dim = n_dim
        self.embedding = nn.Embedding(n_tokens, n_dim)
        #self.pos_encoder = PositionalEncoder(n_dim, dropout)
        self.in_layer = nn.Linear(n_dim, n_hidden)
        self.relu = nn.ReLU()
        self.hidden = nn.Linear(n_hidden, n_hidden)
        self.out_layer = nn.Linear(n_hidden, n_classes)
        return
    
    def init_weights(self):
        """initialise the weights of the fully-connected layers
        using Xavier Uniform distribution
        """
        nn.init.xavier_uniform(self.in_layer)
        nn.init.xavier_uniform(self.hidden)
        nn.init.xavier_uniform(self.out_layer)
        return
    
    def forward(self, x):
        """Feed-forward function.

        Parameters
        ----------
        x: torch.Tensor
            Tensor to feed through network
        
        Returns
        -------
        x: torch.Tensor
            Transformed tensor
        """
        x = self.embedding(x) * math.sqrt(self.n_dim)
        # flatten the embedding by taking sum of squares
        x = torch.sqrt(torch.sum(x**2, dim=1))
        #x = self.pos_encoder(x)
        x = self.relu(self.in_layer(x))
        x = self.relu(self.hidden(x))
        x = self.out_layer(x)
        return x

class TokenDataset(Dataset):
    """A class to load the data, generate a vocabulary and allow for
    parallelised reading through a pytorch DataLoader while training.

    Specify datafile to load data for training (resets vocabulary)
    or specify vocabfile to load a pre-defined vocabulary (for eval).
    If loading via vocabfile, you can then call load_raw_data to bring
    your data into the process.

    Parameters
    ----------
    datafile: str, default=None
        Filepath to data to load for training
    max_str_len: int, default=50
        Maximum allowed length of any given string in the data
    vocabfile: str, default=None
        Filepath to a pre-defined vocab file for evaluation
    
    Attributes
    ----------
    `self.n_classes`: int
        Number of classes found in the data
    `self.raw_data`: pandas.DataFrame
        Raw data as loaded from the file
    `self.vocab_`: set
        Every unique 'word' in the training dataset
    `self.vocab2index_`: dict
        Mapping from word to a unique index
    `self.index2vocab_`: dict
        Reverse mapping of vocab2index
    
    Returns
    -------
    None
    """
    def __init__(self, 
                 datafile: Optional[str] = None,
                 max_str_len: Optional[str] = 50,
                 vocabfile: Optional[str] = None) -> None:
        if datafile is not None:
            self.load_raw_data(datafile)
            self.n_classes = len(self.raw_data['Class'].unique())
            self.max_str_len = max_str_len
            # build our letter 'vocabulary'
            self.vocab_ = set()
            ## this is potentially slow, is there a better way to do this?
            for row in self.raw_data['tokens']:
                self.vocab_.update(row)
            self.vocab2index_ = {}
            self.index2vocab_ = {}
            for i, letter in enumerate(self.vocab_):
                self.vocab2index_[letter] = i
                self.index2vocab_[i] = letter
        elif vocabfile is not None:
            vocab_dict = joblib.load(vocabfile)
            self.vocab_ = vocab_dict['vocab']
            self.vocab2index_ = vocab_dict['vocab2index']
            self.index2vocab_ = vocab_dict['index2vocab']
            self.max_str_len = vocab_dict['maxstrlen']
        else:
            raise ValueError('Either datafile or vocabfile must be not None')
        return
    
    def load_raw_data(self, file: str) -> None:
        """Load and process raw data from a file

        Parameters
        ----------
        file: str
            Filepath to datafile
        
        Returns
        -------
        None
        """
        self.raw_data = pd.read_csv(file)
        self.raw_data['tokens'] = self.raw_data['Name'].apply(
            lambda x: [l.lower() for l in list(x)]
        )
        return
    
    def train_test_split(self, 
                         train_frac: Optional[float] = 0.8
                        ) -> Tuple[object, object]:
        """Split the raw data into a training and test dataset
        and return respective TokenDataset objects

        Parameters
        ----------
        train_frac: float, default=0.8
            Fraction of dataset to reserve for training
        
        Returns
        -------
        train: TokenDataset
            Training dataset
        test: TokenDataset
            Test dataset
        """
        train_ids = []
        test_ids = []
        for i in range(len(self.raw_data)):
            if random.choices([True, False], [train_frac, 1-train_frac]):
                train_ids.append(i)
            else:
                test_ids.append(i)
        test = copy.copy(self)
        test.raw_data = test.raw_data.iloc[test_ids]
        self.raw_data = self.raw_data.iloc[train_ids]
        return self, test
    
    def shuffle(self):
        """Shuffle the raw data randomly"""
        self.raw_data = self.raw_data.sample(frac=1).reset_index(drop=True)
        return
    
    def save_vocab(self, path: str) -> None:
        """Save the vocabulary defined for training, for later use.
        Can be loaded back in by instancing a new TokenDataset with
        vocabfile parameter.

        Parameters
        ----------
        path: str
            Filepath to save vocabulary to
        
        Returns
        -------
        None
        """
        vocab_dict = {
            'vocab': self.vocab_,
            'index2vocab': self.index2vocab_,
            'vocab2index': self.vocab2index_,
            'maxstrlen': self.max_str_len
        }
        joblib.dump(vocab_dict, path)
        return

    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        ## encode the individual letters
        name = self.raw_data.iloc[idx]['Name']
        token_tensor = torch.zeros(1, self.max_str_len)
        for i, letter in enumerate(list(name)):
            if i >= self.max_str_len:
                break
            _lower = letter.lower()
            token_tensor[0][i] = self.vocab2index_[_lower]
        labels = self.raw_data.iloc[idx]['Class'] - 1
        return token_tensor, labels

def train_model(data: DataLoader,
                n_classes: int,
                max_str_len: int,
                epochs: Optional[int] = 10,
                embed_dim: Optional[int] = 300,
                hid_size: Optional[int] = 30,
                lr: Optional[float] = 0.01) -> NounClassifier:
    """Train the noun classifier model. Uses CrossEntropyLoss
    as the cost function and uses Adam as the optimiser.

    Parameters
    ----------
    data: DataLoader
        Input dataset
    n_classes: int
        Number of classes to classify
    max_str_len: int
        Maximum length of any given string
    epochs: int, default=10
        Number of epochs to train for
    embed_dim: int, default=300
        Size of the embedding space
    hid_size: int, default=30
        Size of the hidden layer
    lr: float, default=0.01
        Learning rate
    
    Returns
    -------
    model: NounClassifier
        Trained model
    """
    n_tokens = len(data.dataset.vocab_)
    model = NounClassifier(
        n_tokens,
        n_classes,
        embed_dim,
        hid_size
    )
    model.train()
    model.zero_grad()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        epoch_loss = 0
        n_batches = 0
        for i, batch in enumerate(data):
            n_batches += 1
            batch_data = batch[0].long().reshape(-1, max_str_len)
            batch_labels = batch[1]
            optimizer.zero_grad()
            output = model(batch_data)
            batch_loss = criterion(output, batch_labels)
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()
            if i % 500 == 0 and i > 0:
                print(f'Epoch {epoch} Batch {i} loss: {epoch_loss / max_str_len}')
        print('-----------')
        print('Avg Epoch loss: ', epoch_loss / n_batches)
        print('-----------')
    return model        


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    train_parse = subparsers.add_parser('train')
    train_parse.add_argument('--data', default='data.csv')
    train_parse.add_argument('--vocab', default='model_vocab.pkl')
    train_parse.add_argument('--max_str_len', default=50)
    train_parse.add_argument('--batch_size', default=64)
    train_parse.add_argument('--embed_dim', default=100)
    train_parse.add_argument('--epochs', default=20)
    train_parse.add_argument('--hidden_dim', default=30)
    train_parse.add_argument('--lr', default=0.01)
    train_parse.add_argument('--save_model_as',
                             default='embedding_classifier.pt')

    eval_parse = subparsers.add_parser('eval')
    eval_parse.add_argument('--data', default='data.csv')
    eval_parse.add_argument('--model', default='embedding_classifier.pt')
    eval_parse.add_argument('--vocab', default='model_vocab.pkl')

    args = parser.parse_args()

    if args.mode == 'train':
        dataset = TokenDataset(args.data, max_str_len=args.max_str_len)
        dataset.save_vocab(args.vocab)
        dataset.shuffle()
        n_classes = dataset.n_classes
        train, test = dataset.train_test_split()
        train_dl = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    
        model = train_model(
            train_dl, 
            n_classes, 
            args.max_str_len, 
            epochs=args.epochs,
            embed_dim=args.embed_dim,
            hid_size=args.hidden_dim,
            lr=args.lr
        )
        torch.save(model.state_dict(), args.save_model_as)
    elif args.mode == 'eval':
        model = torch.load(args.model)
        model.eval()
        dataset = TokenDataset(vocabfile=args.vocab)
        dataset.load_raw_data(args.data)
        output = model(dataset)

    


