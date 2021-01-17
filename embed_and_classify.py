"""
This approach makes use of FastText to build sub-word n-grams
as part of the embedding, then passes the sum of the word vectors
through an XGBoost classifier to predict the classes.

Pros:
- Simple to implement, quick to train
- FastText allows OOV words to be predicted

Cons:
- XGBoost classifier is limited to only the given classes
- No positional encoding - might be important given the
lack of context for the words?

Results:
- This actually seems to work pretty well! It struggles
around the more similar classes - e.g. artist, athlete and
office holder - but overall seems to do a good job at correctly
classifying nouns into at least vaguely the right type. 
"""
import re
import joblib
import argparse
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from gensim.models import FastText
from xgboost import XGBClassifier
from utils import make_class_map

import seaborn as sn
import matplotlib.pyplot as plt

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data - this includes ensuring the classes
    start at 0, as well as removing punctuation and numbers from
    the string, and turning it into lower case. String is then split
    into a list, divided by spaces. 

    Parameters
    ----------
    data: pandas.DataFrame
        Input data to be preprocessed
    
    Returns
    -------
    data: pandas.DataFrame
        Transformed dataframe
    """
    if 'Class' in data.columns and data['Class'].min() > 0:
        data['Class'] = data['Class'] - data['Class'].min()
    data['Name'] = data['Name'].apply(
        lambda x: re.sub(r'[^\w ]', '', x).lower().split(' ')
    )
    return data

def load_and_split_data(path: str, train_size: Optional[float] = 0.8
                        ) -> Tuple[int, Tuple[list, list], Tuple[list, list]]:
    """Load data from a file, do preprocessing and split
    into training and test samples

    Parameters
    ----------
    path: str
        Filepath to data file to read
    train_size: float, default=0.8
        Fraction of sample to use as training data
    
    Returns
    -------
    n_classes: int
        Number of classes in the data
    train: Tuple[List[float], List[int]]
        Training data; [0] == data, [1] == labels
    test: Tuple[List[float], List[int]]
        Test data; [0] == data, [1] == labels
    """
    raw_data = pd.read_csv('data.csv')
    n_classes = len(raw_data['Class'].unique())

    raw_data = preprocess_data(raw_data)
    print(raw_data.head(5))

    train, test = train_test_split(raw_data, train_size=0.8)
    train_x = train['Name'].to_list()
    train_y = train['Class'].to_list()
    test_x = test['Name'].to_list()
    test_y = test['Class'].to_list()
    return n_classes, (train_x, train_y), (test_x, test_y)

def train_embedding(data: List[List[str]],
                    window: Optional[int] = 10,
                    embed_dim: Optional[int] = 100,
                    epochs: Optional[int] = 10, 
                    min_count: Optional[int] = 0,
                    lr: Optional[float] = 0.025,
                    min_n: Optional[int] = 3,
                    max_n: Optional[int] = 6, 
                    save_as: Optional[str] = 'fasttext.model'
                    ) -> FastText:
    """Train the FastText embedding with some input data.
    For more information, please see the gensim documentation!

    Parameters
    ----------
    data: List[List[str]]
        Input data in the form of a list of lists of strings
    window: int, default=10
        Size of the FastText context window
    embed_dim: int, default=100
        Size of the embedding space
    epochs: int, default=10
        Number of training epochs
    min_counts: int, default=0
        Minimum number of occurrences a word must have in order
        to be included - I'd advise, in this instance, to set to 0
    lr: float, default=0.025
        Learning rate
    min_n: int, default=3
        Minimum n_gram size
    max_n: int, default=6
        Maximum n_gram size
    save_as: str, default='fasttext.model'
        Filepath to save trained FastText model as
    
    Returns
    -------
    ft_model: FastText
        Trained model
    """
    ft_model = FastText(window=window)
    ft_model.build_vocab(sentences=data)
    ft_model.train(sentences=data, 
                   total_examples=len(data),
                   epochs=int(epochs),
                   min_count=int(min_count),
                   vector_size=int(embed_dim),
                   alpha=float(lr),
                   min_n=int(min_n),
                   max_n=int(max_n))
    ft_model.save(save_as)
    return ft_model

def get_word_vectors(data: List[List[str]], embedding: FastText) -> np.array:
    """Transform a dataset into the embedded vectors. Finds vector
    for each individual word, then takes sum of each vector to find
    vector for the overall noun.

    Parameters
    ----------
    data: List[List[str]]
        Input data to transform
    embedding: FastText
        Trained fasttext embedding model
    
    Returns
    -------
    vectors: np.array
        Transformed embedding vectors
    """
    vectors = []
    for entry in data:
        vect = [embedding.wv[x] for x in entry]
        vect = [sum(x) for x in zip(*vect)]
        vectors.append(vect)
    vectors = np.array(vectors)
    return vectors

def train_xgb_model(data_X: np.array, data_Y: List[int],
                    save_as: Optional[str] = 'xgb_classifier.model',
                    **xgb_parameters) -> XGBClassifier:
    """Train the XGBoost Classifier using the embedded vectors.
    Any kwargs parameters passed to this function will be passed
    on to the XGBClassifier object for tuning.

    Parameters
    ----------
    data_X: np.array
        Data in embedded vector form
    data_Y: List[int]
        Target labels for training data
    save_as: str, default='xgb_classifier.model'
        Filepath to save trained model to
    xgb_parameters: kwargs dict
        Any additional parameters to be passed to Classifier
    
    Returns
    -------
    xgb_model: XGBClassifier
        Trained xgboost model
    """
    xgb_model = XGBClassifier(**xgb_parameters)
    print('Fitting XGBoost model')
    xgb_model.fit(data_X, data_Y)
    train_pred = xgb_model.predict(data_X)
    print(train_pred[:5], data_Y[:5])
    joblib.dump(xgb_model, save_as)
    return xgb_model, train_pred

def embed_and_predict(data: List[List[str]],
                      embedding: FastText,
                      classifier: XGBClassifier) -> np.array:
    """Transform a dataset into embedded vectors then predict
    the class using the classifier.

    Parameters
    ----------
    data: List[List[str]]
        Dataset to predict
    embedding: FastText
        FastText embedding model
    classifier: XGBClassifier
        XGBoost classifier model
    
    Returns
    -------
    pred_y: np.array
        Predicted labels for each entry in data
    """
    vectors = get_word_vectors(data, embedding)
    pred_y = classifier.predict(vectors)
    return pred_y

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest='mode')

    train_parse = subparser.add_parser('train')
    train_parse.add_argument('--data', default='data.csv')
    # path to a pre-trained embedding model
    train_parse.add_argument('--embedding', type=str, default=None)
    # embed_params must be comma-separated, e.g.
    # window=10,epochs=20,embed_dim=200
    train_parse.add_argument('--embed_params', type=str, default=None)
    train_parse.add_argument('--save_emb_as', default='fasttext.model')
    train_parse.add_argument('--save_cls_as', default='xgb_classifier.model')
    # parameters for training xgboost
    train_parse.add_argument('--objective', default='multi:softmax')
    train_parse.add_argument('--verbosity', default=2)
    train_parse.add_argument('--min_split_loss', default=0.2)
    train_parse.add_argument('--max_depth', default=4)
    train_parse.add_argument('--min_child_weight', default=10)
    train_parse.add_argument('--subsample', default=0.6)

    eval_parse = subparser.add_parser('eval')
    eval_parse.add_argument('--data', default='data.csv')
    eval_parse.add_argument('--embedding', default='fasttext.model')
    eval_parse.add_argument('--classifier', default='xgb_classifier.model')
    eval_parse.add_argument('--save_to', default='predictions.csv')

    args = parser.parse_args()

    if args.mode == 'train':
        n_classes, train, test = load_and_split_data(args.data)
        train_x, train_y = train
        test_x, test_y = test

        if not args.embedding:
            print('Training new embedding')
            if args.embed_params:
                embed_params = {p.split('=')[0]: p.split('=')[1] 
                                for p in args.embed_params.split(',')}
                print(embed_params)
                ft_model = train_embedding(train_x,
                                           save_as=args.save_emb_as,
                                           **embed_params)
            else:
                ft_model = train_embedding(train_x, save_as=args.save_emb_as)
        else:
            print('Loading pre-trained embedding')
            ft_model = FastText.load(args.embedding)

        train_vectors = get_word_vectors(train_x, ft_model)
        print(train_vectors.shape)

        xgb_parameters = {
            'objective': args.objective,
            'num_class': n_classes,
            'verbosity': args.verbosity,
            'min_split_loss': args.min_split_loss,
            'max_depth': args.max_depth,
            'min_child_weight': args.min_child_weight,
            'subsample': args.subsample
        }

        xgb_model, train_pred = train_xgb_model(
            train_vectors,
            train_y,
            args.save_cls_as,
            **xgb_parameters
        )
        train_f1_score = f1_score(train_pred, train_y, average='macro')
        print(train_f1_score)

        # now predict test data and see if it works as well
        test_pred = embed_and_predict(test_x, ft_model, xgb_model)
        test_f1_score = f1_score(test_pred, test_y, average='macro')
        print(test_f1_score)

        # get the confusion matrices
        train_confuse = confusion_matrix(train_y, train_pred)
        test_confuse = confusion_matrix(test_y, test_pred)
        print(train_confuse)
        print(test_confuse)

        sn.heatmap(train_confuse, annot=True)
        plt.savefig('../MyHomeFolder/train_confusion.png', bbox_inches='tight')
        plt.clf()

        sn.heatmap(test_confuse, annot=True)
        plt.savefig('../MyHomeFolder/test_confusion.png', bbox_inches='tight')
        plt.clf()

    elif args.mode == 'eval':
        data = pd.read_csv(args.data)
        input = preprocess_data(data)
        embedding = FastText.load(args.embedding)
        classifier = joblib.load(args.classifier)
        pred = embed_and_predict(input, embedding, classifier)
        output = data[['Name']]
        output['Predicted'] = pred
        output.to_csv(args.save_to)

    

    

