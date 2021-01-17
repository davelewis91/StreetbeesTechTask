"""
Named Entity Recognisation

This approach seems reasonable - take a pre-trained
NER model (through spaCy) and use transfer learning to 
update the parameters for our specific classes. Never actually
tried this before though, or even used spaCy, so let's see how it goes!

Pros:
- make use of pretrained model to kick-start training
- NER is designed for exactly this kind of thing

Cons:
- our desired classes not supported by default
- NER typically used to identify nouns in a sentence,
while here the nouns are isolated, so I don't know how it will
perform

Results:
- well... it doesn't seem to work, and I'm not sure why. Spacy kept
failing silently - the model 'trains' but no losses are returned and
it doesn't predict anything. Couldn't work out what was going wrong,
not within a limited time frame anyway.
"""
import random
import warnings
from typing import Dict, NewType, TypeVar, List, Tuple

import spacy
from spacy.util import minibatch, compounding

from utils import make_class_map, load_data
from utils import PDDataFrame

TokenDataSet = NewType('TokenDataSet', List[Tuple[str, dict]])
SpacyModelName = NewType('SpacyModelName', str)
SpacyModel = TypeVar('SpacyModel')

def entityize_data(data: PDDataFrame, classmap: Dict[int, str]) -> TokenDataSet:
    """Transform a loaded text dataset into appropriately
    annotated entries for a spaCy model to train on. 

    >> entityize_data(load_data('data.csv'))
    ... [('A Company Name', {'entities': [0, 14, 'Company']})
        ...]
    
    Parameters
    ----------
    data: pandas.DataFrame
        Input data to entity-ise
    classmap: dict
        Mapping from class ID to class name
    
    Returns
    -------
    entities: TokenDataSet
        Data records in correct spaCy form
    """
    datadict = data.to_dict(orient='records')
    entities = []
    for record in datadict:
        entity = [0, len(record['Name']), classmap[record['Class']]]
        entities.append((record['Name'], {'entities': entity}))
    return entities

def get_model(model: SpacyModelName = None) -> Tuple:
    """Load a pretrained model from a spacy model name,
    or create one from a blank model

    Parameters
    ----------
    model: str, default=None
        Name of the model to load
    
    Returns
    -------
    model: Spacy language model
        Loaded or newly created model
    optim: Spacy optimizer
        Model optimizer object
    """
    # load up black spacy model
    if model is None:
        model = spacy.blank('en')
        # create NER model and add to pipeline
        ner = model.create_pipe('ner')
        model.add_pipe(ner, last=True)
        optim = model.begin_training()
    else:
        model = spacy.load(model)
        optim = model.resume_training()
    return model, optim

def train_ner_model(model: SpacyModelName,
                    train: TokenDataSet, 
                    test: TokenDataSet, 
                    classmap: Dict[int, str],
                    outdir: str,
                    n_iter: int = 100) -> SpacyModel:
    """Trains a spaCy NER model using the input datasets.

    Parameters
    ----------
    model: str or None
        Name of spacy model to use (if None, uses a blank 'en' model)
    train: TokenDataSet
        Training data set
    test: TokenDataSet
        Test data set
    classmap: dict
        Class ID to class name mapping
    outdir: str
        Directory to save trained model to
    n_iter: int, default=100
        Number of training iterations to make
    
    Returns
    -------
    model: spacy language model
        Trained NER model
    """
    # load up black spacy model
    model, optim = get_model(model)
    ner = model.get_pipe('ner')
    # add list of classes to the model
    for _, v in classmap.items():
        ner.add_label(v)
    
    # only want to train the NER model - disable the rest
    disable = [pipe for pipe in model.pipe_names if pipe != 'ner']

    # begin training
    with model.disable_pipes(*disable):
        sizes = compounding(1, 4, 1.001)
        training_loss = []
        for _ in range(n_iter):
            random.shuffle(train)
            # break the training data into batches
            batches = minibatch(train, size=sizes)
            batch_loss = {}
            for batch in batches:
                text, label = zip(*batch)
                model.update(text, label, sgd=optim, drop=0.35, losses=batch_loss)
            #print('Batch losses:', batch_loss)
    return model

if __name__ == '__main__':
    classmap = make_class_map('classes.txt')
    data = load_data('data.csv')
    tokens = entityize_data(data, classmap)
    train = []
    test = []
    for t in tokens:
        if random.choices([0, 1], [0.8, 0.2]) == 0:
            train.append(t)
        else:
            test.append(t)
    model = train_ner_model('en_core_web_sm', train, test, classmap, './')
    print('testing with: ', test[0])
    doc = model(test[0][0])
    print(doc)
    for ent in doc.ents:
        print(ent.label_, ent.text)