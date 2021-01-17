## Streetbees Technical Task

### Overview

The aim of the task is to build a model that classifies the given list of strings into a number of types of nouns. 

Turns out, this is quite difficult! Or it is in this case anyway - the nouns are presented on their own, with no surrounding context with which to infer the type of the noun. This presents particular difficulty in cases where the noun classes are semantically similar - for example, in the case of 'Athlete', 'Artist' and 'Office Holder', the nouns are all people's names, often without any other information, and inferring the right context from a name alone is almost impossible. 

I've taken three different approaches to solving this problem:

* Named Entity Recognisation using spaCy
* Positional encoding and classification with pytorch
* Fasttext embedding feeding into a tree-based classifier, with gensim and XGBoost

To start with, I thought NER might be a good way to go about it - in theory, it's designed for exactly this kind of work. However, this approach threw up two problems: 1) the nouns are isolated and have no surrounding sentence to extract context from, which is what NER relies upon, and 2) I had never used spaCy before! I've left the attempt I made as a script, `ner_transfer.py`, for your reference, but it doesn't work; the model seems to train, but I guess spaCy is failing silently somewhere in the background because it returns an empty list for the losses and won't predict anything.

The second approach I thought might be the best one - the code is available in the `torch_embed_and_classify.py` script. The idea behind it was to break the nouns into letters, rather than words, and build a 'word' embedding with positional encoding, a la transformers, then feed that embedding into a fully-connected layer or two for the classification. I got the model to train, minus the positional encoding, but the model didn't seem to actually learn anything; the losses converged after a single epoch. I could have spent hours tuning parameters to find the best combination, but I also wasn't sure it was a more fundamental issue with the idea of embedding -> classification, so instead I moved on to a third approach, which is related to this approach and might indicate whether the issue is with the parameters or the idea itself.

The third approach, in `embed_and_classify.py`, is similar in idea to the 2nd, but forgoes the positional encoding in favour of building subword n-grams with Facebook's FastText embedding algorithm. This has the double benefit of being similar to a full positional encoding and also allowing the use of out-of-vocabulary words to be evaluated, which is super useful when dealing exclusively in nouns. The embedded vectors are then fed into an XGBoost classifier to predict the classes for each noun. This seemed to work really well - with minimal hyperparameter tuning, I was able to achieve an F1 score of 0.63 on the training set and 0.61 on the test set. The confusion matrices (below) also show relatively little confusion across classes - the most confusion occurs between the very similar classes such as people, films/albums/written work, and animals/plants, as would be expected. 

### How to run the code

The `ner_transfer.py` script can be run just by calling `python3 ner_transfer.py` - no additional arguments are required (it doesn't work anyway).

The other scripts have two run modes: `train` and `eval`. You can specify which mode to run in, along with any other required parameters, at runtime, for example:

`python3 embed_and_classify.py train --data data.csv`
`python3 torch_embed_and_classify.py eval --data data.csv`

The default parameters are the parameters I used for the final version of the model, so if you run with the default parameters you should get the same results as me. 

For the full list of available parameters, please take a look at the scripts. You can also call `--help` on either script to get the list of command line arguments available.

All code runs in Python 3.8, I'm not sure about older versions. A pip requirements file is available, to get the list of required packages and versions for installation. 

### Expansion

If I had more time, I think I'd keep fiddling with the third approach - I'm pretty sure the power is coming from the fasttext embedding, and it would be interesting to see whether any extra performance could be squeaked out from other classification models, or by further tuning the hyperparameters.

As for the part of the brief around making sure the model can handle unseen classes, this is a general difficulty with machine learning, and there's no obvious solution to it, that I know of, anyway! However, besides continuous monitoring of the model to detect changes in model performance that might be indicative of a change in data distribution, there might be a couple of approaches one could investigate:

* Prediction probability thresholding
* Anomaly detection

The 1st approach is definitely the simplest - look at the output from the classification model, and if the highest class probability is below some threshold, say that the entry belongs to some unknown class that the model can't correctly identify. This approach requires no changes to the existing model, and would allow you to build up a dataset of 'unknown' data points, that you could then use to train a new model after you've worked out what the new class is! But, choosing the threshold may be difficult as you would have to do it while building the model, when you don't have any examples of 'unknown' classes to identify the right threshold... 

The 2nd approach is more complex, but perhaps more complete. You could feed the embedded vectors into something like an Isolation Forest or a Gaussian Mixture Model, to do a binary classification with hugely imbalanced classes - i.e. take a semi-supervised approach, where you are identifying data points that are different to anything previously seen. You can then send any non-anomalous data points into the regular classifier, and hold back any anomalous data points to investigate later. The semi-supervised approach means you don't have to have many/any unknown class data to build the model - but you do have a build a third model, and you run the risk of removing otherwise acceptable data from your input. 