[![Build Status](https://travis-ci.org/sildar/Kaggle-Toxic-CCC.svg?branch=master)](https://travis-ci.org/sildar/Kaggle-Toxic-CCC)
# Kaggle-Toxic-CCC

## Challenge

More info on the challenge on the [Kaggle website](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).

The goal is to detect comments that are toxic, with a few distinct non-exclusive classes (e.g. insult, threat, identity hate, ...).


## Data

Data is not included, use is restricted to people who registered for the challenge


## Usage

Used libraries should be included in the requirements.txt file.

Uses mainly the following libraries : keras, tensorflow, pandas, nltk, sklearn, numpy.

Uses fasttext word embeddings released in the public domain. Can be found [here](https://fasttext.cc/docs/en/english-vectors.html)
under the Creative Commons Attribution-Share-Alike License 3.0.

```
python ./gru_classification.py
```

should be enough, it assumes a ./data/ directory for both inputs (datasets, word embeddings)
and output (submission file is written in the data/ dir)
