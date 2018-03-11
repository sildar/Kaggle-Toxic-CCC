[![Build Status](https://travis-ci.org/sildar/Kaggle-Toxic-CCC.svg?branch=master)](https://travis-ci.org/sildar/Kaggle-Toxic-CCC)
# Kaggle-Toxic-CCC

## Challenge

More info on the challenge on the [Kaggle website](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).

The goal is to detect comments that are toxic, with a few distinct non-exclusive classes (e.g. insult, threat, identity hate, ...).


## Data

Data is not included, use is restricted to people who registered for the challenge.


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

## Description

I started with a [Kaggle kernel](https://www.kaggle.com/rohitanil/lemmatization-and-pooled-gru) that provided an already parameters-optimized GRU RNN.

Code was messy, so I simplified it, pep8, added requirements and travisCI to check them, etc...

The kernel used lemmatization on training and testing data. However, the word embeddings that were used (seed Data section) were trained on non-lemmatized text.
It seemed surprising that lemmatizing and then casting to a non-lemmatized space would yield better results than keeping words as is.
So I removed lemmatization, which improved slightly the results (0.9820 -> 0.9828).

I tried some thresholding (setting very high scores (>0.99) to 1 and very low scores (<0.01) to 0) as a hacky way to gain a few decimals but these had lower scores.

## Things to try

I wanted to try and improve on the obscene and insult categories, as I expect a shallow approach with a list of swear words to perform very well on those well-defined categories.
However, I calculated the ROC scores of each category during training. It appears that the obscene category is (somewhat unsurprisingly) already very well detected by the RNN.
It is the best performing category so probably not the one to investigate first.
While there might be a little bit of overfitting, I doubt a hardcoded list would perform significantly better.

The worst performing category is the toxic category, which is a bit expected since the frontier between toxicity and, e.g., humor and sarcasm, is not always easy to detect, especially in a written form.

Better scores should be in reach with multiple scoring methods / NNs, but the difference between overfitting by trial and error and intrinsically better models is not clear at that stage
(see [a linear combination of scoring in this kernel](https://www.kaggle.com/rednivrug/blend-it-all) which is currently the best performing kernel and pretty close to top results).

## Things to improve on

Apart from the *things to try*, there are some more software development improvements to consider.

There is no testing as this is a POC. Travis CI is set however, so just adding a setup.py file and some testing would be kind of quick.
Travis CI is only used as a way to check that the imports are covered by the requirements.txt file.

Parameters are hardcoded and should be in another file or passed as script parameters.

Documentation is a bit lacking, but again, this is a POC with around 150 lines. Still, GRU training should be in its own function, outputting too.