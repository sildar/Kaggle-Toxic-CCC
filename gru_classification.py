#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from nltk import word_tokenize

from keras.models import Model, load_model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
import logging
import sys

import os
os.environ['OMP_NUM_THREADS'] = '3'

np.random.seed(42)

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')


def preprocess(trainfile, testfile, max_features, maxlen):
    """
    Preprocesses training and testing data
    Tokenizes and pads sentences
    A tokenizer is also given back for keeping track of the word index
    """
    logging.info("Starting Extraction")
    train = pd.read_csv(trainfile)
    test = pd.read_csv(testfile)

    y_train = train[["toxic", "severe_toxic", "obscene",
                     "threat", "insult", "identity_hate"]].values

    logging.info("Extracting train dataset content")
    X_train1 = [" ".join(word_tokenize(traincommentcontent))
                for traincommentcontent in train['comment_text']]

    logging.info("Extracting test dataset content")
    X_test1 = [" ".join(word_tokenize(testcommentcontent))
               for testcommentcontent in test['comment_text']]
    logging.info("Extraction finished")

    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(X_train1) + list(X_test1))
    X_train = tokenizer.texts_to_sequences(X_train1)
    X_test = tokenizer.texts_to_sequences(X_test1)
    x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(X_test, maxlen=maxlen)

    return x_train, x_test, y_train, tokenizer


def main():

    trainingfile = 'data/train.csv'
    testingfile = 'data/test.csv'

    max_features = 30000
    maxlen = 100
    x_train, x_test, y_train, tokenizer = preprocess(trainingfile, testingfile,
                                                     max_features, maxlen)

    logging.info("Loading word embeddings")
    EMBEDDING_FILE = 'data/crawl-300d-2M.vec'
    embeddings_index = {}
    with open(EMBEDDING_FILE, encoding='utf-8') as f:
        for line in f:
            linecontent = line.rstrip().rsplit(' ')
            word = linecontent[0]
            vec = linecontent[1:]
            embeddings_index[word] = np.asarray(vec, dtype='float32')

    embed_size = 300
    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.zeros((nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    class RocAucEvaluation(Callback):
        def __init__(self, validation_data=(), interval=1):
            super(Callback, self).__init__()

            self.interval = interval
            self.X_val, self.y_val = validation_data

        def on_epoch_end(self, epoch, logs={}):
            if epoch % self.interval == 0:
                y_pred = self.model.predict(self.X_val, verbose=0)
                score = roc_auc_score(self.y_val, y_pred)
                allclasses = roc_auc_score(self.y_val, y_pred, average=None)
                logging.info("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch + 1, score))
                logging.info("Classes ROC scores : %s", str(allclasses))

    def get_model():
        inp = Input(shape=(maxlen, ))
        x = Embedding(max_features, embed_size,
                      weights=[embedding_matrix])(inp)
        x = SpatialDropout1D(0.4)(x)
        x = Bidirectional(GRU(80, return_sequences=True, activation='relu',
                              dropout=0.3, recurrent_dropout=0.))(x)
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        conc = concatenate([avg_pool, max_pool])
        outp = Dense(6, activation="sigmoid")(conc)

        model = Model(inputs=inp, outputs=outp)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        return model

    if not os.path.exists('./data/gru_model.h5'):
        logging.info("Training model")
        model = get_model()

        batch_size = 32
        epochs = 2

        X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train,
                                                      train_size=0.95,
                                                      random_state=233)

        RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)
        model.fit(X_tra, y_tra, batch_size=batch_size,
                  epochs=epochs, validation_data=(X_val, y_val),
                  callbacks=[RocAuc], verbose=2)
        model.save('./data/gru_model.h5')
    else:
        logging.info("Loading model")
        model = load_model('./data/gru_model.h5')

    logging.info("Predicting on test set")
    y_pred = model.predict(x_test, batch_size=1024)

    submission = pd.read_csv('data/sample_submission.csv')
    submission[["toxic", "severe_toxic", "obscene",
                "threat", "insult", "identity_hate"]] = y_pred

    logging.info("Printing to output file")
    submission.to_csv('data/submission.csv', index=False)


if __name__ == '__main__':
    # temporary hack to test travis
    if len(sys.argv) == 2 and sys.argv[1] == 'travis':
        quit()

    main()
