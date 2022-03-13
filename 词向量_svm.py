import joblib
import numpy as np
import pandas as pd
from keras_preprocessing import sequence
from numpy import reciprocal
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC

from word_vector_deal import creat_data

if __name__ == '__main__':
    data = creat_data()
    tokenizer = joblib.load('模型/tokenizer.model')
    maxlen = 350
    X = tokenizer.texts_to_sequences(data['cut_words'].values)
    X = sequence.pad_sequences(X, maxlen=maxlen)
    y = data['label']
    # train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)
    clf = SVC(verbose=True)
    param_distribution = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'sigmoid', 'poly'],
                          'gamma': [0.1, 1, 10]}
    random_search_cv = RandomizedSearchCV(
        clf, param_distribution, verbose=1,
        n_iter=10, n_jobs=7,
        cv=3)
    random_search_cv.fit(X, y)
    # joblib.dump(clf, '模型/词向量-svm-model.m')
    # y_pred = clf.predict(test_x)
    # print(classification_report(test_y, y_pred))
    print(random_search_cv.best_params_)
    print(random_search_cv.best_score_)
    print(random_search_cv.best_estimator_)
