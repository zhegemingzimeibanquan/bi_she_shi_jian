import joblib
import pandas as pd
from sklearn.metrics import classification_report

from url_deal import data_deal as dd
import data_try as dt

if __name__ == '__main__':
    test_xss = pd.read_table('../data/test_xss.txt', header=None)
    test_normal = pd.read_table('../data/test_normal.txt', header=None)

    data = dt.step1(test_xss, test_normal)
    data = dd.filter3(data)
    X = data[['len', 'url_count', 'evil_char', 'evil_word', 'shang']]
    y = data['label']
    print(data[['len', 'url_count', 'evil_char', 'evil_word', 'shang']].describe())
    data[['len', 'url_count', 'evil_char', 'evil_word', 'shang']].to_csv('../data/final_data_v1_test.csv')

    clf = joblib.load('../model/xss-svm-model.m')
    y_pre = clf.predict(X)

    print(classification_report(y, y_pre))
