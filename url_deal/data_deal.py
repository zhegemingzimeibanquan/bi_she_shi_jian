import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def filter1(data):
    min_max_scaler = MinMaxScaler()
    clf = StandardScaler()
    data.drop(data[data['evil_char'] > 9].index, inplace=True)
    data.drop(data[data['evil_word'] > 3].index, inplace=True)
    data.drop(data[data['len'] > 250].index, inplace=True)
    char_x = data[['evil_char', 'evil_word']]
    char_x_done = min_max_scaler.fit_transform(char_x)
    data[['evil_char', 'evil_word']] = char_x_done
    len_x = data[['len', 'shang']]
    len_x_done = clf.fit_transform(len_x)
    data[['len', 'shang']] = len_x_done
    return data


def filter2(data):
    min_max_scaler = MinMaxScaler()
    clf = StandardScaler()
    char_x = data[['evil_char', 'evil_word']]
    char_x_done = min_max_scaler.fit_transform(char_x)
    data[['evil_char', 'evil_word']] = char_x_done
    len_x = data[['len', 'shang']]
    len_x_done = clf.fit_transform(len_x)
    data[['len', 'shang']] = len_x_done
    return data


def filter3(data):
    stander = joblib.load('../model/stander.m')
    min_max_scaler = joblib.load('../model/min_max.m')
    char_x = data[['evil_char', 'evil_word']]
    char_x_done = min_max_scaler.transform(char_x)
    data[['evil_char', 'evil_word']] = char_x_done
    len_x = data[['len', 'shang']]
    len_x_done = stander.transform(len_x)
    data[['len', 'shang']] = len_x_done
    return data


if __name__ == '__main__':
    data = pd.read_csv('../data/train_data.csv')
    # print(data[['len', 'url_count', 'evil_char', 'evil_word', 'shang']].describe())
    df1 = data[['len', 'label']]
    data.drop(data[data['evil_char'] > 9].index, inplace=True)
    data.drop(data[data['evil_word'] > 3].index, inplace=True)
    # print(data[['len', 'url_count', 'evil_char', 'evil_word', 'shang']].describe())
    df2 = data[['len', 'label']]
    df1[df1['label'] == 1]['len'].hist(width=100)
    df1[df1['label'] == 0]['len'].hist(width=50)
    plt.legend(['1', '0'])
    # plt.show()
    df2[df2['label'] == 1]['len'].hist(width=100)
    df2[df2['label'] == 0]['len'].hist(width=50)
    plt.legend(['1', '0'])
    # plt.show()
    data.drop(data[data['len'] > 250].index, inplace=True)
    df3 = data[['len', 'label']]
    df3[df3['label'] == 1]['len'].hist(width=100)
    df3[df3['label'] == 0]['len'].hist(width=50)
    plt.legend(['1', '0'])
    # plt.show()
    min_max_scaler = MinMaxScaler()
    clf = StandardScaler()

    char_x = data[['evil_char', 'evil_word']]
    min_max_scaler.fit(char_x)
    joblib.dump(min_max_scaler, '../model/min_max.m')
    char_x_done = min_max_scaler.transform(char_x)
    data[['evil_char', 'evil_word']] = char_x_done

    len_x = data[['len', 'shang']]
    clf.fit(len_x)
    len_x_done = clf.transform(len_x)
    joblib.dump(clf, '../model/stander.m')
    data[['len', 'shang']] = len_x_done

    print(data[['len', 'url_count', 'evil_char', 'evil_word', 'shang']].describe())
    data[['len', 'url_count', 'evil_char', 'evil_word', 'shang', 'label']].to_csv('../data/final_data_v1.csv')
