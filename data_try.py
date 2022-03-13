import re
from urllib import parse

import pandas as pd

import url_futher as myfu


def reversed_code(url_string_data):
    reversed_data = url_string_data[0].apply(lambda x: parse.unquote(x))
    return reversed_data

def step1(xss, normal):
    re_xss = reversed_code(xss)
    re_normal = reversed_code(normal)
    set = pd.DataFrame(columns=['url', 'len', 'url_count', 'evil_char', 'evil_word', 'shang', 'label'])
    set['url'] = re_xss
    set['len'] = set['url'].apply(lambda x: myfu.get_len(x))
    set['url_count'] = set['url'].apply(lambda x: myfu.get_url_count(x))
    set['evil_char'] = set['url'].apply(lambda x: myfu.get_evil_char(x))
    set['evil_word'] = set['url'].apply(lambda x: myfu.get_evil_word(x))
    set['shang'] = set['url'].apply(lambda x: myfu.getshan(x))
    set['label'] = 1
    set['url'].apply(lambda x: re.sub(',', '', x))

    set1 = pd.DataFrame(columns=['url', 'len', 'url_count', 'evil_char', 'evil_word', 'shang', 'label'])
    set1['url'] = re_normal
    set1['len'] = set1['url'].apply(lambda x: myfu.get_len(x))
    set1['url_count'] = set1['url'].apply(lambda x: myfu.get_url_count(x))
    set1['evil_char'] = set1['url'].apply(lambda x: myfu.get_evil_char(x))
    set1['evil_word'] = set1['url'].apply(lambda x: myfu.get_evil_word(x))
    set1['shang'] = set1['url'].apply(lambda x: myfu.getshan(x))
    set1['label'] = 0
    set1['url'].apply(lambda x: re.sub(',', '', x))
    final = pd.concat([set, set1], axis=0)
    return final


if __name__ == '__main__':
    data_normal = pd.read_table('data/train_normal.txt', header=None)
    data_xss = pd.read_table('data/train_xss.txt', header=None)

    re_data_xss = reversed_code(data_xss)
    re_data_normal = reversed_code(data_normal)
    # print(re_data_xss[0:9])
    # print(re_data_normal[0:9])

    data_set = pd.DataFrame(columns=['url', 'len', 'url_count', 'evil_char', 'evil_word', 'shang', 'label'])
    data_set['url'] = re_data_xss
    data_set['len'] = data_set['url'].apply(lambda x: myfu.get_len(x))
    data_set['url_count'] = data_set['url'].apply(lambda x: myfu.get_url_count(x))
    data_set['evil_char'] = data_set['url'].apply(lambda x: myfu.get_evil_char(x))
    data_set['evil_word'] = data_set['url'].apply(lambda x: myfu.get_evil_word(x))
    data_set['shang'] = data_set['url'].apply(lambda x: myfu.getshan(x))
    data_set['label'] = 1
    data_set['url'].apply(lambda x: re.sub(',', '', x))

    data_set1 = pd.DataFrame(columns=['url', 'len', 'url_count', 'evil_char', 'evil_word', 'shang', 'label'])
    data_set1['url'] = re_data_normal
    data_set1['len'] = data_set1['url'].apply(lambda x: myfu.get_len(x))
    data_set1['url_count'] = data_set1['url'].apply(lambda x: myfu.get_url_count(x))
    data_set1['evil_char'] = data_set1['url'].apply(lambda x: myfu.get_evil_char(x))
    data_set1['evil_word'] = data_set1['url'].apply(lambda x: myfu.get_evil_word(x))
    data_set1['shang'] = data_set1['url'].apply(lambda x: myfu.getshan(x))
    data_set1['label'] = 0
    data_set1['url'].apply(lambda x: re.sub(',', '', x))
    # (data_set1)
    final_data = pd.concat([data_set, data_set1], axis=0)
    #final_data.to_csv('data/train_data1.csv')
    #print(final_data)



