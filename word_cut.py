import collections
import logging
import pickle
import re

import nltk
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import word2vec

import data_try as dt


# re.sub(',', '', x)
def GeneSeg(payload):
    payload = re.sub(r'\d+', "0", payload)
    payload = re.sub(r'(http|https)://[a-zA-Z0-9\.@&/#!#\?]+', "http://u", payload)
    r = '''
            (?x)[\w\.]+?\(
            |\)
            |"\w+?"
            |'\w+?'
            |http://\w
            |</\w+>
            |<\w+>
            |<\w+
            |\w+=
            |>
            |[\w\.]+
        '''
    return nltk.regexp_tokenize(payload, r)


def build_dataset(datas, words):
    count = [["UNK", -1]]
    counter = collections.Counter(words)
    count.extend(counter.most_common(3000))
    vocabulary = [c[0] for c in count]
    data_set = []
    for data in datas:
        d_set = []
        for word in data:
            if word in vocabulary:
                d_set.append(word)
            else:
                d_set.append("UNK")
                count[0][1] += 1
        data_set.append(d_set)
    return data_set


def create_dictionaries(model):
    """
    创建词语字典，并返回word2vec模型中词语的索引，词向量
    :type mobjectobject
    """
    gensim_dict = Dictionary()  # 创建词语词典
    gensim_dict.doc2bow(model.wv.index_to_key, allow_update=True)

    w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # 词语的索引，从1开始编号
    w2vec = {word: model.wv[word] for word in w2indx.keys()}  # 词语的词向量
    return w2indx, w2vec


def word_colle(data):
    xss = []
    normal = []
    for row in data[data['label'] == 1]['cut_words'].values:
        xss = xss + row
    # print(xss)
    for row in data[data['label'] == 0]['cut_words'].values:
        normal = normal + row
    # word_counts = collections.Counter(xss)  # 对分词做词频统计
    # word_counts_xss_top = word_counts.most_common(30)  # 获取前10最高频的词
    # print(word_counts_xss_top)  # 输出检查
    # word_counts1 = collections.Counter(normal)  # 对分词做词频统计
    # word_counts_normal_top = word_counts1.most_common(50)  # 获取前10最高频的词
    # # print(word_counts_normal_top)  # 输出检查
    # word_counts = collections.Counter(xss)  # 对分词做词频统计
    # word_counts_xss_top = word_counts.most_common(3000)  # 获取前3000最高频的词
    # for word in xss:
    #     if ((word in word_counts_xss_top) == False):
    #         xss = ['UKN' if i == word else i for i in xss]
    return xss, normal


if __name__ == '__main__':
    data_normal = pd.read_table('data/train_normal.txt', header=None)
    data_xss = pd.read_table('data/train_xss.txt', header=None)
    re_data_xss = dt.reversed_code(data_xss)
    re_data_normal = dt.reversed_code(data_normal)
    data_set = pd.DataFrame(columns=['url', 'cut_words', 'label'])
    data_set['url'] = re_data_xss
    data_set['label'] = 1
    data_set1 = pd.DataFrame(columns=['url', 'cut_words', 'label'])
    data_set1['url'] = re_data_normal
    data_set1['label'] = 0
    final_data = pd.concat([data_set, data_set1], axis=0, ignore_index=True)
    final_data['cut_words'] = final_data['url'].apply(lambda x: GeneSeg(x))
    # xssword, normalword = word_count(final_data)
    # with open("data/xss_split.txt", "w") as output:
    #     output.write(str(xssword))
    # with open("data/normal_split.txt", "w") as output1:
    #     output1.write(str(normalword))
    # word_counts = collections.Counter(xssword)  # 对分词做词频统计
    # word_counts_xss_top = word_counts.most_common(30)  # 获取前10最高频的词
    # print(word_counts_xss_top)  # 输出检查
    # print(final_data)
    # print(type(final_data['cut_words'].values))
    words_xss, words_normal = word_colle(final_data)
    xssword = build_dataset(final_data[final_data['label'] == 1]['cut_words'].values, words_xss)
    normalword = build_dataset(final_data[final_data['label'] == 0]['cut_words'].values, words_normal)
    word = xssword + normalword
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)  # 将日志输出到控制台
    model = word2vec.Word2Vec(word, vector_size=128, sg=0, window=10)
    model.save('模型/word2vec_v1.model')
    index_dict, word_vectors = create_dictionaries(model)
    print(index_dict)
    print('\n')
    # print(word_vectors)
    # model = word2vec.Word2Vec.load('模型/word2vec_v1.model')
    output = open("模型/词向量组.pkl", 'wb')
    pickle.dump(index_dict, output)  # 索引字典
    pickle.dump(word_vectors, output)  # 词向量字典
    output.close()
