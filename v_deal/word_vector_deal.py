import joblib
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import Embedding, LSTM, Dropout, Dense
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from url_deal import data_try as dt
from word_cut import GeneSeg


def text_to_index_array(p_new_dic, p_sen):
    if type(p_sen) == list:
        new_sentences = []
        for sen in p_sen:
            new_sen = []
            for word in sen:
                try:
                    new_sen.append(p_new_dic[word])  # 单词转索引数字
                except:
                    new_sen.append(0)  # 索引字典里没有的词转为数字0
            new_sentences.append(new_sen)
        return np.array(new_sentences)  # 转numpy数组
    else:
        new_sentences = []
        sentences = []
        p_sen = p_sen.split(" ")
        for word in p_sen:
            try:
                sentences.append(p_new_dic[word])  # 单词转索引数字
            except:
                sentences.append(0)  # 索引字典里没有的词转为数字0
        new_sentences.append(sentences)
        return new_sentences


def creat_data():
    data_normal = pd.read_table('../data/train_normal.txt', header=None)
    data_xss = pd.read_table('../data/train_xss.txt', header=None)
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

    data_normal_X = pd.read_table('../data/validation_normal.txt', header=None)
    data_xss_X = pd.read_table('../data/validation_xss.txt', header=None)
    re_data_xss_x = dt.reversed_code(data_xss_X)
    re_data_normal_x = dt.reversed_code(data_normal_X)
    data_set_x = pd.DataFrame(columns=['url', 'cut_words', 'label'])
    data_set_x['url'] = re_data_xss_x
    data_set_x['label'] = 1
    data_set1_x = pd.DataFrame(columns=['url', 'cut_words', 'label'])
    data_set1_x['url'] = re_data_normal_x
    data_set1_x['label'] = 0
    final_data_x = pd.concat([data_set_x, data_set1_x], axis=0, ignore_index=True)
    final_data_x['cut_words'] = final_data['url'].apply(lambda x: GeneSeg(x))

    final_data = pd.concat([final_data, final_data_x], axis=0, ignore_index=True)
    return final_data


def show_train_history(train_history, train, velidation):
    """
    可视化训练过程 对比
    :param train_history:
    :param train:
    :param velidation:
    :return:
    """
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[velidation])
    plt.title("Train History")  # 标题
    plt.xlabel('Epoch')  # x轴标题
    plt.ylabel(train)  # y轴标题
    plt.legend(['train', 'test'], loc='upper left')  # 图例 左上角
    plt.show()


def train_lstm(p_n_symbols, p_X_train, p_y_train, p_X_test, p_y_test, X_test_l):
    # 参数设置
    maxlen = 350  # 文本保留的最大长度
    batch_size = 128  # 训练过程中 每次传入模型的特征数量
    n_epoch = 10  # 迭代次数

    print('创建模型...')
    model = Sequential()
    model.add(Embedding(output_dim=128,  # 输出向量维度
                        input_dim=p_n_symbols,  # 输入向量维度
                        mask_zero=True,  # 使我们填补的0值在后续训练中不产生影响（屏蔽0值）
                        #                    weights=[p_embedding_weights],  # 对数据加权
                        input_length=maxlen))  # 每个特征的长度

    model.add(LSTM(units=128,
                   activation='tanh',
                   recurrent_activation='sigmoid'))
    model.add(Dropout(0.5))  # 每次迭代丢弃50神经元 防止过拟合
    model.add(Dense(units=512,
                    activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=2,  # 输出层1个神经元 1代表正面 0代表负面
                    activation='sigmoid'))
    model.summary()

    print('编译模型...')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print("训练...")
    train_history = model.fit(p_X_train, p_y_train, batch_size=batch_size, epochs=n_epoch,
                              validation_split=0.2,
                              callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.003)])

    print("评估...")
    score, acc = model.evaluate(p_X_test, p_y_test, batch_size=batch_size)
    label = model.predict(p_X_test)
    print('Test score:', score)
    print('Test accuracy:', acc)
    count = 0
    # for (a, b, c) in zip(p_y_test, X_test_l, label):
    #     count = count + 1
    #     if (count > 5):
    #         break
    #     print("原文为：" + "".join(b))
    #     print("预测倾向为", a)
    #     print("真实倾向为", c)

    show_train_history(train_history, 'accuracy', 'val_accuracy')  # 训练集准确率与验证集准确率 折线图
    show_train_history(train_history, 'loss', 'val_loss')  # 训练集误差率与验证集误差率 折线图

    """保存模型"""
    model.save('model/v_model_LSTM.h5')
    print("模型保存成功")


if __name__ == '__main__':
    final_data = creat_data()
    # words_xss, words_normal = word_colle(final_data)
    # xssword = build_dataset(final_data[final_data['label'] == 1]['cut_words'].values, words_xss)
    # normalword = build_dataset(final_data[final_data['label'] == 0]['cut_words'].values, words_normal)
    # word = normalword + xssword
    # # print(word[0:5])
    # label_list = ([0] * len(normalword) + [1] * len(xssword))
    # X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(word, label_list, test_size=0.4)
    # X_validation_l, X_test_l, y_validation_l, y_test_l = train_test_split(X_test_l, y_test_l, test_size=0.5)
    # f = open("model/v_group.pkl", 'rb')  # 预先训练好的
    # index_dict = pickle.load(f)  # 索引字典，{单词: 索引数字}
    # word_vectors = pickle.load(f)  # 词向量, {单词: 词向量(100维长的数组)}
    #
    # n_symbols = len(index_dict) + 1  # 索引数字的个数，因为有的词语索引为0，所以+1
    # embedding_weights = np.zeros((n_symbols, 128))  # 创建一个n_symbols * 128的0矩阵
    #
    # for w, index in index_dict.items():  # 从索引为1的词语开始，用词向量填充矩阵
    #     embedding_weights[index, :] = word_vectors[w]  # 词向量矩阵，第一行是0向量（没有索引为0的词语，未被填充）
    # X_train = text_to_index_array(index_dict, X_train_l)
    # X_test = text_to_index_array(index_dict, X_test_l)
    # X_validation = text_to_index_array(index_dict, X_validation_l)
    #
    # y_train = np.array(y_train_l)  # 转numpy数组
    # y_test = np.array(y_test_l)
    # y_validation = np.array(y_validation_l)
    # maxlen = 0
    # for x in X_train:
    #     if (len(x) > maxlen):
    #         maxlen = len(x)
    # # print("样子： ", X_train_l[0])
    # # print("形状： ", X_train[0])
    # # print("最大长度：", maxlen)
    # X_train = sequence.pad_sequences(X_train, maxlen=250, padding='post', truncating='post')
    # X_test = sequence.pad_sequences(X_test, maxlen=250, padding='post', truncating='post')
    # X_validation = sequence.pad_sequences(X_test, maxlen=250, padding='post', truncating='post')
    # # print(X_train[0])

    tokenizer = Tokenizer(num_words=80, lower=True)
    tokenizer.fit_on_texts(final_data['cut_words'].values)
    word_index = tokenizer.word_index
    print('共有 %s 个不相同的词语.' % len(word_index))

    X = tokenizer.texts_to_sequences(final_data['cut_words'].values)
    # 经过上一步操作后，X为整数构成的两层嵌套list
    X = sequence.pad_sequences(X, maxlen=350)
    # 经过上步操作后，此时X变成了numpy.ndarray
    Y = pd.get_dummies(final_data['label']).values

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    print(X_train.shape)
    joblib.dump(filename='../model/tokenizer.model', value=tokenizer)
    train_lstm(80, X_train, y_train, X_test, y_test, X_test)
