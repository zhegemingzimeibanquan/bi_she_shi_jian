import os

import numpy as np
import pandas as pd
import tensorflow as tf
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.layers import LSTM, Dense
from tensorflow.python.keras.models import Sequential


def build_model(hidden_layers,
                layer_size,
                learning_rate):
    model = keras.models.Sequential([
        LSTM(layer_size, input_shape=(1, 5), activation='selu', return_sequences=True)
    ])

    for _ in range(hidden_layers):
        model.add(LSTM(layer_size, activation='selu', return_sequences=True))
    model.add(LSTM(layer_size, return_sequences=False))

    model.add(Dense(2, activation='relu'))

    # 定义我们自己的  学习率 优化器
    optimizer1 = optimizers.adam_v2.Adam(learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer1, metrics=["accuracy"])

    return model


if __name__ == '__main__':
    data = pd.read_csv('../data/final_data_v1.csv')
    X = data[['len', 'url_count', 'evil_char', 'evil_word', 'shang']]
    X = X.values.reshape(32266, 1, 5)
    y = pd.get_dummies(data['label']).values
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.4, random_state=0)
    # model = Sequential()
    # model.add(LSTM(32, return_sequences=True,
    #                input_shape=(1, 5)))  # 返回维度为 32 的向量序列
    # model.add(LSTM(32, return_sequences=True))  # 返回维度为 32 的向量序列
    # model.add(LSTM(32))  # 返回维度为 32 的单个向量
    # model.add(Dense(2, activation='relu'))
    #
    # model.compile(loss='binary_crossentropy',
    #               optimizer='rmsprop',
    #               metrics=['accuracy'])
    # model.fit(train_x, train_y,
    #           batch_size=64, epochs=10,
    #           )
    # y_pred = model.predict(test_x)
    # Y_tru = []
    # y_pre_list = []
    # for x in y_pred:
    #     if (x[0] > x[1]):
    #         y_pre_list.append(0)
    #     else:
    #         y_pre_list.append(1)
    # for x in test_y:
    #     if (x[0] > x[1]):
    #         Y_tru.append(0)
    #     else:
    #         Y_tru.append(1)
    # print(classification_report(Y_tru, y_pre_list))

    ###定义回调函数
    log_dir = './search_housing_LSTM_logs'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    save_model_dir = './model'
    if not os.path.exists(save_model_dir):
        os.mkdir(save_model_dir)
    save_model_file = os.path.join(save_model_dir, 'search_housing_LSTM.h5')

    callback1 = [
        tf.keras.callbacks.TensorBoard(log_dir),
        tf.keras.callbacks.ModelCheckpoint(save_model_file, save_best_only=True, monitor='val_loss'),
        tf.keras.callbacks.EarlyStopping(patience=5, min_delta=1e-2)
    ]

    sklearn_model = KerasClassifier(
        model=build_model, hidden_layers=1, layer_size=30, learning_rate=3e-3
        , epochs=20, callbacks=callback1
    )

    from scipy.stats import reciprocal

    # 定义搜索空间
    param_distribution = {
        'hidden_layers': [1, 2, 3, 4, 5],
        'layer_size': np.arange(1, 100),
        'learning_rate': reciprocal(1e-4, 1e-2)
    }

    from sklearn.model_selection import RandomizedSearchCV

    # cross_calidation: 训练分成 n 分， 使用 n-1分 训练， 1分测试
    # cv
    # n_jobs=最大并行数
    # 保留最好的多少个 参数组合
    random_search_cv = RandomizedSearchCV(
        sklearn_model, param_distribution,
        n_iter=10, n_jobs=7,
        cv=3)
    random_search_cv.fit(train_x, train_y, validation_data=(test_x, test_y))

    # 查看做好的参数
    print(random_search_cv.best_params_)
    print(random_search_cv.best_score_)
    print(random_search_cv.best_estimator_)

    # 获得最好的模型
    model = random_search_cv.best_estimator_
    y_pre = model.predict(test_x)
    Y_tru = []
    y_pre_list = []
    for x in y_pre:
        if x[0] > x[1]:
            y_pre_list.append(0)
        else:
            y_pre_list.append(1)
    for x in test_y:
        if x[0] > x[1]:
            Y_tru.append(0)
        else:
            Y_tru.append(1)
    print(classification_report(Y_tru, y_pre_list))
    loss, acc = model.evaluate(test_x, test_y)
    print("loss:", loss)
    print("acc:", acc)
