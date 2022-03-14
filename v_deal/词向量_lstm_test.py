import joblib
import pandas as pd
from keras.models import load_model
from keras.preprocessing import sequence
from sklearn.metrics import classification_report
from url_deal import data_try as dt
from word_cut import GeneSeg

if __name__ == '__main__':
    data_normal = pd.read_table('../data/test_normal.txt', header=None)
    data_xss = pd.read_table('../data/test_xss.txt', header=None)
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
    # words_xss, words_normal = word_colle(final_data)
    # xssword = build_dataset(final_data[final_data['label'] == 1]['cut_words'].values, words_xss)
    # normalword = build_dataset(final_data[final_data['label'] == 0]['cut_words'].values, words_normal)
    # word = normalword + xssword
    # # print(word[0:5])
    # label_list = ([0] * len(normalword) + [1] * len(xssword))
    # f = open("model/v_group.pkl", 'rb')  # 预先训练好的
    # index_dict = pickle.load(f)
    # X_l = text_to_index_array(index_dict, word)
    # print(X_l)
    # y = np.array(label_list)
    maxlen = 350
    tokenizer = joblib.load('../model/tokenizer.model')
    X = tokenizer.texts_to_sequences(final_data['cut_words'].values)
    # 经过上一步操作后，X为整数构成的两层嵌套list
    X = sequence.pad_sequences(X, maxlen=maxlen)
    # 经过上步操作后，此时X变成了numpy.ndarray
    Y = pd.get_dummies(final_data['label']).values
    Y_tru=final_data['label']
    # X = sequence.pad_sequences(X_l, maxlen=maxlen, padding='post')
    model = load_model('../model/v_model_LSTM.h5')
    y_pre = model.predict(X)
    y_pre_list=[]
    for x in y_pre:
        if(x[0]>x[1]):
            y_pre_list.append(0)
        else:
            y_pre_list.append(1)
    print("评估...")
    score, acc = model.evaluate(X, Y, batch_size=128)
    print('Test score:', score)
    print('Test accuracy:', acc)
    print(classification_report(Y_tru, y_pre_list))
