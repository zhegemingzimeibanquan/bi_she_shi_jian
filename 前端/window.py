from tkinter import *
from keras.models import load_model
import joblib
import pandas as pd
from keras.preprocessing import sequence
from url_deal.data_deal import filter3
from url_deal.url_futher import get_url_count, get_len, get_evil_word, get_evil_char, getshan
from v_deal.word_cut import GeneSeg


def deal_lstm(x):
    data_set = pd.DataFrame(columns=['cut_words'])
    data_set = data_set.append([{'cut_words': GeneSeg(x)}], ignore_index=True)
    tokenizer = joblib.load('../model/tokenizer.model')
    X = tokenizer.texts_to_sequences(data_set['cut_words'].values)
    X = sequence.pad_sequences(X, maxlen=350)
    model = load_model('../model/v_model_LSTM.h5')
    y_pre = model.predict(X)
    if y_pre[0][0] <= y_pre[0][1]:
        return "危险"
    else:
        return "安全"


def deal_svm(x):
    data = pd.DataFrame(columns=['len', 'url_count', 'evil_char', 'evil_word', 'shang'])
    data = data.append([{'len': get_len(x),
                         'url_count': get_url_count(x), 'evil_char': get_evil_char(x),
                         'evil_word': get_evil_word(x), 'shang': getshan(x)}], ignore_index=True)
    data = filter3(data)
    clf = joblib.load('../model/xss-svm-model.m')
    y_pre = clf.predict(data)
    if y_pre == 1:
        return "危险"
    else:
        return "安全"


def right_mouse_down(event):
    txt.delete(0.0, END)


def run():
    var = txt.get(0.0, END)
    lb3.configure(text=deal_svm(var))
    lb4.configure(text=deal_lstm(var))


def createNewWindow():
    app = Tk()
    app.title('数据描述')
    app.geometry('700x700')
    var = txt.get(0.0, END)
    lb1 = Label(app, text='长度:  {}'.format(get_len(var)), font=('华文新魏', 16), width=20, height=2)
    lb1.grid(column=0, row=0)
    lb2 = Label(app, text='网址存在:  {}'.format(('是' if (get_url_count(var) == 1) else '否')), font=('华文新魏', 16), width=20,
                height=2)
    lb2.grid(column=0, row=2)
    lb3 = Label(app, text='风险字:  {}'.format(get_evil_char(var)), font=('华文新魏', 16), width=20, height=2)
    lb3.grid(column=0, row=4)
    lb4 = Label(app, text='风险词:  {}'.format(get_evil_word(var)), font=('华文新魏', 16), width=20, height=2)
    lb4.grid(column=0, row=6)
    lb5 = Label(app, text='熵:  {}'.format(getshan(var)), font=('华文新魏', 16), width=20, height=2)
    lb5.grid(column=0, row=8)
    lb6 = Label(app, text='词向量:', font=('华文新魏', 16), width=20, height=2)
    lb6.grid(column=0, row=10)
    data_set = pd.DataFrame(columns=['cut_words'])
    data_set = data_set.append([{'cut_words': GeneSeg(var)}], ignore_index=True)
    tokenizer = joblib.load('../model/tokenizer.model')
    X = tokenizer.texts_to_sequences(data_set['cut_words'].values)
    X = sequence.pad_sequences(X, maxlen=350)
    lb7 = Label(app, text='{}'.format(X[0]), font=('华文新魏', 16))
    lb7.grid(row=11)
    app.mainloop()


if __name__ == '__main__':
    root = Tk()
    root.title('xss攻击识别')
    root.geometry('800x250')
    # root.geometry('240x240')  # 这里的乘号不是 * ，而是小写英文字母 x
    lb = Label(root, text='请输入xss', font=('华文新魏', 16), width=20, height=2)
    lb.grid(column=0, columnspan=5)
    txt = Text(root, height=3, bg='#d3fbfb', relief=SUNKEN)
    txt.grid(column=0, row=2)
    txt.bind('<ButtonPress-1>', right_mouse_down)
    txt.insert(0.3, 'xss here')
    lb1 = Label(root, text='predict on svm:', font=('华文新魏', 16), width=20, height=2)
    lb1.grid(column=0, row=3)
    lb2 = Label(root, text='predict on lstm:', font=('华文新魏', 16), width=20, height=2)
    lb2.grid(column=0, row=4)
    lb3 = Label(root, font=('华文新魏', 16), width=20, height=2)
    lb3.grid(column=4, row=3)
    lb4 = Label(root, font=('华文新魏', 16), width=20, height=2)
    lb4.grid(column=4, row=4)
    btn1 = Button(root, text='submit', command=run)
    btn1.grid(column=2, row=5, columnspan=3)
    btn2 = Button(root, text='describe', command=createNewWindow)
    btn2.grid(column=0, row=5, columnspan=3)
    root.mainloop()
