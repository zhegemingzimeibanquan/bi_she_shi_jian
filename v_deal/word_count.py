import matplotlib.pyplot as plt

from v_deal.word_vector_deal import creat_data

if __name__ == '__main__':
    final_data = creat_data()
    words = final_data['cut_words'].values
    diccount = dict()
    count = 0
    for vars in words:
        for var in vars:
            if (var not in diccount):
                diccount[var] = 1  # 第一遍字典为空 赋值相当于 i=1，i为words里的单词
                count = count + 1
                # print(diccount)
            else:
                diccount[var] = diccount[var] + 1
    diccount = sorted(diccount.items(), key=lambda x: x[1], reverse=True)
    diccount = dict(diccount)
    y = []
    for temp_x, temp_y in diccount.items():
        y.append(temp_y)
    # print(x,y)
    y = y[0:100]
    x = range(len(y))
    # plt.plot(range(len(y)), y)
    plt.plot(x, y, marker='o', mec='r', mfc='w')
    plt.show()
