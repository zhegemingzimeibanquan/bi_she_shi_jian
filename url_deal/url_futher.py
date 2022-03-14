import math
import re


def get_len(url):
    return len(url)


def get_url_count(url):
    if re.search('(http://)|(https://)', url, re.IGNORECASE):
        return 1
    else:
        return 0


def get_evil_char(url):
    return len(re.findall("[<>,\'\"/]", url, re.IGNORECASE))


def get_evil_word(url):
    return len(re.findall("(alert)|(script)|(prompt)|( )|(onerror)|(onload)|(eval)|(src=)", url, re.IGNORECASE))


def getshan(url):
    tmp_dict = {}
    url_len = len(url)
    for i in range(0, url_len):
        if url[i] in tmp_dict.keys():
            tmp_dict[url[i]] = tmp_dict[url[i]] + 1
        else:
            tmp_dict[url[i]] = 1
    shannon = 0
    for i in tmp_dict.keys():
        p = float(tmp_dict[i]) / url_len
        shannon = shannon - p * math.log(p, 2)
    return shannon


if __name__ == '__main__':
    # text=risottoa"/><iframe src="http://xssed.com">+de+setas&commit=IR
    # b=qq&nk=1003186692&s=100&t=1483368593
    texturl = 'text=risottoa"/><iframe src="http://xssed.com">+de+setas&commit=IR'
    print('url:' + texturl +
          "\nlens:%d" % get_len(texturl) +
          '\ncount:%d' % get_url_count(texturl) +
          '\nevil_char:%d' % get_evil_char(texturl) +
          '\nevil_word:%d' % get_evil_word(texturl) +
          '\nshang:%d' % getshan(texturl))
