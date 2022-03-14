#!/usr/bin/env python
# encoding: utf-8 """
import string
import urllib
from urllib.parse import urlparse

from bs4 import BeautifulSoup


# 获取源代码
def crawlcontent(url):
    pattern1 = '<.*?(href=".*?").*?'
    headers = {'User-Agent',
               'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.186 Safari/537.36'}
    opener = urllib.request.build_opener()
    opener.addheaders = [headers]
    data = opener.open(url).read().decode('utf-8')
    # f = open("C:/Users/Administrator/PycharmProjects/URL/PhishTankURLs.html", "w+",encoding='utf-8')
    # f.write(data)
    # f.close()
    return data


def html(url):
    f = open("C:/Users/Administrator/PycharmProjects/URL/PhishTankURLs.html", "w+", encoding='utf-8')
    f.write(crawlcontent(url))
    f.close()
    p = open("C:/Users/Administrator/PycharmProjects/URL/PhishTankURLs.txt", "w+", encoding='utf-8')
    p.write(crawlcontent(url))


def geturl(url):
    soup = BeautifulSoup(crawlcontent(url), 'html.parser')
    for x in soup.findAll('a'):
        return x.attrs['href']

    html = urllib.request.urlopen(crawlcontent(url)).read()
    soup = BeautifulSoup(html)
    links = soup.findAll('a')
    return links


# 提取主机名
def hostname(url):
    parsed_result = urlparse(url)
    a = parsed_result.netloc

    return (a)


# 提取域名
def domain(url):
    host = hostname(url)
    domain = '.'.join(host.split('.')[-3:])
    return (domain)


# 获取URL长度
def getLength(url):
    length = len(url)

    return length


# 是否具有@符合
def aite(url):
    fuhao = '@'
    if fuhao in url:
        return 1
    else:
        return 0


# 是否路径中含有域名
def pathhasdomin(url):
    d = hostname(url)
    parsed_result = urlparse(url)
    a = parsed_result.path
    if d in a:
        return 1
    else:
        return 0


# 是否含有敏感词
def sensitiveword(url):
    i = 0
    flag = 0
    sensitive = ['secure', 'account', 'webscr', 'login', 'ebayiaphi', 'signin', 'banking', 'confirm']
    for i in range(len(sensitive)):
        if sensitive[i] in url:
            flag = 1
            break
        else:
            flag = 0
            continue
    if flag == 1:
        return 1
    else:
        return 0


# 是否含有错误端口
def mistakePort(url):
    url = hostname(url)
    if ":" in url:
        return 1
    else:
        return 0


# 获取自//之后的内容
def removeURLHeader(url):
    return url.split('//', 1)[-1]


# 获取路径
def getPath(url):
    url = removeURLHeader(url)
    if '/' in url:

        startpos = url.index('/') + 1
        moweipos = url.rindex('/')
        suburl = url[startpos:moweipos]
        return suburl
    else:
        return ""


# 是(1)否(0)
def brandname(url):  # 商标名
    i = 0
    flag = 0
    url = getPath(url)
    brandnamelist = ['53.com', 'Chase', 'Microsoft', 'ANZ', 'Citibank', 'Paypal', 'AOL', 'eBay', 'USBank', 'Banamex',
                     'E-Gold', 'Visa', 'Bankofamerica', 'Google', 'Warcraft', 'Barclays', 'HSBC', 'Westpac',
                     'battle.net', 'LIoyds', 'Yahoo']
    for i in range(len(brandnamelist)):
        if brandnamelist[i] in url:
            flag = 1
            break
        else:
            flag = 0
            continue
    if flag == 1:
        return 1
    else:
        return 0


# 文件名
def getFile(url):
    startpos = url.rindex('/') + 1
    url = url[startpos:]

    return url


# 二级域名
def getUrlSubDomain(url):
    host = hostname(url)
    subDomain = '.'.join(host.split('.', 2)[1:-1])

    return subDomain


# 含数字个数
def getDigitsCount(url):
    i = 0
    count = 0
    for i in range(len(url)):
        if url[i] in string.digits:
            count += 1

    return count


# 大写字母个数
def getCountUpcase(url):
    i = 0
    count = 0
    for i in range(len(url)):
        if url[i] in string.ascii_uppercase:
            count += 1

    return count


# 前缀个数
def getPrefixCount(url):
    prefix = ['_', '-']
    i = 0
    count = 0
    for i in range(len(url)):
        if url[i] in prefix:
            count += 1

    return count


# 主机名项数
def termcout(url):
    url = hostname(url)
    url = url.split('.')
    return len(url)


# URL中数字-字符转换频次
def ZhuanHuanPingci(url):
    urlwhole = removeURLHeader(url)
    count = 0
    length = len(urlwhole)
    for i in range(length):
        if urlwhole[i] in string.digits and i + 1 < length and (
                urlwhole[i + 1] in string.ascii_lowercase or urlwhole[i + 1] in string.ascii_uppercase):

            count += 1
        else:
            if (urlwhole[i] in string.ascii_lowercase or urlwhole[i] in string.ascii_uppercase) and i + 1 < length and \
                    urlwhole[
                        i + 1] in string.digits:
                count += 1

    return count


# 是否含有关键词
def targetword(url):
    url = removeURLHeader(url)
    # startpos = url.index('/') + 1
    url = hostname(url)
    url = domain(url)
    # url = url[startpos:]
    normalDomain = ['paypal.com', 'aol.com', 'qq.com', 'made-in-china.com', 'google.com', 'facebook.com', 'yahoo.com',
                    'live.com', 'dropbox.com', 'wellsfargo.com', 'cmr.no', 'academia.edu', 'regions.com',
                    'shrinkthislink.com', 'maximumasp.com', 'popularenlinea.com', 'readydecks.com', 'meezanbank.com',
                    'vencorex.com', 'ketthealth.com', 'obhrmanager.com', 'bluehost.com', 'msubillings.edu',
                    'genxgame.com', 'gripeezoffer.com', 'bek-intern.de', 'ebay.com', 'chase.com', 'revoluza.com',
                    'dhl.com', 'flexispy.com', 'att.com', 'uwsp.edu', 'match.com', 'alnoorhospital.com', 'ourtime.com']
    if url in normalDomain:
        return 1
    else:
        return 0


if __name__ == '__main__':
    url = 'text=risottoa"/><iframe src="http://xssed.com">+de+setas&commit=IR'
    # text=risottoa"/><iframe src="http://xssed.com">+de+setas&commit=IR
    # b=qq&nk=1003186692&s=100&t=1483368593
    content = []
    tezhen = []
    tezhen.append(
        {'主机名': hostname(url), 'URL长度': getLength(url), '是否含有@符号（是(1)否(0)）': aite(url), '路径中是否含有域名': pathhasdomin(url),
         '主机名长度': len(hostname(url)), '域名': domain(url), '是否含有敏感词': sensitiveword(url), '错误端口': mistakePort(url),
         '商标名称': brandname(url), '文件名': getFile(url), '获取二级域名': getUrlSubDomain(url), '含数字个数': getDigitsCount(url),
         '大写字母个数': getCountUpcase(url), '前缀个数': getPrefixCount(url), '主机名项数': termcout(url),
         '数字-字符转换频次': ZhuanHuanPingci(url), '是否含有关键词': targetword(url)})
    content.append({'URL': url, '特征': tezhen})
    print(content)
    # with open('C:/Users/Administrator/PycharmProjects/URL/url特征.json', 'w') as fp:
    #    json.dump(content, fp=fp, indent=4, ensure_ascii=False, )
