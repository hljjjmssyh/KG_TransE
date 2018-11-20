# import json
# import requests
# import time
#
# import random
import codecs
#
#
# # with open('prediction.json',encoding='utf-8') as f:
# #     b = f.readlines()
# #     dic = []
# #     for i in b:
# #         j = eval(i)
# #         print(j)
# #         del j['news_comment']
# #         dic.append(j)
# #     with open('rep.json','w') as s:
# #         json.dump(dic,s)
#
#
#
# def find_return(text):
#     url = 'https://bosonnlp.com/analysis/sentiment?analysisType='
#     d = {'data': '真替他难受'}
#     # prox = {"http": "http://10.10.1.10:3128", "https": "http://10.10.1.10:1080", }
#     d['data'] = text
#     proxies = {'https': 'https://220.191.15.20:3456'}
#     r = requests.post(url, data=d)
#     r.connection.close()
#     # headers = {
#     #     'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36'
#     # }
#     #
#     # res = requests.post('https://bosonnlp.com/analysis/sentiment?analysisType=', data=d,proxies=proxies)
#     return float((r.text.split(',')[0][2:]))
#
#
# def judge(dict):
#     a = find_return(dict['news_comment'])
#     if dict['polarity'] == 'neutral':
#         if 0.65 > a > 0.35:
#             return True
#         else:
#             return False
#     elif dict['polarity'] == 'positive':
#         if a > 0.85:
#             return True
#         else:
#             return False
#     else:
#         if a < 0.15:
#             return True
#         else:
#             return False
#
#
# with open('stageB_data_train.json', encoding='utf-8') as f:
#     with open('z.json', 'w',encoding='utf-8') as s:
#         b = f.readlines()
#         dict = []
#         z = 0
#         try:
#             for i in b[1078:]:
#                 j = eval(i)
#                 print(z)
#                 z += 1
#                 time.sleep(0.5)
#                 if judge(j):
#                     dict.append(i)
#         finally:
#             s.writelines(dict)
#
txt='pos.0.txt'
file = codecs.open(txt,'r',encoding='gb2312')
listlist=file.readlines()
print(listlist)
