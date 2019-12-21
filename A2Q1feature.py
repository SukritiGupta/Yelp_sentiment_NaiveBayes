import json
import string
import math
import time
from sklearn.feature_extraction.text import CountVectorizer
# from utils import getStemmedDocuments

l=[0,0,0,0,0]
vocab={}
n=[0,0,0,0,0]

start=time.time()
with open('train.json', 'r') as f:

    for line in f:
        tweet = json.loads(line)
        s=(int)(tweet['stars'])
        s=s-1
        t=CountVectorizer(ngram_range=(1,2)).build_analyzer()(tweet['text'])
#         t=getStemmedDocuments(t1,False)
        l[s]=l[s]+len(t)
        n[s]=n[s]+1
        for w in t:
            if w=='' or w==' ':
                # print("aaaaaaa")
                break

            if w in vocab:
                vocab[w][s]=vocab[w][s]+1
            else:
                wl=[1,1,1,1,1]
                wl[s]=wl[s]+1
                vocab[w]=wl


print(vocab['perfect'])
v=len(vocab)
print(len(vocab))
print(l)
ld=[0,0,0,0,0]
size=0
for i in [0,1,2,3,4]:
    l[i]+=v
    ld[i]=math.log(l[i])
    print(i)
    size+=n[i]
for i in [0,1,2,3,4]:
    print(i)
    print(n[i]/size)
    n[i]=math.log(n[i]/size)
print(l)
print(n)
print(vocab['perfect'])
logvocab={}
for w,x in vocab.items():
    for i in [0,1,2,3,4]:
#         print(x[i])
#         print(w)
        x[i]=math.log(x[i])-ld[i]
    logvocab[w]=x

correct=0
total=0
# n=[-1.8963699309639557, -2.508278510342461, -2.20936629854321, -1.5147894521793739, -0.8235872882885679]
print(n)
import numpy as np
cm=np.zeros((5,5))
p=[0,0,0,0,0]
# sleep(100)
with open('test.json', 'r') as f:

    for line in f:
        tweet = json.loads(line)
        s=(int)(tweet['stars'])
        # t=tweet['text'].split()
#         t1=CountVectorizer().build_analyzer()(tweet['text'])
#         t=getStemmedDocuments(t1, False)
        t=CountVectorizer(ngram_range=(1,2)).build_analyzer()(tweet['text'])

        
        for i in [0,1,2,3,4]:
            p[i]=n[i]       
        
        # print(n)
        for w in t:
            if w=='' or w==' ':
                # print("aaaaaaa")
                break
            if w in logvocab:
                x=logvocab[w]
                for i in [0,1,2,3,4]:
                    p[i]+=x[i]
        prediction=p.index(max(p))+1
        # print(p)
        if s==prediction:
            correct+=1
        total+=1
        cm[prediction-1][s-1]+=1
        # if prediction!=5:
        #     print(prediction)

print("test")

print(correct)
print(total)
print(correct/total*100)

end=time.time()
print(cm)


# [712, 657, 1836, 8104, 18215]
# 5502016
# [21606745, 11605111, 14551062, 25436445, 38931997]
# 0
# 1
# 2
# 3
# 4
# 0
# 0.15011255029240642
# 1
# 0.08140826216365785
# 2
# 0.10977018800759808
# 3
# 0.21985446985446985
# 4
# 0.43885452968186783
# [27108761, 17107127, 20053078, 30938461, 44434013]
# [-1.8963699309639557, -2.508278510342461, -2.20936629854321, -1.5147894521793739, -0.8235872882885679]
# [712, 657, 1836, 8104, 18215]
# [-1.8963699309639557, -2.508278510342461, -2.20936629854321, -1.5147894521793739, -0.8235872882885679]
# test
# 84618
# 133718
# 63.28093450395609
# [[  1.66810000e+04   4.11900000e+03   1.76700000e+03   8.37000000e+02
#     1.42700000e+03]
#  [  9.12000000e+02   7.90000000e+02   2.14000000e+02   2.10000000e+01
#     2.00000000e+00]
#  [  7.57000000e+02   1.84100000e+03   1.69300000e+03   2.40000000e+02
#     4.00000000e+01]
#  [  1.30900000e+03   3.49800000e+03   9.53100000e+03   1.86250000e+04
#     1.05240000e+04]
#  [  5.10000000e+02   5.90000000e+02   1.32600000e+03   9.63500000e+03
#     4.68290000e+04]]
                