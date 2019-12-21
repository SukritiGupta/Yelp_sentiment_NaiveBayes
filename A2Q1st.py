import json
import string
import math
import time
from sklearn.feature_extraction.text import CountVectorizer
from utils import getStemmedDocuments

l=[0,0,0,0,0]
vocab={}
n=[0,0,0,0,0]

start=time.time()
with open('train.json', 'r') as f:

    for line in f:
        tweet = json.loads(line)
        s=(int)(tweet['stars'])
        s=s-1
        t1=CountVectorizer().build_analyzer()(tweet['text'])
        t=getStemmedDocuments(t1,False)
        # print(t)
        # print(t1)
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
unchangvocab={}
unchangvocab=vocab

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
        t1=CountVectorizer().build_analyzer()(tweet['text'])
        t=getStemmedDocuments(t1, False)

        
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
# print(end-start)




# [14, 8, 19, 136, 366]
# 31021
# [10843518, 5824327, 7304887, 12777019, 19583362]
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
# [10874539, 5855348, 7335908, 12808040, 19614383]
# [-1.8963699309639557, -2.508278510342461, -2.20936629854321, -1.5147894521793739, -0.8235872882885679]
# [14, 8, 19, 136, 366]
# [-1.8963699309639557, -2.508278510342461, -2.20936629854321, -1.5147894521793739, -0.8235872882885679]
# test
# 63264
# 133718
# 47.31150630431206
# [[  3.49900000e+03   5.89000000e+02   3.10000000e+02   3.32000000e+02
#     4.97000000e+02]
#  [  1.34000000e+02   2.28000000e+02   9.20000000e+01   3.30000000e+01
#     4.50000000e+01]
#  [  1.11000000e+02   2.12000000e+02   3.68000000e+02   2.14000000e+02
#     1.46000000e+02]
#  [  3.70000000e+02   4.67000000e+02   1.28000000e+03   2.68300000e+03
#     1.64800000e+03]
#  [  1.60550000e+04   9.34200000e+03   1.24810000e+04   2.60960000e+04
#     5.64860000e+04]]