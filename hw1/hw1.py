import numpy as np
import random
import copy

def sign(w,data):
    sum=0
    for i in range(len(w)):
        sum+=w[i]*data[i]
    if sum>0:
        return 1.0
    else:
        return -1.0

def correct(w,y,data):
    for i in range(len(w)):
        w[i]=w[i]+y*data[i]
    return w

txt = np.fromfile('hw1_train.dat',sep='\t', dtype=float)

num=int(len(txt)/11)
data=np.zeros((num,11))
y=np.zeros((num))
for i in range(num):
    # value of x_0
    data[i][0]=0
    for j in range(10):
        data[i][j+1]=txt[11*i+j]/4
    y[i]=txt[11*i+10]

num_update=[]
for exp in range(1000):

    w=np.zeros((11),dtype=float)
    order=list(range(num))
    update=0
    iter_num=5

    for iter in range(iter_num):
        w_last=copy.deepcopy(w)
        random.shuffle(order)
        for i in order:
            if sign(w,data[i])!=y[i]:
                w=correct(w,y[i],data[i])
                update+=1
        if(w.all()==w_last.all()):
            break
    
    if(exp%50==0):
        print(exp)
    
    num_update.append(update)
    
print(np.median(num_update))

