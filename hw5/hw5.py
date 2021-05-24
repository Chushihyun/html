import numpy as np
import copy
import sys
sys.path.append("./libsvm/python")
from svmutil import *
import random
from sklearn.model_selection import train_test_split


def problem15(x,y):
    l=copy.deepcopy(y)
    for i in range(len(y)):
        if y[i]==3.0:
            l[i]=1
        else:
            l[i]=-1
    m = svm_train(l,x,'-c 10 -t 0 -s 0 -q')
    coef=np.array(m.get_sv_coef())[:,0]
    A=np.array(m.get_sv_indices())
    sv=x[np.array(m.get_sv_indices())-1]
    w=sv.T@coef
    print(np.linalg.norm(w))

def run_16(x,y,label):
    l=copy.deepcopy(y)
    for i in range(len(y)):
        if y[i]==label:
            l[i]=1
        else:
            l[i]=-1
    m=svm_train(l,x,'-c 10 -d 2 -t 1 -g 1 -r 1 -q')
    ACC, MSE, SCC=svm_predict(l,x,m)

    

def problem16(x,y):
    for i in [1,2,3,4,5]:
        run_16(x,y,i)

def run_17(x,y,label):
    l=copy.deepcopy(y)
    for i in range(len(y)):
        if y[i]==label:
            l[i]=1
        else:
            l[i]=-1
    m=svm_train(l,x,'-c 10 -d 2 -t 1 -g 1 -r 1 -q')
    print(m.l)

def problem17(x,y):
    for i in [1,2,3,4,5]:
        run_17(x,y,i)

def run_18(x,y,C,x_test,y_test):
     m = svm_train(y,x,'-c %f -t 2 -g 10 -q'%(C))
     ACC=svm_predict(y_test,x_test,m)


def problem18(x,y,x_test,y_test):
    l=copy.deepcopy(y)
    for i in range(len(y)):
        if y[i]==6:
            l[i]=1
        else:
            l[i]=-1
            
    l_test=copy.deepcopy(y_test)
    for i in range(len(y_test)):
        if y_test[i]==6:
            l_test[i]=1
        else:
            l_test[i]=-1
    
    for i in [-2,-1,0,1,2]:
        run_18(x,l,10**i,x_test,l_test)
   
def run_19(x,y,r,x_test,y_test):
     m = svm_train(y,x,'-c 0.1 -t 2 -g %f -q'%(r))
     ACC=svm_predict(y_test,x_test,m)


def problem19(x,y,x_test,y_test):
    l=copy.deepcopy(y)
    for i in range(len(y)):
        if y[i]==6:
            l[i]=1
        else:
            l[i]=-1
    l_test=copy.deepcopy(y_test)
    for i in range(len(y_test)):
        if y_test[i]==6:
            l_test[i]=1
        else:
            l_test[i]=-1
                
    for i in [-1,0,1,2,3]:
        run_19(x,l,10**i,x_test,l_test)
        
def run_20(x_train,y_train,r,x_valid,y_valid):
    m = svm_train(y_train,x_train,'-c 0.1 -t 2 -g %f -q'%(r))
    label, acc, val=svm_predict(y_valid,x_valid,m)
    return acc[0]

def problem20(x,y):
    l=copy.deepcopy(y)
    for i in range(len(y)):
        if y[i]==6:
            l[i]=1
        else:
            l[i]=-1
    
    record=np.zeros(5)
    for again in range(1000):
        x_train,x_valid,y_train,y_valid=train_test_split(x,l,test_size=200/len(l))
        acc_best=-1
        i_best=-1
        for i in [-1,0,1,2,3]:
            acc=run_20(x_train,y_train,10**i,x_valid,y_valid)
            if acc>acc_best:
                acc_best=acc
                i_best=i
        record[i_best+1]+=1
        print(record)
    #print(record)

## main ##
y, x = svm_read_problem("train.txt",True)
y_test, x_test=svm_read_problem("test.txt",True)
"""
print("problem 15")
problem15(x,y)
print("problem 16")
problem16(x,y)
print("problem 17")
problem17(x,y)
print("problem 18")
problem18(x,y,x_test,y_test)
print("problem 19")
problem19(x,y,x_test,y_test)
"""
print("problem 20")
problem20(x,y)
