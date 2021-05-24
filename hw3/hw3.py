import numpy as np
import random
import math
import copy

random.seed()

def data_process():
    txt = np.fromfile('hw3_train.dat',sep='\t', dtype=float)
    num=int(len(txt)/11)
    data=np.zeros((num,11))
    label=np.zeros((num))
    for i in range(num):
        data[i][0]=1
        data[i][1:11]=txt[i*11:i*11+10]
        label[i]=txt[i*11+10]
    return data,label
    
def test_data_process():
    txt = np.fromfile('hw3_test.dat',sep='\t', dtype=float)
    num=int(len(txt)/11)
    data=np.zeros((num,11))
    label=np.zeros((num))
    for i in range(num):
        data[i][0]=1
        data[i][1:11]=txt[i*11:i*11+10]
        label[i]=txt[i*11+10]
    return data,label


def linear_regression(data,label):
    num=data.shape[0]
    data_tran=data.transpose()
    A=np.matmul(np.linalg.inv(np.matmul(data_tran,data)),data_tran)
    w=np.matmul(A,label)
    return w


def theta(n):
    return 1/(1+math.exp(-n))


def E_in(data,label,w,type):
    # type: square,cross_entropy,zero_one
    num=data.shape[0]
    if type=="square":
        A=np.matmul(data,w)
        error=np.dot(A-label,A-label)
        return error/num
        
    if type=="cross_entropy":
        error=0
        for i in range(num):
            A=-1*label[i]*np.dot(w,data[i])
            error+=math.log(1+math.exp(A))
        return error/num

    if type=="zero_one":
        error=0
        for i in range(num):
            A=label[i]*np.dot(w,data[i])
            if A<0:
                error+=1
        return error/num
    
    
def sgd_linear(data,label,error_thres):
    num=data.shape[0]
    learning_rate=0.001
    w=np.zeros((11))
    iter=0
    while(E_in(data,label,w,"square")>error_thres):
        n=random.randrange(num)
        w+=learning_rate*2*(label[n]-np.dot(w,data[n]))*data[n]
        iter+=1
    return iter
 
 
def sgd_logistic(data,label,w=None):
    if w is None:
        w=np.zeros((11))
    w=copy.deepcopy(w)
    num=data.shape[0]
    learning_rate=0.001
    for i in range(500):
        n=random.randrange(num)
        A=-1*label[n]*np.dot(w,data[n])
        B=label[n]*data[n]
        a=learning_rate*theta(A)*B
        w+=a
    return w

def data_transform(data,q):
    num=data.shape[0]
    new_data=np.zeros((num,1+10*q))
    for i in range(num):
        new_data[i][0]=data[i][0]
        for j in range(q):
            for k in range(10):
                new_data[i][10*j+k+1]=data[i][k+1]**(j+1)
    return new_data
    

def q_14():
    data,label=data_process()
    w_lin=linear_regression(data,label)
    error=E_in(data,label,w_lin,"square")
    print(error)


def q_15():
    data,label=data_process()
    w_lin=linear_regression(data,label)
    error=E_in(data,label,w_lin,"square")
    iter_count=[]
    for i in range(1000):
        iter=sgd_linear(data,label,1.01*error)
        iter_count.append(iter)
    print(np.mean(iter_count))
    
        
def q_16():
    data,label=data_process()
    E_count=[]
    for i in range(1000):
        w=sgd_logistic(data,label)
        e=E_in(data,label,w,"cross_entropy")
        E_count.append(e)
    print(np.mean(E_count))
    

def q_17():
    data,label=data_process()
    w_lin=linear_regression(data,label)
    E_count=[]
    for i in range(1000):
        w=sgd_logistic(data,label,w_lin)
        e=E_in(data,label,w,"cross_entropy")
        E_count.append(e)
    print(np.mean(E_count))


def q_18():
    data,label=data_process()
    test_data,test_label=test_data_process()
    w_lin=linear_regression(data,label)
    e_in=E_in(data,label,w_lin,"zero_one")
    e_out=E_in(test_data,test_label,w_lin,"zero_one")
    print(abs(e_in-e_out))


def q_19():
    data,label=data_process()
    test_data,test_label=test_data_process()
    new_data=data_transform(data,3)
    new_test_data=data_transform(test_data,3)
    
    w_lin=linear_regression(new_data,label)
    e_in=E_in(new_data,label,w_lin,"zero_one")
    e_out=E_in(new_test_data,test_label,w_lin,"zero_one")
    print(abs(e_in-e_out))
    
def q_20():
    data,label=data_process()
    test_data,test_label=test_data_process()
    new_data=data_transform(data,10)
    new_test_data=data_transform(test_data,10)

    w_lin=linear_regression(new_data,label)
    e_in=E_in(new_data,label,w_lin,"zero_one")
    e_out=E_in(new_test_data,test_label,w_lin,"zero_one")
    print(abs(e_in-e_out))

### main ###

q_14()
q_15()
q_16()
q_17()
q_18()
q_19()
q_20()

