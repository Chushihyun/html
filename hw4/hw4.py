import numpy as np
import math
import copy
from _liblinear.python.liblinearutil import *


def data_process():
    txt = np.fromfile('hw4_train.dat',sep='\t', dtype=float)
    num=int(len(txt)/7)
    data=np.zeros((num,6))
    label=np.zeros((num))
    for i in range(num):
        data[i][:]=txt[i*7:i*7+6]
        label[i]=txt[i*7+6]
    return data,label
 
 
def test_data_process():
    txt = np.fromfile('hw4_test.dat',sep='\t', dtype=float)
    num=int(len(txt)/7)
    data=np.zeros((num,6))
    label=np.zeros((num))
    for i in range(num):
        data[i][:]=txt[i*7:i*7+6]
        label[i]=txt[i*7+6]
    return data,label


def data_transform(data):
    num=data.shape[0]
    new_data=np.zeros((num,28))
    for i in range(num):
        new_data[i][0]=1
        new_data[i][1:7]=data[i][0:6]
        new_data[i][7:13]=data[i][0:6]*data[i][0]
        new_data[i][13:18]=data[i][1:6]*data[i][1]
        new_data[i][18:22]=data[i][2:6]*data[i][2]
        new_data[i][22:25]=data[i][3:6]*data[i][3]
        new_data[i][25:27]=data[i][4:6]*data[i][4]
        new_data[i][27:28]=data[i][5:6]*data[i][5]
    return new_data


# lambda=1/2C
# C=1/2*lambda
# if lambda=10^x, C=1/2*10^(-x)

def q_16():
    raw_data,label=data_process()
    data=data_transform(raw_data)

    raw_test_data,test_label=test_data_process()
    test_data=data_transform(raw_test_data)
    
    lamb=[4,2,0,-2,-4]
    new_lamb=[1/20000,1/200,1/2,50,5000]
    print(new_lamb)
    
    prob=problem(label,data)
    best_para=10
    best_error=-1
    for i in range(5):
        tmp='-s 0 -c '+str(new_lamb[i])+' -e 0.000001'
        para=parameter(tmp)
        
        model=train(prob,para)
        p_label, p_acc, p_val=predict(test_label, test_data, model)
        
        print(p_acc)
        
        if best_error==-1:
            best_error=p_acc[0]
            best_para=lamb[i]
        else:
            if p_acc[0]>best_error:
                best_error=p_acc[0]
                best_para=lamb[i]
    print("finished")
    print("-------------------")
    print(best_para)
    print(best_error)
    



def q_17():
    raw_data,label=data_process()
    data=data_transform(raw_data)

    lamb=[4,2,0,-2,-4]
    new_lamb=[1/20000,1/200,1/2,50,5000]
    print(new_lamb)
    
    prob=problem(label,data)
    best_para=10
    best_error=-1
    
    for i in range(5):
        tmp='-s 0 -c '+str(new_lamb[i])+' -e 0.000001'
        para=parameter(tmp)
        
        model=train(prob,para)
        p_label, p_acc, p_val=predict(label, data, model)
        
        print(p_acc)
        
        if best_error==-1:
            best_error=p_acc[0]
            best_para=lamb[i]
        else:
            if p_acc[0]>best_error:
                best_error=p_acc[0]
                best_para=lamb[i]
    print("finished")
    print("-------------------")
    print(best_para)
    print(best_error)



def q_18():

    raw_data,label=data_process()
    data=data_transform(raw_data)
    raw_test_data,test_label=test_data_process()
    test_data=data_transform(raw_test_data)
    
    lamb=[4,2,0,-2,-4]
    new_lamb=[1/20000,1/200,1/2,50,5000]
    print(new_lamb)
    
    train_data=data[:120]
    train_label=label[:120]
    valid_data=data[120:]
    valid_label=label[120:]
    
    prob=problem(train_label,train_data)
    best_para=10
    best_error=-1
    
    for i in range(5):
        tmp='-s 0 -c '+str(new_lamb[i])+' -e 0.000001'
        para=parameter(tmp)
        
        model=train(prob,para)
        p_label, p_acc, p_val=predict(valid_label, valid_data, model)
        
        print(p_acc[0])
        
        if best_error==-1:
            best_error=p_acc[0]
            best_para=i
        else:
            if p_acc[0]>best_error:
                best_error=p_acc[0]
                best_para=i
    print("finished")
    print("-------------------")
    print(best_para)
    print(best_error)
    
    prob=problem(train_label,train_data)
    tmp='-s 0 -c '+str(new_lamb[best_para])+' -e 0.000001'
    para=parameter(tmp)
    model=train(prob,para)
    p_label, p_acc, p_val=predict(test_label, test_data, model)
    print("--------result---------")
    print(lamb[best_para])
    print(p_acc[0])
    


def q_19():

    raw_data,label=data_process()
    data=data_transform(raw_data)
    raw_test_data,test_label=test_data_process()
    test_data=data_transform(raw_test_data)
    
    lamb=[4,2,0,-2,-4]
    new_lamb=[1/20000,1/200,1/2,50,5000]
    print(new_lamb)
    
    train_data=data[:120]
    train_label=label[:120]
    valid_data=data[120:]
    valid_label=label[120:]
    
    prob=problem(train_label,train_data)
    best_para=10
    best_error=-1
    
    for i in range(5):
        tmp='-s 0 -c '+str(new_lamb[i])+' -e 0.000001'
        para=parameter(tmp)
        
        model=train(prob,para)
        p_label, p_acc, p_val=predict(valid_label, valid_data, model)
        
        print(p_acc[0])
        
        if best_error==-1:
            best_error=p_acc[0]
            best_para=i
        else:
            if p_acc[0]>best_error:
                best_error=p_acc[0]
                best_para=i
    print("finished")
    print("-------------------")
    print(best_para)
    print(best_error)
    
    prob=problem(label,data)
    tmp='-s 0 -c '+str(new_lamb[best_para])+' -e 0.000001'
    para=parameter(tmp)
    model=train(prob,para)
    p_label, acc, p_val=predict(test_label, test_data, model)
    print("--------result---------")
    print(lamb[best_para])
    print(acc[0])
    
def q_20():
    raw_data,label=data_process()
    data=data_transform(raw_data)
    lamb=[4,2,0,-2,-4]
    new_lamb=[1/20000,1/200,1/2,50,5000]
    
    data_1=data[:40]
    label_1=label[:40]
    data_2=data[40:80]
    label_2=label[40:80]
    data_3=data[80:120]
    label_3=label[80:120]
    data_4=data[120:160]
    label_4=label[120:160]
    data_5=data[160:200]
    label_5=label[160:200]
    
    prob_1=problem(np.concatenate((label_2,label_3,label_4,label_5), axis=0),np.concatenate((data_2,data_3,data_4,data_5), axis=0))
    prob_2=problem(np.concatenate((label_1,label_3,label_4,label_5), axis=0),np.concatenate((data_1,data_3,data_4,data_5), axis=0))
    prob_3=problem(np.concatenate((label_1,label_2,label_4,label_5), axis=0),np.concatenate((data_1,data_2,data_4,data_5), axis=0))
    prob_4=problem(np.concatenate((label_1,label_2,label_3,label_5), axis=0),np.concatenate((data_1,data_2,data_3,data_5), axis=0))
    prob_5=problem(np.concatenate((label_1,label_2,label_3,label_4), axis=0),np.concatenate((data_1,data_2,data_3,data_4), axis=0))

    best_para=10
    best_error=-1
    
    for i in range(5):
        tmp='-s 0 -c '+str(new_lamb[i])+' -e 0.000001'
        para=parameter(tmp)
        acc=[]
        
        model_1=train(prob_1,para)
        p_label, p_acc, p_val=predict(label_1, data_1, model_1)
        acc.append(p_acc[0])

        model_2=train(prob_2,para)
        p_label, p_acc, p_val=predict(label_2, data_2, model_2)
        acc.append(p_acc[0])
        
        model_3=train(prob_3,para)
        p_label, p_acc, p_val=predict(label_3, data_3, model_3)
        acc.append(p_acc[0])
        
        model_4=train(prob_4,para)
        p_label, p_acc, p_val=predict(label_4, data_4, model_4)
        acc.append(p_acc[0])
        
        model_5=train(prob_5,para)
        p_label, p_acc, p_val=predict(label_5, data_5, model_5)
        print(p_acc)
        acc.append(p_acc[0])

        print("#############")
        print(acc)
        print(np.mean(acc))
        
        if best_error==-1:
            best_error=np.mean(acc)
            best_para=i
        else:
            if np.mean(acc)>best_error:
                best_error=np.mean(acc)
                best_para=i
        
    print("finished")
    print("-------------------")
    print(lamb[best_para])
    print(best_error)
    


### main ###

#q_16()
#q_17()
#q_18()
#q_19()
#q_20()


