import numpy as np
import random

def generate_data(size):
    list=[]
    for i in range(size):
        list.append(np.random.uniform(-1,1))
    output_list=[]
    for x in list:
        if x>=0:
            output_list.append([x,1])
        else:
            output_list.append([x,-1])
    output_sorted_list=sorted(output_list, key = lambda s: s[0])
    return output_sorted_list
    
def add_noise(list,noise_rate):
    r=noise_rate
    output_list=[]
    for x in list:
        if np.random.rand()<noise_rate:
            output_list.append([x[0],x[1]*-1])
        else:
            output_list.append(x)
    return output_list

def evaluate_E(list,threshold,positive=1):
    t=threshold
    err_cnt=0
    for x in list:
        if x[0]<t and x[1]==positive:
            err_cnt+=1
        elif x[0]>=t and x[1]==-1*positive:
            err_cnt+=1
    return err_cnt/len(list)

def find_best_threshold(list):
    threshold_list=[]
    threshold_list.append(-1.0)
    a1=list[0][0]
    for i in range(1,len(list)):
        if list[i][0]!=a1:
            a2=list[i][0]
            threshold_list.append((a1+a2)/2)
            a1=a2
    best_dir=1
    best=-1
    best_E=2
    for t in threshold_list:
        new_E=evaluate_E(list,t)
        if new_E<best_E:
            best_E=new_E
            best=t
            best_dir=1
    for t in threshold_list:
        new_E=evaluate_E(list,t,-1)
        if new_E<best_E:
            best_E=new_E
            best=t
            best_dir=-1
            
    return best,best_dir,best_E
    
def run_n_times(size,noise,times):
    e_out_size=10000
    total=0
    for i in range(times):
        data=generate_data(size)
        noised_data=add_noise(data,noise)
        thres,dir,e_in=find_best_threshold(noised_data)
        e_out_data=generate_data(e_out_size)
        noised_out_data=add_noise(e_out_data,noise)
        e_out=evaluate_E(noised_out_data,thres,dir)
        total+=e_out-e_in
    return total/times
    
def q_16():
    size=2
    noise=0
    times=10000
    result=run_n_times(size,noise,times)
    print(result)
    
def q_17():
    size=20
    noise=0
    times=10000
    result=run_n_times(size,noise,times)
    print(result)
    
def q_18():
    size=2
    noise=0.1
    times=10000
    result=run_n_times(size,noise,times)
    print(result)
    
def q_19():
    size=20
    noise=0.1
    times=10000
    result=run_n_times(size,noise,times)
    print(result)
    
def q_20():
    size=200
    noise=0.1
    times=10000
    result=run_n_times(size,noise,times)
    print(result)


###########  main  #########
q_16()
q_17()
q_18()
q_19()
q_20()

