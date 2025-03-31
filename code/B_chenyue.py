import heapq
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def build_heap(N: int, seed: int = None) -> np.ndarray:
    """
    输出：heap: 以一维数组形式存储的完整堆
    参数：
        N: 堆的层数
        seed: 随机数种子，默认为 None
    """
    if seed is not None:
        np.random.seed(seed)  # 设置随机数种子
    length = N * (N + 1) // 2  
    heap = np.random.rand(length)  
    return heap

def dui(n):#随机生成一个n层堆
    r=[]
    for i in range(int((n+1)*(2+n)/2)):
        x=random.random()
        r.append(x)
    heapq.heapify(r)
    return r

def oneround(s,t,num):
    num2=[]
    for i in range(len(s)):
        if t[i]<=t[i+1]:
            s[i]+=t[i]
            num2.append(num[i])
        else:
            s[i]+=t[i+1]
            num2.append(num[i+1]) 
    return s,num2
        

def findmin(n):
    r=build_heap(n,seed=42)
    s=r[int(n*(n+1)/2):int((n+2)*(n+1)/2)]
    num=list(range(0,n+1))
    for i in np.arange(n,0,-1):
        t=r[int(i*(i-1)/2):int((i)*(i+1)/2)]
        s,num=oneround(t,s,num)
    return s,num
m=range(3,40)
#n=100
x,y,std=[],[],[]
std2=[]
for n in m:
    for i in range(10000):
        s,num=findmin(n)
        x.append(num[0])
        #y.append(s[0])
    x2=[i**2 for i in x]
    x_mean=np.mean(x)
    x2_mean=np.mean(x2)
    std2.append(np.std(x))
    x_std=np.sqrt(x2_mean-x_mean**2)
    std.append(x_std)

plt.plot(m,std,marker='o',label='w')
plt.plot(m,std2,marker='o',label='std')
plt.xlabel('n')
plt.ylabel('w')
plt.legend()
plt.show()


plt.plot(m,std,marker='o',label='w')
plt.plot(m,std2,marker='o',label='std')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('n')
plt.ylabel('w')
plt.legend()
plt.show()
x,y=[],[]
n=50
for i in range(10000):
    s,num=findmin(n)
    x.append(num[0])
    y.append(s[0])

counts=Counter(x)
value=list(counts.values())
num=list(counts.keys())
fre=np.array(value)/sum(value)
plt.bar(num,fre,width=0.5,align='center')
plt.xlabel('x')
plt.ylabel('p(x)')
plt.show()