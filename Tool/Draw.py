import matplotlib.pyplot as plt
import numpy as np

def loadDataSet(filename): #读取数据
    x=[]
    y=[]
    label=[]
    fr=open(filename)
    for line in fr.readlines():
        lineArr=line.strip().split(' ')
        x.append(float(lineArr[0]))
        y.append(float(lineArr[1]))
        if float(lineArr[2])>0:
        	label.append('red')
        else:
        	label.append('blue')
    plt.scatter(x, y,c=label)
   # return dataMat,labelMat #返回数据特征和数据类别

def loadans(filename): #读取数据
    w=[]
    fr=open(filename)
    for line in fr.readlines():
    	lineArr=line.strip().split()
    	w.append(float(lineArr[0]))
    	w.append(float(lineArr[1]))
    	w.append(float(lineArr[2]))

    x1=-5
    x2=10
    y1=(-w[2]-w[0]*x1)/w[1]
    y2=(-w[2]-w[0]*x2)/w[1]
    plt.plot([x1,x2],[y1,y2])


loadDataSet("E:\\traindata4.txt")
loadans("E:\\wrs.txt")

plt.show()