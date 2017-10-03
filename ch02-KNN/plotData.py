# -*- coding:utf-8 -*-
import kNN
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()

datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')
ax = fig.add_subplot(111)
#ax.scatter(datingDataMat[:, 1], datingDataMat[:,2]) 没有使用样本分类的特征值，绘制的图看不出有用的数据模式信息
#ax.scatter(datingDataMat[:, 1], datingDataMat[:,2], 15.0*array(datingLabels), 15.0*array(datingLabels)) 用第二列，第三列的数据绘制的图，区别度不大
ax.scatter(datingDataMat[:, 0], datingDataMat[:,1], 15.0*array(datingLabels), 15.0*array(datingLabels))
plt.show()