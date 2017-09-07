# -*- coding:utf-8 -*-
# 决策树（上）ID3与C4.5——从原理到算法实现http://blog.csdn.net/HerosOfEarth/article/details/52347820
from math import log
import treePlotter
import operator
import pickle

# 导入数据
def createDataSet():
    dataSet = [['youth', 'no', 'no', 1, 'refuse'],
               ['youth', 'no', 'no', '2', 'refuse'],
               ['youth', 'yes', 'no', '2', 'agree'],
               ['youth', 'yes', 'yes', 1, 'agree'],
               ['youth', 'no', 'no', 1, 'refuse'],
               ['mid', 'no', 'no', 1, 'refuse'],
               ['mid', 'no', 'no', '2', 'refuse'],
               ['mid', 'yes', 'yes', '2', 'agree'],
               ['mid', 'no', 'yes', '3', 'agree'],
               ['mid', 'no', 'yes', '3', 'agree'],
               ['elder', 'no', 'yes', '3', 'agree'],
               ['elder', 'no', 'yes', '2', 'agree'],
               ['elder', 'yes', 'no', '2', 'agree'],
               ['elder', 'yes', 'no', '3', 'agree'],
               ['elder', 'no', 'no', '1', 'refuse']]
    labels = ['age', 'working?', 'house?', 'credit_situation']
    # dataSet = [[1,1,'yes'],
    #            [1,1,'yes'],
    #            [1,0,'no'],
    #            [0,1,'no'],
    #            [0,1,'no']]
    # labels = ['no surfacing', 'flippers']
    return dataSet, labels

def calcShannonEnt(dataSet):
    '''
    计算香农熵
    :param dataSet:数据集 
    :return: 计算结果
    '''
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet: #遍历每个实例，统计标签的频数
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2) #以2为底的对数
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    '''
    按照给定特征划分数据集
    :param dataSet: 带划分的数据集
    :param axis: 划分数据集的特征（编号或索引）
    :param value: 需要返回的特征的值
    :return: 划分结果列表
    '''
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def calcConditionalEntropy(dataSet, i, featList, uniqueVals):
    '''
    计算X_i给定的条件下， Y的条件熵
    :param dataSet: 数据集
    :param i: 维度i
    :param featList:数据集特征列表 
    :param uniqueVals: 数据集特征集合
    :return: 条件熵
    '''
    conditionEnt = 0.0
    for value in uniqueVals:
        subDataSet = splitDataSet(dataSet, i, value)       #第i维度特征下的各个value值对应的各个subDataSet
        prob = len(subDataSet) / float(len(dataSet))       #极大似然估计概率
        conditionEnt += prob * calcShannonEnt(subDataSet)  #条件熵的计算
    return conditionEnt

def calcInformationGain(dataSet, baseEntropy, i):
    '''
    计算信息增益
    :param dataSet:数据集 
    :param baseEntropy: 数据集的信息熵
    :param i: 特征维度i
    :return: 特征i对数据集的信息增益I(D|X_i)
    '''
    featList = [example[i] for example in dataSet]  #第i维特征列表
    uniqueVals = set(featList) # 转换成集合，去重
    newEntropy = calcConditionalEntropy(dataSet, i, featList, uniqueVals)
    infoGain = baseEntropy - newEntropy #信息增益，就是熵的减少
    return infoGain

def calcInformationGainRatio(dataSet, baseEntropy, i):
    '''
    计算信息增益比
    :param dataSet:数据集 
    :param baseEntropy: 数据集的信息熵
    :param i: 特征维度i
    :return: 特征i对数据集的信息增益比Ir（D|X_i）
    '''
    featList = [example[i] for example in dataSet]
    uniqueVals = set(featList)
    splitInfo = 0.0
    for value in uniqueVals:
        subDataSet = splitDataSet(dataSet, i, value)
        prob = len(subDataSet)/float(len(dataSet))
        splitInfo += -prob * log(prob, 2)
    #if (splitInfo == 0): # fix the overflow bug
    #   continue
    return calcInformationGain(dataSet, baseEntropy, i) / splitInfo

'''
算法5.2 （ID3算法）
输入： 训练数据集D，特征集A，阈值ε；
输出： 决策树T.
算法5.3 （C4.5算法）
输入： 训练数据集D，特征集A，阈值ε；
输出： 决策树T.
'''
def chooseBestFeatureToSplitByID3(dataSet):
    '''
    选择最好的数据集特征划分
    :param dataSet: 数据集
    :return: 划分结果
    '''
    numFeatures = len(dataSet[0]) - 1 #最后一列是分类标签，不属于特征向量
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures): #遍历所有特征
        infoGain = calcInformationGain(dataSet, baseEntropy, i) # 计算信息增益
        if(infoGain > bestInfoGain):  #选择最大的信息增益
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature #返回最优特征对应的维度

def chooseBestFeatureToSplitByC45(dataSet):
    '''
    选择最好的数据集划分方式
    :param dataSet: 数据集
    :return: 最好的划分维度
    '''
    numFeatures = len(dataSet[0]) - 1  # 最后一列yes分类标签，不属于特征变量
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGainRate = 0.0
    bestFeature = -1
    for i in range(numFeatures):  # 遍历所有维度特征
        infoGainRate = calcInformationGainRatio(dataSet, baseEntropy, i)    # 计算信息增益比
        if (infoGainRate > bestInfoGainRate):  # 选择最大的信息增益比
            bestInfoGainRate = infoGainRate
            bestFeature = i
    return bestFeature  # 返回最佳特征对应的维度

def majorityCnt(classList):
    '''
    采用多数表决的方法决定叶节点的类别
    :param classList: 所有的类标签列表
    :return: 出现次数最多的类
    '''
    classCount = {}
    for vote in classList:    # 统计所有类标签的频数
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),
                              key=operator.itemgetter(1),
                              reverse=True)      # 排序
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    '''
    创建决策树
    :param dataSet: 训练数据集
    :param labels: 所有的类标签
    :return: 构建的决策树
    '''
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(dataSet):
        return classList[0]              # 第一个递归结束条件：所有的类标签完全相同
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)    # 第二个递归结束条件：用完了所有特征
    bestFeat = chooseBestFeatureToSplitByID3(dataSet)    # ID3算法的最优划分特征
    #bestFeat = chooseBestFeatureToSplitByC45(dataSet)   # C4.5算法的最优划分特征
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}           # 使用字典类型存储树的信息
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]             # 复制所有类标签，保证每次递归调用时不改变原始列表的内容
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree



def classify(inputTree, featLabels, testVec):
    '''
    利用决策树进行分类
    :param inputTree:构造好的决策树模型
    :param featLabels: 所有的类标签
    :param testVec: 测试数据
    :return: 分类决策结果
    '''
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree, filename):
    '''
    保存决策树到文件
    :param inputTree: 构建的决策树 
    :param filename:  保存路径
    :return: 无
    '''
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    '''
    从文件读取决策树
    :param filename:文件路径名 
    :return: 决策树
    '''
    fr = open(filename, 'rb')
    return pickle.load(fr)

if __name__ == "__main__":
    dataSet, labels = createDataSet()
    myTree = createTree(dataSet, labels)
    print myTree
    treePlotter.createPlot(myTree)