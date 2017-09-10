# -*- coding:utf-8 -*-

from __future__ import division
import numpy as np
import math
import operator

class treeNode:
    def __init__(self, feature=1, value=None, labelCounts=None,
                  leftBranch=None,
                  rightBranch=None):
        self.feature = feature
        self.value = value
        self.labelCounts = labelCounts
        self.leftBranch = leftBranch
        self.rightBranch = rightBranch

    def getLabel(self):
        if self.labelCounts == None:
            return None
        else:
            max_counts = 0
            for key in self.labelCounts.keys():
                if self.labelCounts[key] > max_counts:
                    label = key
                    max_counts = self.labelCounts[key]
        return label

class RandomForestsClassifier:
    def __init__(self, n_estimators=20):
        self.n_estimators = n_estimators
        self.list_tree = []

    # 划分的数据集
    def divideSet(self, dataSet, feature, value):
        splitFunction = None
        # 连续型特征
        if isinstance(value, int) or isinstance(value, float):
            splitFunction = lambda featVec : featVec[feature] >= value
        # 离散型特征
        else:
            splitFunction = lambda featVec: featVec[feature] == value
        set1 = [featVec for featVec in dataSet if splitFunction(featVec)]
        set2 = [featVec for featVec in dataSet if not splitFunction(featVec)]
        return (set1, set2)

    # 统计dataSet下所有类标签的频数
    def uniqueCounts(self, dataSet):
        labelCounts = {}
        for featVec in dataSet:  # 遍历每个实例，统计标签的频数
            currentLabel = featVec[-1]
            if currentLabel not in labelCounts.keys():
                labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1
        return labelCounts

    def majorityCnt(self, dataSet):
        '''
        采用多数表决的方法决定叶结点的分类
        :param: 所有的类标签列表
        :return: 出现次数最多的类
        '''
        labelCounts = {}
        labelCounts = self.uniqueCounts(dataSet)
        sortedLabelCounts = sorted(labelCounts.items(),
                                  key=operator.itemgetter(1),
                                  reverse=True)  # 排序
        return sortedLabelCounts[0][0]

    def calcGini(self, dataSet):
        '''
        计算基尼指数
        :param dataSet: 数据集
        :return: 计算结果
        '''
        numEntries = len(dataSet)
        labelCounts = self.uniqueCounts(dataSet)
        Gini = 1.0
        for key in labelCounts.keys():
            prob = float(labelCounts[key]) / numEntries
            Gini -= prob * prob
        return Gini

    # 构造CATR决策树
    def createTree(self, dataSet):
        if len(dataSet) == 0:
            return treeNode()
        currentGini = self.calcGini(dataSet)
        bestGain = 0
        bestCriteria = None
        bestBinarySplit = None
        featCount = len(dataSet[0]) - 1
        featRange = range(0, featCount)
        np.random.shuffle(featRange)
        for feature in featRange[0:int(math.ceil(math.sqrt(featCount)))]:
            featValues = {}
            for featVec in dataSet:
                featValues[featVec[feature]] = 1
            for value in featValues.keys():
                (set1, set2) = self.divideSet(dataSet, feature, value)
                GiniGain = currentGini - (len(set1) * self.calcGini(set1) + len(set2) * self.calcGini(set2)) / len(dataSet)
                if GiniGain > bestGain and len(set1)>0 and len(set2)>0:
                    bestGain = GiniGain
                    bestCriteria = (feature, value)
                    bestBinarySplit = (set1, set2)
        if bestGain > 0:
            leftBranch = self.createTree(bestBinarySplit[0])
            rightBranch = self.createTree(bestBinarySplit[1])
            return treeNode(feature=bestCriteria[0], value=bestCriteria[1],
                             leftBranch=leftBranch, rightBranch=rightBranch)
        else:
            return treeNode(labelCounts=self.uniqueCounts(dataSet))

    # 以文本形式显示决策树
    def printTree(self, tree, indent='  '):
        if tree.labelCounts != None:
            print str(tree.labelCounts)
        else:
            print str(tree.feature) + ':' + str(tree.value) + '?'
            print indent + 'L->', self.printTree(tree.leftBranch, indent+'  ')
            print indent + 'R->', self.printTree(tree.rightBranch, indent+'  ')

    #利用决策树进行预测
    def predict(self, observation, tree):
        if tree.labelCounts != None:
            return tree.getLabel()
        else:
            value = observation[tree.feature]
            branch = None
            if isinstance(value, int) or isinstance(value, float):
                if value >= tree.value: branch = tree.leftBranch
                else: branch = tree.rightBranch
            else:
                if value == tree.value: branch = tree.leftBranch
                else: branch = tree.rightBranch
            return self.predict(observation, branch)

    def generateBootstrapSample(self, dataSet):
        m = len(dataSet)
        samples = []
        for i in range(m): # 自助采样法 采样m个样本
            samples.append(dataSet[np.random.randint(m)])
        return samples

    #构造随机森林
    def fit(self, dataSet):
        for i in range(self.n_estimators):
            samples = self.generateBootstrapSample(dataSet)
            currentTree = self.createTree(samples)
            self.list_tree.append(currentTree)
        print 'list_tree is: ', self.list_tree

    # 利用随机森林对给定观测数据进行预测
    def predict_randomForests(self, observation):
        from numpy import *
        results = {}
        for i in range(len(self.list_tree)):
            currentResult = self.predict(observation, self.list_tree[i])
            if currentResult not in results.keys():
                results[currentResult] = 0
            results[currentResult] += 1
        sortedResults = sorted(results.items(),
                                  key=operator.itemgetter(1),
                                  reverse=True)  # 排序
        return sortedResults[0][0]