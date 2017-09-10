# -*- coding:utf-8 -*-

from numpy import *
from itertools import *
import treePlotter
#########################################################################################################################
######################################    CART 分类树                           #########################################
######################################    Part 1  特征值是离散情况的            #########################################
######################################    Part 2  特征值是连续情况的            #########################################
#########################################################################################################################

def calcGini(dataSet):
    '''
    计算基尼指数
    :param dataSet: 数据集
    :return: 计算结果
    '''
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet: # 遍历每个实例，统计标签的频数
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    Gini = 1.0
    for key in labelCounts.keys():
        prob = float(labelCounts[key]) / numEntries
        Gini -= prob * prob
    return Gini

# ############ 这个函数暂时封印，考虑的太简单了###################
# def calcGiniWithFeat(dataSet, feature, value):
#     '''
#     计算给定特征下的基尼指数
#     :param dataSet: 数据集
#     :param feature: 特征维度
#     :param value: 该特征变量所取的值
#     :return: 基尼指数计算结果
#     '''
#     D0 = []; D1 = []
#     # 根据特征划分数据
#     for featVec in dataSet:
#         if featVec[feature] == value:
#             D0.append(featVec)
#         else:
#             D1.append(featVec)
#     Gini = len(D0) / len(dataSet) * calcGini(D0) + len(D1) / len(dataSet) * calcGini(D1)
#     return Gini

def majorityCnt(classList):
    '''
    采用多数表决的方法决定叶结点的分类
    :param: 所有的类标签列表
    :return: 出现次数最多的类
    '''
    classCount={}
    for vote in classList:                  # 统计所有类标签的频数
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1),
                              reverse=True) # 排序
    return sortedClassCount[0][0]

#######################################################################################################
##########################            Part 1 特征为离散值的情况           #############################
#######################################################################################################
# http://blog.csdn.net/herosofearth/article/details/52425952（有错误的，但流程可参考）
# http://www.cnblogs.com/qwj-sysu/p/5974421.html（是正确的）

def featuresplit(features):
    count = len(features)#特征值的个数
    if count < 2:
        print "please check sample's features,only one feature value"
        return -1
    # 由于需要返回二分结果，所以每个分支至少需要一个特征值，所以要从所有的特征组合中选取1个以上的组合
    # itertools的combinations 函数可以返回一个列表选多少个元素的组合结果，例如combinations(list,2)返回的列表元素选2个的组合
    # 我们需要选择1-（count-1）的组合
    featureIndex = range(count)
    featureIndex.pop(0)
    combinationsList = []
    resList = []
    # 遍历所有的组合
    for i in featureIndex:
        temp_combination = list(combinations(features, len(features[0:i])))
        combinationsList.extend(temp_combination)
        combiLen = len(combinationsList)
    # 每次组合的顺序都是一致的，并且也是对称的，所以我们取首尾组合集合
    # zip函数提供了两个列表对应位置组合的功能
    resList = zip(combinationsList[0:combiLen/2], combinationsList[combiLen-1:combiLen/2-1:-1])
    return resList


# 得到特征的划分结果之后，我ex们使用二分后的特征值划分数据集
def splitDataSet(dataSet, axis, values):
    '''
    按照给定特征划分数据集
    :param dataSet: 带划分的数据集
    :param axis: 划分数据集的特征（编号或索引）
    :param values: axis特征下的特征值（单个或者一个元组集合）
    :return: 划分结果数据集
    '''
    retDataSet = []
    for featVec in dataSet:
        for value in values:
            if featVec[axis] == value:
                reducedFeatVec = featVec[:axis]       # 找到满足条件的样本后，剔除样本中的该特征
                reducedFeatVec.extend(featVec[axis + 1:])
                retDataSet.append(reducedFeatVec)     # 添加进返回数据集
    return retDataSet


# 遍历每个特征的每个二分特征值，得到最好的特征以及二分特征值
def chooseBestFeatureToSplit(dataSet):
    '''
    
    :param dataSet: 数据集
    :return: 选择的最好特征以及特征二分值
    '''
    numFeatures = len(dataSet[0]) - 1
    bestGiniGain = 1.0; bestFeat = -1; bestBinarySplit = ()
    for i in range(numFeatures): # 遍历特征
        featList = [example[i] for example in dataSet] #得到特征列
        uniqueVals = list(set(featList))                #从特征列获取该特征的特征值的set集合
        #三个特征值的二分结果：
        #[(('young',), ('old', 'middle')), (('old',), ('young', 'middle')), (('middle',), ('young', 'old'))]
        for split in featuresplit(uniqueVals):    # split就会是(('young',), ('old', 'middle'))  是('young', 'middle')), (('middle',) 等
            GiniGain = 0.0
            if len(split) == 1:
                continue
            (left, right) = split                  # 举例：left = 'young', right = ('old', 'middle')
            # 对于每一个可能的二分结果计算gini指数
            # 左子树Gini指数
            left_subDataSet = splitDataSet(dataSet, i, left)
            left_prob = len(left_subDataSet) / float(len(dataSet))
            GiniGain += left_prob * calcGini(left_subDataSet)
            #右子树Gini指数
            right_subDataSet = splitDataSet(dataSet, i, right)
            right_prob = len(right_subDataSet) / float(len(dataSet))
            GiniGain += right_prob * calcGini(right_subDataSet)
            if(GiniGain <= bestGiniGain):  # 比较是否是最好的结果
                bestGiniGain = GiniGain
                bestFeat = i
                bestBinarySplit = (left, right) # 记录下最好的特征，及其特征二分点
    return bestFeat, bestBinarySplit


def createTree(dataSet, labels):
    '''
    创建决策树
    :param dataSet: 训练数据集
    :param labels: 所有的类标签
    :return: 构造的决策树
    '''
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]             # 第一个递归结束条件：所有的类标签完全相同
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)   # 第二个递归结束条件：用完了所有特征
    bestFeat, bestBinarySplit = chooseBestFeatureToSplit(dataSet)
    print bestFeat, bestBinarySplit, labels
    bestFeatLabel = labels[bestFeat]
    if bestFeat == -1:
        return majorityCnt(classList)
    myTree = {bestFeatLabel: {}}
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = list(set(featValues))
    for value in bestBinarySplit:
        subLabels = labels[:]
        if len(value) < 2:
            del(subLabels[bestFeat])
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

#######################################################################################################
##########################            Part 2 特征为连续值的情况           #############################
#######################################################################################################
# http://www.cnblogs.com/qwj-sysu/p/5981231.html

def splitDataSet2(dataSet, axis, value, threshold):
    '''
    :param threshold: <= 或者 >=
    :return: 划分的数据集
    '''
    retDataSet = []
    if threshold == 'lt':
        for featVec in dataSet:
            if featVec[axis] <= value:
                retDataSet.append(featVec)
    else:
        for featVec in dataSet:
            if featVec[axis] > value:
                retDataSet.append(featVec)
    return retDataSet

def chooseBestFeatureToSplit2(dataSet):
    '''
    选择最好的划分特征及其值
    :param dataSet: 数据集
    :return: 最佳的 划分特征及特征值
    '''
    numFeatures = len(dataSet[0]) - 1
    bestGiniGain = 1.0; bestFeature = -1; bestValue=""
    for i in range(numFeatures):          # 遍历特征
        featList = [example[i] for example in dataSet]
        uniqueVals = list(set(featList))   #从特征列获取该特征的特征值的set集合
        uniqueVals.sort()
        for value in uniqueVals:          #遍历所有的特征值
            GiniGain = 0.0
            #左子树Gini指数
            left_subDataSet = splitDataSet2(dataSet, i, value, 'lt')
            left_prob = len(left_subDataSet) / float(len(dataSet))
            GiniGain += left_prob * calcGini(left_subDataSet)
            print left_prob, calcGini(left_subDataSet)
            # 右子树Gini指数
            right_subDataSet = splitDataSet2(dataSet, i, value, 'gt')
            right_prob = len(right_subDataSet) / float(len(dataSet))
            GiniGain += right_prob * calcGini(right_subDataSet)
            print right_prob, calcGini(right_subDataSet)
            print GiniGain
            if(GiniGain < bestGiniGain):        #比较是否为最好的结果
                bestGiniGain = GiniGain         #记录最好的特征及其下的最好特征值
                bestFeature = i
                bestValue = value
    return bestFeature, bestValue

#### 生成cart：总体上和离散值的差不多，
#### 主要差别在于分支的值要加上大于或者小于等于号
def createTree2(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):   # 第一个递归结束条件：所有的类标签完全相同
        return classList[0]
    if len(dataSet[0]) == 1:                              # 第二个递归结束条件：用完了所有特征
        return majorityCnt(classList)
    bestFeat, bestValue = chooseBestFeatureToSplit2(dataSet)
    print bestFeat, bestValue, labels
    bestFeatLabel = labels[bestFeat]
    if bestFeat == -1:
        return majorityCnt(classList)
    myTree = {bestFeatLabel: {}}
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = list(set(featValues))
    subLabels = labels[:]
    print bestValue
    myTree[bestFeatLabel][bestFeatLabel + '<=' + str(round(float(bestValue), 3))] = \
        createTree2(splitDataSet2(dataSet, bestFeat, bestValue, 'lt'), subLabels)
    myTree[bestFeatLabel][bestFeatLabel + '>' + str(round(float(bestValue), 3))] = \
        createTree2(splitDataSet2(dataSet, bestFeat, bestValue, 'gt'), subLabels)
    return myTree


#test 1------------离散型测试1
if __name__ == '__main__':
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    print lenses
    lensesTree = createTree(lenses, lensesLabels)
    treePlotter.createPlot(lensesTree)

#test 2------------离散型测试2    有问题 先挂起
# if __name__ == '__main__':
#     weather = [['sunny', 'hot', 'high', 'FALSE', 'no'],
#                ['sunny', 'hot', 'high', 'TRUE',  'no'],
#                ['overcast', 'hot', 'high', 'FALSE', 'yes'],
#                ['rainy', 'mild', 'high', 'FALSE', 'yes'],
#                ['rainy',  'cool',  'normal', 'FALSE', 'yes'],
#                ['rainy',  'cool',  'normal', 'TRUE', 'no'],
#                ['overcast', 'cool', 'normal',  'TRUE', 'yes'],
#                ['sunny',    'mild',    'high',    'FALSE', 'no'],
#                ['sunny',    'cool',    'normal',   'FALSE', 'yes'],
#                ['rainy',    'mild',    'normal',   'FALSE', 'yes'],
#                ['sunny',    'mild',    'normal',   'TRUE', 'yes'],
#                ['overcast',    'mild',    'high',   'TRUE', 'yes'],
#                ['overcast',    'hot',    'normal',   'FALSE', 'yes'],
#                ['rainy', 'mild', 'high', 'TRUE', 'no']]
#     weatherLabels = ['Outlook' , 'Temperature' , 'Humidity' , 'Wind']
#     print weather
#     print weatherLabels
#     weatherTree = createTree(weather, weatherLabels)
#     treePlotter.createPlot(weatherTree)

#test 3------------连续型测试1    待找连续型数据集