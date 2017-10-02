# -*- coding: utf-8 -*-
'''
#########################本程序是CART回归，包括两部分：回归树和模型树##############################
'''
from numpy import *

class treeNode():
    def __init__(self, feat, value, left, right):
        featureToSplitOn = feat
        vlaueOfSplit = val
        leftBranch = left
        rightBranch = right
# 加载数据
def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)
        dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet, feature, value):
    '''
    通过数组过滤的方法将数据集切分为两个子集
    :param dataSet: 待切分数据集
    :param feature: 待切分特征
    :param value: 该特征的某个值
    :return: 左右连个切分子集
    '''
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1

###############################                     Part 1 回归树子函数   START          ########################################
def regLeaf(dataSet):
    # 负责生成叶节点，在回归树中，该模型其实就是目标变量的均值
    return mean(dataSet[:, -1])
def regErr(dataSet):
    #误差估计函数，该函数在给定数据集上计算目标变量的平方误差
    #这里是通过 均方差乘以数据集中样本个数 来实现的
    return var(dataSet[:, -1]) * shape(dataSet)[0]


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    '''
    找到数据集切分的最佳特征及特征值
    :param dataSet: 数据集
    :param leafType: 
    :param errType: 
    :param ops: 用户指定的参数，用于控制函数停止时机：误差下降最小值与切分的最小样本数
    :return: 最佳切分点
    '''
    tolS = ops[0]; tolN = ops[1]
    #if all the target variables are the same value: quit and return value
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: #exit cond 1
        return None, leafType(dataSet)
    m,n = shape(dataSet)
    #the choice of the best feature is driven by Reduction in RSS error from mean
    S = errType(dataSet)
    bestS = inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set((dataSet[:,featIndex].T.tolist())[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    #if the decrease (S-bestS) is less than a threshold don't do the split
    if (S - bestS) < tolS:
        return None, leafType(dataSet) #exit cond 2
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  #exit cond 3
        return None, leafType(dataSet)
    return bestIndex,bestValue

##############################                     回归树子函数   END                   ########################################

##############################                     回归树剪枝函数   START             ########################################
def isTree(obj):
    # 测试输入变量是否是棵树
    return (type(obj).__name__=='dict')

def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0

def prune(tree, testData):
    if shape(testData)[0] == 0:                   #if we have no test data collapse the tree
        return getMean(tree)
    if (isTree(tree['left']) or isTree(tree['right'])): #if the branches are not trees try to prune them
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    # 深度优先搜索
    # 递归调用prune函数对左右子树,注意与左右子树对应的左右子测试数据集
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    # 当递归搜索到左右子树均为叶节点时，计算测试数据集的误差平方和,比较前后差别以判断是否可以合并
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) +\
                       sum(power(rSet[:, -1] - tree['left'], 2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print "merging"
            return treeMean
        else:
            return tree
    else:
        return tree


##############################                     回归树剪枝函数   END                   ######################################

##############################                     Part 2 模型树相关函数   START          ######################################
def linearSolve(dataSet):
    m, n = shape(dataSet)
    X = mat(ones((m, n))); Y = mat(ones((m, 1)))
    X[:, 1:n] = dataSet[:, 0:n-1] #create a copy of data with 1 in 0th postion
    Y = dataSet[:, -1]
    xTx = X.T*X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\ntry increase the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y

def modelLeaf(dataSet): #create linear model and return coeficients
    ws, X, Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat, 2))

##############################                     模型树相关函数   END                   ######################################

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):#assume dataSet is NumPy Mat so we can array filtering
    '''
    树构建函数，包含三个可选参数，这些可选参数决定了树的类型
    :param dataSet: 数据集 
    :param leafType: 给出简历叶节点的函数
    :param errType: 代表误差计算函数
    :param ops: 包含树构建所需其他参数的元组
    :return: 构建的树
    '''
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)#choose the best split
    if feat == None: return val #if the splitting hit a stop condition return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree

##############################                     用树回归进行预测的代码   START                 ######################################

def regTreeEval(model, inDat):
    return float(model)


def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1, n + 1)))
    X[:, 1:n + 1] = inDat
    return float(X * model)


def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat
##############################                     用树回归进行预测的代码   END                    ####################################
##############################                     对回归树，模型树，预测的测试代码   START        ####################################



def regTest():
    # myDat = loadDataSet('ex00.txt')
    myDat = loadDataSet('ex0.txt')
    myMat = mat(myDat)
    myTree = createTree(myMat)
    # myTree = createTree(myMat, ops=(0, 1))
    print myTree

def modelTest():
    myMat = mat(loadDataSet('exp2.txt'))
    myTree = createTree(myMat, modelLeaf, modelErr, (1, 10))
    print myTree

def predictTest():
    '''
    对 回归树，模型树，普通线形回归树三种模型的测试函数 
    '''
    trainMat = mat(loadDataSet('bikeSpeedVsIq_train.txt'))
    testMat = mat(loadDataSet('bikeSpeedVsIq_test.txt'))

    # 回归树预测
    myTreeReg = createTree(trainMat, ops=(1, 20))
    yHatReg = createForeCast(myTreeReg, testMat[:, 0])
    resultReg = corrcoef(yHatReg, testMat[:, 1], rowvar=0)[0, 1]
    print '回归树R^2值：', resultReg
    #模型树预测
    myTreeModel = createTree(trainMat, modelLeaf,  modelErr,(1, 20))
    yHatModel = createForeCast(myTreeModel, testMat[:, 0], modelTreeEval)
    resultModel = corrcoef(yHatModel, testMat[:, 1], rowvar=0)[0, 1]
    print '模型树R^2值：', resultModel
    #普通线性回归
    ws, X, Y = linearSolve(trainMat)
    yHatLinear = mat(zeros((shape(testMat)[0], 1)))
    print '线性回归的ws值：', ws
    for i in range(shape(testMat)[0]):
        yHatLinear[i, 0] = testMat[i, 0]*ws[1, 0] + ws[0, 0]
    resultLinear = corrcoef(yHatLinear, testMat[:, 1], rowvar=0)[0, 1]
    print '线性回归R^2值：', resultLinear



##############################                     对回归树，模型树，预测的测试代码   END          ######################################

if __name__ == '__main__':
    #regTest()
    #modelTest()
    predictTest()