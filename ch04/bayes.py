# -*- coding:utf-8 -*-
from numpy import *

#创建一些实验样本
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]
    return postingList, classVec

#创建一个包含在所有文档中出现的不重复词的词表
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

#将一例数据inputSet从文本转换为词向量格式           这个是词集模型 set-of-words 还有一种词袋模型 bag-of-words
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print "the word: %s is not in my Vocabulary" % word
    return returnVec

#词袋模型 bag-of-words
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


#朴素贝叶斯分类器训练函数------第一个版本
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    # p0Num = zeros(numWords);p1Num = zeros(numWords)
    # p0Denom = 0.0; p1Denom = 0.0
    # 将所有词的出现数初始化为1，并将分母都初始化为2
    p0Num = ones(numWords); p1Num = ones(numWords)
    p0Denom = 2.0; p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 避免下溢出问题：对成绩取自然对数
    # p1Vect = p1Num / p1Denom
    # p0Vect = p0Num / p0Denom
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive

#朴素贝叶斯分类函数
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    #四个输入，要分类的向量vec2Classify，以及使用trainNB0()训练的到的三个概率
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1-pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

#一个便利函数
def testingNB():
    listOPsts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPsts)
    trainMat = []
    for postinDoc in listOPsts:
        trainMat.append(array(setOfWords2Vec(myVocabList, postinDoc)))
    p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classifid as: ', classifyNB(thisDoc, p0V, p1V, pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)

def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList = []; classList = []; fullText = []
    for i in range(1,26):
        #选垃圾邮件文档
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        #选正常邮件文档
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    #构建词表
    vocabList = createVocabList(docList)
    trainingSet = range(50); testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    #构建词向量矩阵
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print "classified error: ", docList[docIndex]
    print "the error rate is: ", float(errorCount)/len(testSet)

#从个人广告中获取区域倾向
#计算高频词汇
def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]

#RSS源分类器
def localWords(feed1, feed0):
    import feedparser
    docList = []; classList = []; fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText) # 高频词都是些: and you the for with
    #print "top30Words is :", top30Words
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet = range(2 * minLen); testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    trainMat = []; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(wordVector, p0V, p1V, pSpam) != classList[docIndex]:
            print "classified error: ", docList[docIndex]
            errorCount += 1
    print "the error rate is: ", float(errorCount)/len(testSet)
    return vocabList, p0V, p1V

#最具表整形的词汇显示函数
def getTopWords(ny, sf):
    import operator
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = []; topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0:
            topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print "SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**"
    for item in sortedSF:
        print item[0]
    sortedNY = sorted(topNY, key=lambda pair:pair[1], reverse=True)
    print "NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**"
    for item in sortedNY:
        print item[0]