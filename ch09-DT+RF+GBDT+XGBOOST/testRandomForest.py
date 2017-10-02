# -*- coding:utf-8 -*-
from sklearn.datasets import load_iris
import numpy as np
import randomForest

if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    y = iris.target
    temp_data = np.concatenate([X, y.reshape((150, 1))], axis=1)
    # 由于上述代码要求输入的观测数据存储在二维列表中，需将numpy二维数组转换成列表
    data = []
    for i in range(temp_data.shape[0]):
        temp = []
        for j in range(temp_data.shape[1]):
            temp.append(temp_data[i][j])
        data.append(temp)
    rowRange = range(150)
    np.random.shuffle(rowRange)
    # 从鸢尾花数据集(容量为150)按照随机均匀抽样的原则选取70%的数据作为训练数据
    training_data = [data[i] for i in rowRange[0:105]]
    # 按照随机均匀抽样的原则选取30%的数据作为检验数据
    testing_data = [data[i] for i in rowRange[105:150]]
    classifier = randomForest.RandomForestsClassifier(n_estimators=10) #初始化随机森林
    classifier.fit(training_data) # 利用训练数据进行拟合

    finalResults = []
    for row in testing_data:
        currentResult = classifier.predict_randomForests(row[0: 4]) #对检验数据集进行分类
        finalResults.append(currentResult)
    errorVector = np.zeros((45, 1))
    errorVector[np.array(finalResults) != (np.array(testing_data))[:, 4]] = 1
    errorRate = errorVector.sum() / 45 # 计算错误率
    print 'The errorRate is: ', errorRate