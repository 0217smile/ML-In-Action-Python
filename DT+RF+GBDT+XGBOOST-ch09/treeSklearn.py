# -*- coding:utf-8 -*-


######################################## 以下为三种可视化方法  START ########################################################
from sklearn.datasets import load_iris
from sklearn import tree
from IPython.display import Image


# 加载scikit-learn自带的数据集iris，训练决策树模型clf
# iris = load_iris()
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(iris.data, iris.target)
#
# #将决策树模型clf存入dot文件iris.dot
# with open('iris.dot', 'w') as f:
#     f = tree.export_graphviz(clf, out_file=f)

############################## 第一种：
# 在第一种是用graphviz的dot命令生成决策树的可视化文件，
# 敲完这个命令后当前目录就可以看到决策树的可视化文件iris.pdf.
#  打开可以看到决策树的模型图。
##                             dot -Tpdf iris.dot -o iris.pdf

############################# 第二种：用pydotplus生成iris.pdf
# import pydotplus
# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = pydotplus.graph_from_dot_data(dot_data)
# graph.write_pdf("iris2.pdf")

############################# 第三种： 直接把图产生在Ipython的notebook上
# from IPython.display import Image
# import pydotplus
# dot_data = tree.export_graphviz(clf, out_file=None,
#                          feature_names=iris.feature_names,
#                          class_names=iris.target_names,
#                          filled=True, rounded=True,
#                          special_characters=True)
# graph = pydotplus.graph_from_dot_data(dot_data)
# Image(graph.create_png())
######################################## 以下为三种可视化方法  END ########################################################

from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

iris = datasets.load_iris()
X = iris.data[:, [0, 2]]
y = iris.target

# 训练模型，限制树的最大深度4
clf = DecisionTreeClassifier(max_depth=4)
clf.fit(X, y)

# 画图
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.show()
