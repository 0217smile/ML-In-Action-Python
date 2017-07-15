# -*- coding:utf-8 -*-

from numpy import *

random.rand(4, 4)

randMat = mat(random.rand(4, 4))

invRandMat = randMat.I

myEye = invRandMat*randMat

myEye - eye(4)