# -*- coding:utf-8 -*-

from __future__ import division
import numpy as np
import pandas as pd

class WEAKC:
    def __init__(self, X, y):
        '''
        this weak classifier is a decision Stump
        it's just a basic example from <统计学习方法>
        :param X: a N*M matrix 
        :param y: a length M vector
        M is the number of traincase
        '''
        self.X = np.array(X)
        self.y = np.array(y)
        self.N = self.shape[0]

    def train(self, W, steps=100):
        '''
        
        :param W: a N length vector
        '''
        # print W
        min = 10000000000000.0
        t_val = 0
        t_point = 0
        t_b = 0
        self.W = np.array(W)
        for i in range(self.N):
            q, err = self.findmin(i, 1, steps)
            if err <  min:
                min = err
                t_val = q
                t_point = i
                t_b = 1
        for i in range(self.N):
            q, err = self.findmin(i, -1, steps)
            if err < min:
                min = err
                t_val = q
                t_point = i
                t_b = -1
        self.t_val = t_val
        self.t_point = t_point
        self.t_b = t_b
        print self.t_val, self.t_point, self.t_b

    def findmin(self, i, b, steps):
        t = 0
        now = self.predintrain(self.X, i, t, b).transpose()
        err = np.sum((now != self.y) * self.W)
        # print now
        pgd = 0
        buttom = np.min(self.X[i, :])
        up = np.max(self.X[i, :])
        mins = 100000000;
        minst = 0
        st = (up - buttom) / steps
        for t in np.arange(buttom, up, st):
            now = self.pre