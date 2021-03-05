#!/usr/bin/env python3
#coding:utf-8
#
# パターンが2次元連続変数で
# 多変量正規分布(Multivariate Normal Distribution) でモデル化した場合の
# 識別境界面
# データを少しずつ変化させたときの境界面の変化．
# この例では，双曲線から楕円に変化していきます．
#
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.cm as cm
import functools
import scipy.stats as st
import scipy.special as sp
import math
import sys


class Norm2D:
    """
    2変量正規分布 (2 dimensional normal distribution)
    """
    def __init__(self, X):
        self.__X     = np.copy(X)
        self.__Mu    = np.mean(self.__X, axis=0)
        self.__Sigma = np.cov(self.__X, rowvar=False, bias=True)
        self.__dim   = len(self.__Mu)
        self.__N     = len(self.__X)

    def get_N(self):
        """ 学習データ数の取得 """
        return self.__N

    def get_X(self):
        """ 学習データの取得 """
        return self.__X

    def get_param(self):
        """ 正規分布パラメタの取得 """
        return self.__Mu, self.__Sigma

    def pdf(self, x):
        """ 与えられた分布の(x)における確率密度値を求める"""
        return st.multivariate_normal.pdf(x, mean=self.__Mu, cov=self.__Sigma)

    def sampling(self, N):
        """ 与えられた分布に従ってN点サンプリングする"""
        return st.multivariate_normal.rvs(mean=self.__Mu, cov=self.__Sigma, size=N)

    def this_likelihood(self, x):
        """ パターンxが与えられた時の与えられた分布での尤度を求める """
        L = 1.0
        for n in range(len(x)):
            L *= self.pdf(x[n])
        return L

    def this_log_likelihood(self, x):
        """ パターンxが与えられた時の与えられた分布での対数尤度を求める """
        logL = 0.0
        for n in range(len(x)):
            logL += log(self.pdf(x[n]))
        return logL



#
# 例題3
# x(ω1) = {(-2, 0), (0, 1), (0, -1), (2, 0)}
# x(ω2) = {(3, 2), (4, -1), (6, 1), (7, -2)}
x1 = np.array([[-2.0, 0.0],
               [0.0, 1.0],
               [0.0,-1.0],
               [2.0, 0.0]])
x2 = np.array([[3.0, 2.0],
               [4.0,-1.0],
               [6.0, 1.0],
               [7.0,-2.0]])
# Set the drow range
x = y = np.arange(-20, 20, 0.1)

#
# Init. a graph
#
fig = plt.figure(figsize=(7,7))

ax = fig.add_subplot(111, aspect='equal')

#
# Create the drawing mesh
#
X, Y = np.meshgrid(x, y)
pos = np.dstack((X,Y))



#
# Create Distributions
#
dist1 = Norm2D(x1)

for rate in np.arange(0.8, 1.8, 0.02):
    plt.cla()

    print('rate=%.2f'% rate)
    dist2 = Norm2D(x2*rate)

    #
    # Calc. pdf
    #
    Z1 = dist1.pdf(pos)
    Z2 = dist2.pdf(pos)

    Z = np.log(np.fmax(Z1, Z2))    # Logarithmic expression
    Zdiff = Z1 - Z2

    maxZ = np.max(Z)
    minZ = np.min(Z)

    minZ = -20    # Clip the lowest Z value



    #
    # draw the contour
    #
    levels = np.linspace(minZ, maxZ, 50)
    cs = ax.contourf(X, Y, Z, levels=levels, cmap=cm.inferno, extend='both')

    #
    # draw the boundary
    #
    ax.contour(X, Y, Zdiff, levels=[-1.0e-300, 1.0e-300], colors='r', linestyles='-')

    #
    # data
    #
    ax.scatter(dist1.get_X()[:,0], dist1.get_X()[:,1], s=40, c='red',  marker='o', alpha=0.5, linewidths='2')
    ax.scatter(dist2.get_X()[:,0], dist2.get_X()[:,1], s=40, c='blue', marker='o', alpha=0.5, linewidths='2')

    #
    # Draw
    #
    ax.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.pause(0.2)

plt.show()
