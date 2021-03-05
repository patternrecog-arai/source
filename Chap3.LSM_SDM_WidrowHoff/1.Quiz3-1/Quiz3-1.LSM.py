#!/usr/bin/env python3
#coding:utf-8
"""
最小自乗法による評価関数最小法の解析解
2次元データ
"""
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

def show2D( vecW, x1, x2, ax ):
    """
    2次元表示
    """
    ax.cla() # 描画クリア

    ax.grid()
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect('equal', 'datalim')

    # 学習用パターン
    ax.scatter(x1[:,1], x1[:,2], c='red' , marker="o")
    ax.scatter(x2[:,1], x2[:,2], c='blue', marker="o")

    # 分離境界線
    x_fig = np.array(np.arange(0,1.1,0.1))
    y_fig = -(vecW[1]/vecW[2])*x_fig - (vecW[0]/vecW[2])

    ax.plot(x_fig,y_fig, c='green')


def show3D( vecW, x1, x2, ax ):
    """
    3次元表示
    """
    ax.cla() # 描画クリア

    ax.grid()
    ax.set_zlim(0, 2)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x0")

    # 学習用パターン
    ax.scatter(x1[:,1], x1[:,2], x1[:,0], c='red' , marker="o")
    ax.scatter(x2[:,1], x2[:,2], x2[:,0], c='blue', marker="o")

    # 分離境界面
    x = np.arange(0, 1.1, 0.1)
    y = np.arange(0, 1.1, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = -vecW[1]/vecW[0]*X - vecW[2]/vecW[0]*Y 

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color='green')


def show( vecW, x1, x2, sp1, sp2 ):
    """
    3D + 2D 表示
    """

    show3D(vecW, x1, x2, sp1)
    show2D(vecW, x1, x2, sp2)
    plt.pause(0.1)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    # 学習データ数
    train_num = 20

    
    # class1の学習データ
    sample1 = [[0.15, 0.20], [0.20, 0.10], [0.20, 0.30], [0.25, 0.20], [0.25, 0.40],
               [0.30, 0.10], [0.30, 0.30], [0.35, 0.20], [0.40, 0.10], [0.40, 0.30]]

    # class2の学習データ
    sample2 = [[0.55, 0.35], [0.50, 0.25], [0.50, 0.45], [0.45, 0.15], [0.45, 0.35],
               [0.40, 0.25], [0.40, 0.45], [0.35, 0.35], [0.30, 0.45], [0.35, 0.25]]

    # class1の教師ラベル
    label_X1 = np.ones(train_num//2)        # ラベル1

    # class2の教師ラベル
    label_X2 = np.ones(train_num//2) * -1   # ラベル-1

    x0 = np.ones(train_num//2) # x0は常に1

    # 拡張特徴ベクトルを並べた行列
    X1 = np.c_[x0, sample1]
    X2 = np.c_[x0, sample2]

    # 全学習データを統合
    matX   = np.r_[X1, X2]
    labels = np.r_[label_X1, label_X2]
    #print('matX=', matX)
    #print('labels=', labels)

    # 重みベクトルを解析的に求める
    vecW = np.dot(np.dot(np.linalg.inv(np.dot(matX.T,matX)),(matX.T)), labels)

    print( vecW )

    # グラフ描画の準備
    fig = plt.figure(figsize=(12,6))
    fig.suptitle('Least Squares Method (2-dimensional patterns)')
    sp1 = fig.add_subplot(1, 2, 1, projection='3d')
    sp2 = fig.add_subplot(1, 2, 2)

    show(vecW, X1, X2, sp1, sp2) # 表示
    plt.show() # 最後は表示をkeep
