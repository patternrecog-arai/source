#!/usr/bin/env python3
#coding:utf-8
"""
Widrow-Hoffの学習則による評価関数最小解
2次元データ
"""
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def train(vecW, vecX, label, rho):
    """
    Widrow-Hoffの学習則
        vecW:  重みベクトル
        vecX:  学習パターンベクトル
        label: 教師信号 (1 or -1)
        rho:   学習率
    """
    eps = np.dot(vecW , vecX) - label  # epsilon
    dJp = 2.0* eps * vecX              # Jpの微分
    vecW = vecW - rho * dJp            # W更新
    Jp = eps ** 2                      # Jp

    return vecW, Jp


def show2D( vecW, vecWtrue, X1, X2, loop_cnt, ax ):
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

    # Loop回数表示
    str_cnt = "loop #%d" % (loop_cnt)
    ax.text(0.75, 0.9, str_cnt, size = 12, color = "green")

    # 学習用パターン
    ax.scatter(X1[:,1], X1[:,2], c='red' , marker="o")
    ax.scatter(X2[:,1], X2[:,2], c='blue', marker="o")

    # 分離境界線(解析解)
    x_fig = np.array(np.arange(0,1.1,0.1))
    y_fig = -(vecWtrue[1]/vecWtrue[2])*x_fig - (vecWtrue[0]/vecWtrue[2])
    ax.plot(x_fig,y_fig, c='black')

    # 分離境界線
    x_fig = np.array(np.arange(0,1.1,0.1))
    y_fig = -(vecW[1]/vecW[2])*x_fig - (vecW[0]/vecW[2])
    ax.plot(x_fig,y_fig, c='green')


def show3D( vecW, X1, X2, ax ):
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
    ax.scatter(X1[:,1], X1[:,2], X1[:,0], c='red' , marker="o")
    ax.scatter(X2[:,1], X2[:,2], X2[:,0], c='blue', marker="o")

    # 分離境界面
    x = np.arange(0, 1.1, 0.1)
    y = np.arange(0, 1.1, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = -vecW[1]/vecW[0]*X - vecW[2]/vecW[0]*Y 

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.3, color='green')


def show( vecW, vecWtrue, X1, X2, loop_cnt, sp1, sp2 ):
    """
    3D + 2D 表示
    """

    show3D(vecW, X1, X2, sp1)
    show2D(vecW, vecWtrue, X1, X2, loop_cnt, sp2)
    plt.pause(0.001)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-r', '--rho', type=float, default=0.01, help='learning rate rho')
    parser.add_argument('-l', '--loop', type=int, default=1500, help='loop count')
    args = parser.parse_args()    


    train_num = 20 # 学習データ数

    # class1の学習データ
    sample1 = [[0.15, 0.20], [0.20, 0.10], [0.20, 0.30], [0.25, 0.20], [0.25, 0.40],
               [0.30, 0.10], [0.30, 0.30], [0.35, 0.20], [0.40, 0.10], [0.40, 0.30]]

    # class2の学習データ
    sample2 = [[0.55, 0.35], [0.50, 0.25], [0.50, 0.45], [0.45, 0.15], [0.45, 0.35],
               [0.40, 0.25], [0.40, 0.45], [0.35, 0.35], [0.30, 0.45], [0.35, 0.25]]

    # class1の教師ラベル
    label_X1 = np.ones(train_num//2)                        # ラベル1

    # class2の教師ラベル
    label_X2 = np.ones(train_num//2) * -1                   # ラベル-1

    x0 = np.ones(train_num//2) # x0は常に1

    # 拡張特徴ベクトルを並べた行列
    X1 = np.c_[x0, sample1]
    X2 = np.c_[x0, sample2]

    # 全学習データを統合
    matX   = np.r_[X1, X2] 
    labels = np.r_[label_X1, label_X2]
    #print('matX=', matX)
    #print('labels=', labels)

    vecW = np.array([1,0,1]) #初期の重みベクトル 適当に決める


    # グラフ描画の準備
    fig = plt.figure(figsize=(12,6))
    fig.suptitle('Widrow - Hoff Rule (2-dimensional patterns)')
    sp1 = fig.add_subplot(1, 2, 1, projection='3d')
    sp2 = fig.add_subplot(1, 2, 2)

    # 最小二乗法での解析解
    vecWtrue = np.dot(np.dot(np.linalg.inv(np.dot(matX.T,matX)),(matX.T)), labels)

    # Widrow-Hoffの学習則での解法
    order = np.arange(len(labels))
    for j in range(args.loop):
        maxJp = 0
        aveJp = 0
        s_order = np.random.permutation(order) # 毎回Training dataの提示順番をshuffle
        for pos in s_order:
            vecX  = matX[pos]
            label = labels[pos]
            vecW, Jp = train(vecW, vecX, label, args.rho) # 学習
            if Jp > maxJp: maxJp = Jp
            aveJp += Jp
        aveJp /= len(matX)

        cnt = j+1

        eprint("[%d] aveJp=%7.3f  maxJp=%7.5f " % (j, aveJp, maxJp), end="")
        eprint("w = ",  vecW )
        if cnt%10 == 0:
            show(vecW, vecWtrue, X1, X2, cnt, sp1, sp2) # 表示

    show(vecW, vecWtrue, X1, X2, cnt, sp1, sp2) # 表示
    plt.show() # 最後は表示をkeep
