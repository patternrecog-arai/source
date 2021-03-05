#!/usr/bin/env python3
#coding:utf-8
"""
[sample] 2次元のパーセプトロンの学習規則の実装例
"""
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

def train(vecW, vecX, label, rho):
    """
    誤り訂正学習
        vecW:  重みベクトル
        vecX:  学習パターンベクトル
        label: クラスラベル (1 or -1)
        rho:   学習率
    """
    is_mod = False # vecWを更新したか
    if (np.dot(vecW,vecX) * label < 0):
        vecW = vecW + label*rho*vecX
        is_mod = True
        return vecW, is_mod
    else:
        return vecW, is_mod


def show2D( vecW, X1, X2, ax ):
    """
    2次元表示
    """
    ax.cla() # 描画クリア

    # 学習用パターン
    ax.scatter(X1[:,1], X1[:,2], c='red' , marker="o")
    ax.scatter(X2[:,1], X2[:,2], c='blue', marker="o")

    # 分離境界線
    x_fig = np.array(np.arange(-6,7,1))
    y_fig = -(vecW[1]/vecW[2])*x_fig - (vecW[0]/vecW[2])

    ax.grid()
    ax.set_xlim(-6,6)
    ax.set_ylim(-6,6)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect('equal', 'datalim')
    ax.plot(x_fig,y_fig, c='green')


def show3D( vecW, X1, X2, ax ):
    """
    3次元表示
    """
    ax.cla() # 描画クリア

    # 学習用パターン
    ax.scatter(X1[:,1], X1[:,2], X1[:,0], c='red' , marker="o")
    ax.scatter(X2[:,1], X2[:,2], X2[:,0], c='blue', marker="o")

    # 分離境界面
    x = np.arange(-6, 6, 0.5)
    y = np.arange(-6, 6, 0.5)
    X, Y = np.meshgrid(x, y)
    Z = -vecW[1]/vecW[0]*X - vecW[2]/vecW[0]*Y 

    ax.grid()
    ax.set_zlim(0, 6)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x0")
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color='green')


def show( vecW, X1, X2, sp1, sp2 ):
    """
    3D + 2D 表示
    """

    show3D(vecW, X1, X2, sp1)
    show2D(vecW, X1, X2, sp2)
    plt.pause(0.1)
    


if __name__ == '__main__':
    tp = lambda x:list(map(float, x.split(',')))
    parser = argparse.ArgumentParser(description=__doc__, epilog='Wの指定は","区切り (-w 1.0,2.0,3.0 のように)')
    parser.add_argument('-r', '--rho',  type=float, default=0.01, help='learning rate rho')
    parser.add_argument('-w', '--wvec', type=tp,    default=[1.0,-1.0,1.0], help='initial W vector')
    args = parser.parse_args()

    train_num = 100 # 学習データ数

    # class1の学習データ
    x1_1 = np.random.rand(train_num//2) * 5 # x成分
    x1_2 = np.random.rand(train_num//2) * 5 # y成分
    label_X1 = np.ones(train_num//2)        # ラベル1

    # class2の学習データ
    x2_1 = (np.random.rand(train_num//2) * 5) * -1 # x成分
    x2_2 = (np.random.rand(train_num//2) * 5) * -1 # y成分
    label_X2 = np.ones(train_num//2) * -1          # ラベル-1

    x0 = np.ones(train_num//2) # x0は常に1

    # 拡張特徴ベクトルを並べた行列
    X1 = np.c_[x0, x1_1, x1_2]
    X2 = np.c_[x0, x2_1, x2_2]

    # 全学習データを統合
    matX   = np.r_[X1, X2]
    labels = np.r_[label_X1, label_X2]

    # 初期の重みベクトル 適当に決める
    vecW = np.array(args.wvec)

    print('初期重みベクトル=', vecW)
    print('ρ=', args.rho)

    # グラフ描画の準備
    fig = plt.figure(figsize=(12,6))
    fig.suptitle('Perceptron (2-dimensional patterns)')
    sp1 = fig.add_subplot(1, 2, 1, projection='3d')
    sp2 = fig.add_subplot(1, 2, 2)

    # Perceptronの学習則
    loop = 1000 # 最大loop回数
    num_update = 0 # Wの更新回数
    for j in range(loop):
        mod = False # 重みベクトルvecWを更新したか
        for vecX, label in zip(matX, labels):
            vecW, is_mod = train(vecW, vecX, label, args.rho) # 学習
            if is_mod:
                mod=True
                num_update += 1
                print('[%2d] W = (%7.3f, %7.3f, %7.3f)' % (num_update, vecW[0], vecW[1], vecW[2]))

                show(vecW, X1, X2, sp1, sp2) # 表示

        if mod == False:
            print("loop回数", j, ", Wの更新回数", num_update, "で収束しました")
            break

    plt.show() # 最後は表示をkeep
