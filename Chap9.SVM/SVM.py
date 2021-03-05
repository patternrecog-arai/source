#!/usr/bin/env python3
#coding:utf-8
# 最急降下法によるSVM解法
#
# Hard margin と Soft margin に対応 
# (Command line option -C で設定可能. default: Hard Margin)
# (学習用データはランダムに生成．
#  Hard Margin用，Soft Margin用を変えることができます．
#  "class1の学習データ" "class2の学習データ" という文字列を
#  検索して修正箇所を探してください)
# 全て行列＋ベクトル演算で書き直してあるので，速いです．
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys


def train(matX, labels, alpha, beta, eta, C):
    eta_al = eta / len(labels) # update ratio of alpha
    eta_be = eta / len(labels) # update ratio of beta
    #C = 0.5 # Soft margin
    #                1
    # L(α) = α^t 1 - - α^t H α, where α^t t = 0 and α_i >= 0
    #                2
    # add regularization term 
    #                1             1
    # L(α) = α^t 1 - - α^t H α - β - || α^t t ||^2 
    #                2             2
    #                1            1
    # L(α) = α^t 1 - - α^t H α -  - β α^t t t^t α 
    #
    # 1 = [1,1,1,......1]
    # t = [t_1, t_2, ... , t_N]
    # H: (H_{nm} = t_n t_m k(x_n, x_m) = t_n t_m x_n^t x_m)
    #
    #
    #  ∂L        
    # ---- = 1 - H a - β  t t^t a   (1)
    #  ∂α      
    #             ∂L
    # α' = α + η ----  (2)
    #             ∂α
    #
    # (1),(2)式の実装
    vecOne = np.ones(alpha.size)
    matLabels = np.outer(labels, labels)
    matH = matLabels * matX.dot(matX.T)
    delta = vecOne - matH.dot(alpha) - beta * matLabels.dot(alpha)
    alpha += eta_al * delta
    for i in range(len(alpha)):
        if alpha[i] < 0.0: alpha[i] = 0.0 # α_i >= 0
        if alpha[i] > C:   alpha[i] = C   # α_i <= C  (Soft margin)
    #  ∂L    1
    #  --- =  - || α^t t ||^2 
    #  ∂β   2
    #               ∂L
    # β' = β + η ---
    #               ∂β
    for i in range(len(alpha)):     # αはベクトルの更新, βはスカラの更新．同じ重みにしたいのでこうした．
        #beta += eta_be * (alpha.dot(labels) ** 2) / 2.0
        beta += 10.0*eta_be * (alpha.dot(labels) ** 2) / 2.0 # 拘束条件を満たすように制約項の重みをさらに10倍

    return alpha, beta



def show2D( vecW, tData1, tData2, index1, index2, ax ):
    """
    2次元表示
    """
    ax.cla() # 描画クリア

    # 学習用パターン
    ax.scatter(tData1[:,1], tData1[:,2], c='red' , marker="o")
    ax.scatter(tData2[:,1], tData2[:,2], c='blue', marker="o")

    tData1s = tData1[index1]
    tData2s = tData2[index2]
    ax.scatter(tData1s[:,1], tData1s[:,2], c='red' , marker="s", s=100)
    ax.scatter(tData2s[:,1], tData2s[:,2], c='blue', marker="s", s=100)

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


def show3D( vecW, tData1, tData2, index1, index2, ax ):
    """
    3次元表示
    """
    ax.cla() # 描画クリア

    # 学習用パターン
    ax.scatter(tData1[:,1], tData1[:,2], tData1[:,0], c='red' , marker="o")
    ax.scatter(tData2[:,1], tData2[:,2], tData2[:,0], c='blue', marker="o")

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
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.3, color='green')


def show( vecW, tData1, tData2, index1, index2, sp1, sp2 ):
    """
    3D + 2D 表示
    """

    show3D(vecW, tData1, tData2, index1, index2, sp1)
    show2D(vecW, tData1, tData2, index1, index2, sp2)
    plt.pause(0.01)



def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text
    

def generateTrainData( train_num, seed, overwrap ):
    """ Generate Training Data from random data """
    """
        class1:  0<x<1,  0<y<1 の正方形 と
        class2: -1<x<0, -1<y<0 の正方形 を 

        class1 は(-overwrap/2, -overwrap/2)だけ平行移動
        class2 は( overwrap/2,  overwrap/2)だけ平行移動　した領域内を5倍に拡大し
        (bias,bias)だけ平行移動し，
        その領域内をランダムサンプリング

        biasは分布全体の平行移動
        overwrapは2クラスの分布の重なり具合を決める．
        overwrap=0 なら重なり無しで領域は接している   (Hard margin用)
        overwrap<0 なら重なり無しで領域間は離れる     (Hard margin用)
        overwrap>0 なら領域間は重なる                 (Soft margin用)
        overwrap=1 (100%)で完全に重なる

    """
    #overwrap    = 1.0            # クラス間の重なり(100%)
    #overwrap    = 0.5            # クラス間の重なり( 50%)
    #overwrap    = 0.0            # クラス間の重なり(  0%)
    bias      = 1.5            # 偏り

    # Set Random Seed
    if seed != -1:
        np.random.seed(seed)

    # class1の学習データ
    tData1_x = (np.random.rand(train_num//2) - overwrap/2.0) * 5 + bias  # x成分
    tData1_y = (np.random.rand(train_num//2) - overwrap/2.0) * 5 + bias  # y成分 (For Soft Margin SVM)
    tData1_label = np.ones(train_num//2)                         # ラベル1

    # class2の学習データ
    tData2_x = (np.random.rand(train_num//2) * -1 + overwrap/2.0) * 5 + bias  # x成分
    tData2_y = (np.random.rand(train_num//2) * -1 + overwrap/2.0) * 5 + bias  # y成分 (For Soft Margin SVM)
    tData2_label = np.ones(train_num//2) * -1                           # ラベル-1

    x0 = np.ones(train_num//2) # 拡張ベクトル用x0は常に1

    # 特徴ベクトルを並べた行列
    tData1 = np.c_[tData1_x, tData1_y]
    tData2 = np.c_[tData2_x, tData2_y]
    # 拡張特徴ベクトルを並べた行列
    tData1e = np.c_[x0, tData1_x, tData1_y]
    tData2e = np.c_[x0, tData2_x, tData2_y]

    # 全学習データを統合
    matX   = np.r_[tData1, tData2]
    labels = np.r_[tData1_label, tData2_label]

    return matX, labels, tData1e, tData2e



if __name__ == '__main__':
    """
    Main Function
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-n', '--train_num', type=int, default=20,
                        help='number of training data (default=20)')
    parser.add_argument('-o', '--overwrap', type=float, default=0.0,
                        help='overwrap rate of 2 class data (default=0.0[no overwrap])')
    parser.add_argument('-i', '--iter_num', type=int, default=10000,
                        help='itteration number for training (default=10000)')
    parser.add_argument('-s', '--seed', type=int, default=-1,
                        help='random seed number (default:Random)')
    parser.add_argument('-e', '--eta', type=float, default=0.1,
                        help='eta (default=0.1)')
    parser.add_argument('-C', '--softC', type=float, default=1.e10,
                        help='soft margin C (typically, C=0.5) (default C=1.e10 means Hard Margin)')

    args = parser.parse_args()



    # 学習用データの生成
    train_num = args.train_num # 学習データ数
    matX, labels, tData1e, tData2e = generateTrainData( train_num, args.seed, args.overwrap )

    # グラフ描画の準備
    fig = plt.figure(figsize=(16,8))
    fig.suptitle('Support Vector Machine (2-dimensional patterns)')
    sp1 = fig.add_subplot(1, 2, 1, projection='3d')
    sp2 = fig.add_subplot(1, 2, 2)

    # 最急降下法での解法
    loop = args.iter_num # 最大loop回数

    # Lagrange Multipliers
    alpha = np.zeros(train_num)
    beta  = 1.0
    C     = args.softC
    eta   = args.eta

    disp_int = 1 # 表示間隔
    for j in range(loop):
        alpha, beta = train(matX, labels, alpha, beta, eta, C) # 学習

        index = alpha > 0   # サポートベクトル(alpha>0)の配列index
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        w = ((alpha * labels).T).dot(matX)
        b = (labels[index] - matX[index].dot(w)).mean()
        vecW = np.r_[b,w]
        print("\r[", j, "] ", end="")
        print("alpha = ", alpha, end="")
        #print(" beta = ", beta, end="")
        print(", w = ",  vecW , "\033[1A", end="")
        #print("support vector's index = ", index)
        index1 = index[:train_num//2]
        index2 = index[train_num//2:]
        if j==10: disp_int = 2
        if j==40: disp_int = 5
        if j==100: disp_int = 10
        if j==200: disp_int = 100
        if j==1000: disp_int = 200
        if j%disp_int == 0:
            show(vecW, tData1e, tData2e, index1, index2, sp1, sp2) # 表示

    show(vecW, tData1e, tData2e, index1, index2, sp1, sp2) # 表示
    plt.show() # 最後は表示をkeep
    print("\n\n")
