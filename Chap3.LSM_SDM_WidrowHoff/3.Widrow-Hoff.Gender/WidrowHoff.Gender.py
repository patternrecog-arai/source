#!/usr/bin/env python3
#coding:utf-8
"""
Widrow-Hoffの学習則による性別識別器の学習
2次元データ(身長，体重)
"""
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import pandas as pd

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
    eps = np.dot(vecW , vecX) - label   # epsilon
    dJp = 2.0 * eps * vecX              # Jpの微分
    vecW = vecW - rho * dJp             # W更新
    Jp = eps ** 2                       # Jp

    return vecW, Jp


def recog(vecW, matX, labels):
    """
    未知バターンの認識
        vecW:   重みベクトル
        matX:   学習パターン行列
        labels: 教師信号ベクトル (1 or -1)
    """
    estimated = np.dot(matX, vecW)

    TP = ((labels > 0) * (estimated > 0)).sum()
    TN = ((labels < 0) * (estimated < 0)).sum()
    FP = ((labels < 0) * (estimated > 0)).sum()
    FN = ((labels > 0) * (estimated < 0)).sum()
    print("          ------------------")
    print("          |      |推定結果 |")
    print("          |      | Pos Neg |")
    print("          |------+---------|")
    print("|TP FN|   |   Pos| %3d %3d |" % (TP, FN))
    print("|     | = |真    |         |")
    print("|FP TN|   |   Neg| %3d %3d |" % (FP, TN))
    print("          ------------------")
    accuracy = (TP+TN) / len(labels)
    precision = TP / (TP+FP)
    recall = TP / (TP+FN)
    F_measure = 2*precision*recall / (precision + recall)
    print("Accuracy  = %5.2f %%" % (accuracy*100))
    print("Precision = %5.2f %%" % (precision*100))
    print("Recall    = %5.2f %%" % (recall*100))
    print("F-measure = %5.2f %%" % (F_measure*100))


def show2D( vecW, x1, x2, ax ):
    """
    2次元表示
    """
    ax.cla() # 描画クリア

    ax.grid()
    ax.set_xlim(-3,3)
    ax.set_ylim(-3,3)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect('equal', 'datalim')

    # 学習用パターン
    ax.scatter(x1[:,1], x1[:,2], c='red' , marker="o")
    ax.scatter(x2[:,1], x2[:,2], c='blue', marker="o")

    # 分離境界線
    x_fig = np.array(np.arange(-6,7,1))
    y_fig = -(vecW[1]/vecW[2])*x_fig - (vecW[0]/vecW[2])

    ax.plot(x_fig,y_fig, c='green')


def show3D( vecW, x1, x2, ax ):
    """
    3次元表示
    """
    ax.cla() # 描画クリア

    ax.grid()
    ax.set_zlim(0, 3)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x0")

    # 学習用パターン
    ax.scatter(x1[:,1], x1[:,2], x1[:,0], c='red' , marker="o")
    ax.scatter(x2[:,1], x2[:,2], x2[:,0], c='blue', marker="o")

    # 分離境界面
    x = np.arange(-3, 3, 0.5)
    y = np.arange(-3, 3, 0.5)
    X, Y = np.meshgrid(x, y)
    Z = -vecW[1]/vecW[0]*X - vecW[2]/vecW[0]*Y 

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.3, color='green')


def show( vecW, x1, x2, sp1, sp2 ):
    """
    3D + 2D 表示
    """
    print("w = ",  vecW )

    show3D(vecW, x1, x2, sp1)
    show2D(vecW, x1, x2, sp2)
    plt.pause(0.01)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-r', '--rho', type=float, default=0.001, help='learning rate rho')
    parser.add_argument('-l', '--loop', type=int, default=50, help='loop number')
    args = parser.parse_args()    


    # class1の学習データ
    male         = pd.read_csv(filepath_or_buffer='./Pattern/genderPattern2learn.male.dat', encoding='ms932', sep=',', header=None)
    male_values  = male.values
    label_male   = male_values[:,0]      # ラベルはレコードの先頭に入っている
    male_data    = male_values[:,1:]     # パターンはレコードの2カラム目から入っている
    num_male     = male.values.shape[0]  # 学習データ数(男)
    x0_male      = np.ones(num_male)     # x0は常に1

    # class2の学習データ
    female         = pd.read_csv(filepath_or_buffer='./Pattern/genderPattern2learn.female.dat', encoding='ms932', sep=',', header=None)
    female_values  = female.values
    label_female   = female_values[:,0]      # ラベルはレコードの先頭に入っている
    female_data    = female_values[:,1:]     # パターンはレコードの2カラム目から入っている
    num_female     = female.values.shape[0]  # 学習データ数(女)
    x0_female      = np.ones(num_female)     # x0は常に1

    train_num = num_male + num_female
    print('学習パターン数 = ', train_num)

    # 正規化
    all_data = np.r_[male_data, female_data]
    mean = all_data.mean(axis=0)
    std  = all_data.std(axis=0)
    print("平均= ", mean, "標準偏差= ", std)
    male_data   = (male_data-mean)/std
    female_data = (female_data-mean)/std


    # 拡張特徴ベクトルを並べた行列
    x1 = np.c_[x0_male, male_data]
    x2 = np.c_[x0_female, female_data]

    # 全学習データを統合
    matX   = np.r_[x1, x2]
    labels = np.r_[label_male, label_female]

    vecW = np.array([-1,1,1]) #初期の重みベクトル 適当に決める


    # グラフ描画の準備
    fig = plt.figure(figsize=(12,6))
    fig.suptitle('Widrow - Hoff Rule (2-dimensional patterns)')
    sp1 = fig.add_subplot(1, 2, 1, projection='3d')
    sp2 = fig.add_subplot(1, 2, 2)

    # Widrow-Hoffの学習則での解法
    for j in range(args.loop):
        maxJp = 0
        aveJp = 0
        for vecX, label in zip(matX, labels):
            vecW, Jp = train(vecW, vecX, label, args.rho) # 学習
            if Jp > maxJp: maxJp = Jp
            aveJp += Jp

        aveJp /= len(matX)
        print("[%d] aveJp=%7.3f  maxJp=%7.5f " % (j, aveJp, maxJp), end="")
        show(vecW, x1, x2, sp1, sp2) # 表示

    show(vecW, x1, x2, sp1, sp2) # 表示

    
    # 未知テストデータの識別
    test        = pd.read_csv(filepath_or_buffer='./Pattern/genderPattern2recog.dat', encoding='ms932', sep=',', header=None)
    test_values = test.values
    labels_test = test_values[:,0]      # ラベルはレコードの先頭に入っている
    test_data   = test_values[:,1:]     # パターンはレコードの2カラム目から入っている
    num_test    = test.values.shape[0]  # テストデータ数
    x0_test      = np.ones(num_test)    # x0は常に1

    test_data = (test_data-mean)/std
    matTest    = np.c_[x0_test, test_data]

    recog(vecW, matTest, labels_test) # 識別

    plt.show() # 最後は表示をkeep
