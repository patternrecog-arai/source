#!/usr/bin/env python3
#coding:utf-8
# 最急降下法による評価関数最小解
# 2次元データ
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
    eps = np.dot(vecW , vecX) - label	# epsilon
    dJp = 2.0 * eps * vecX 				# Jpの微分
    vecW = vecW - rho * dJp				# W更新
    Jp = eps ** 2						# Jp

    return vecW, Jp

def recog(matW, matX, labels):
    """
    未知バターンの認識
        vecW:   重みベクトル
        matX:   学習パターン行列
        labels: 教師信号ベクトル (1 or -1)
    """
    estimated = np.dot(matX, matW.T)

    estClass = np.argmax(estimated, axis=1)

    cMat = np.zeros((10,10), dtype=int) # confusion matrix

    # Confusion Matrixの生成
    for ans, est in zip(labels, estClass):
        cMat[ans,est] += 1

    # 誤認識パターンの表示
    fig = plt.figure(figsize=(8,8))

    sp = []
    for no in range(25):
        sp.append(fig.add_subplot(5, 5, no+1))
        sp[no].axis('off')
    
    no = 0
    for ans, est, vecX in zip(labels, estClass, matX):
        if ans != est:
            img = vecX[1:].reshape(7,5)
            sp[no%25].cla()
            sp[no%25].axis('off')
            sp[no%25].imshow(img)
            no += 1
            plt.pause(0.01)
            print(" ーーーーー   [No. %d] %d --> %d と誤った" % (no, ans, est))
            for y in range(7):
                print("|", end="")
                for x in range(5):
                    if img[y,x] == 1:
                        print("██", end="")
                    else:
                        print("　", end="") 
                print("|")
            print(" ーーーーー \n")

    # Confusion Matrixの表示
    print( cMat ) 

    print("認識率= %6.3f %% (%d/%d)" % (np.trace(cMat) / len(matX) * 100.0, np.trace(cMat), len(matX)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-r', '--rho', type=float, default=0.001, help='learning rate rho')
    parser.add_argument('-l', '--loop', type=int, default=500, help='loop number')
    args = parser.parse_args()    


    # 学習データ
    df_train      = pd.read_csv(filepath_or_buffer='./Pattern/pattern2learn.dat', encoding='ms932', sep=',', header=None)
    train_values  = df_train.values
    label_train   = train_values[:,0]      # ラベルはレコードの先頭に入っている
    train_data    = train_values[:,1:]     # パターンはレコードの2カラム目から入っている
    num_train     = len(train_data)        # 学習データ数
    x0_train      = np.ones(num_train)     # x0は常に1

    order = train_data.shape[1] + 1	# 拡張特徴ベクトルの次元数
    matW = np.zeros((10,order))

    print('学習パターン数 = ', num_train)
    for i in range(10):
        print("#### Now learning Digit ", i, "####" )
        # 正例のデータ
        posLabel = (label_train == i)
        posData = train_data[posLabel]
        numPos = len(posData)
        x0Pos = np.ones(numPos)
        posB = np.ones(numPos)  # 正例の教師信号は1


        # 負例のデータ
        negLabel = (label_train != i)
        negData = train_data[negLabel]
        numNeg = len(negData)
        x0Neg = np.ones(numNeg)
        negB = np.ones(numNeg) * (-1) # 負例の教師信号は-1

        print("正例: %d  負例: %d" % (numPos, numNeg))

        # 拡張特徴ベクトルを並べた行列
        posDataE = np.c_[x0Pos, posData]
        negDataE = np.c_[x0Neg, negData]

        # 全学習データを統合
        matX = np.r_[posDataE, negDataE]
        B    = np.r_[posB, negB]

        #np.random.seed(3) # 3:2748
        vecW = np.random.rand(matX.shape[1])  # 初期の重みベクトル36次元 乱数で決める
        

        # Widrow-Hoffの学習則での解法
        loop = args.loop # 最大loop回数
        for j in range(loop):
            if j%10==0: # 10回に一度並び替え
                # データをランダムに並び替える
                index = np.random.permutation(np.arange(len(matX))) 
                matXshuffle = matX[index]
                Bshuffle = B[index]

            maxJp = 0
            aveJp = 0
            for vecX, label in zip(matXshuffle, Bshuffle):
                vecW, Jp = train(vecW, vecX, label, args.rho) # 学習
                if Jp > maxJp: maxJp = Jp
                aveJp += Jp
            aveJp /= len(matX)
            #eprint("[%d] maxJp=%7.3f  aveJp=%7.5f " % (j, maxJp, aveJp))

            if aveJp < 0.1: break;
        matW[i] = vecW

    for i in range(10):
        print("W[",i,"]=",matW[i])

    # 未知テストデータ
    test        = pd.read_csv(filepath_or_buffer='./Pattern/pattern2recog.dat', encoding='ms932', sep=',', header=None)
    test_values = test.values
    labels_test = test_values[:,0]      # ラベルはレコードの先頭に入っている
    test_data   = test_values[:,1:]     # パターンはレコードの2カラム目から入っている
    num_test    = test.values.shape[0]  # テストデータ数
    x0_test     = np.ones(num_test)     # x0は常に1
    matTest = np.c_[x0_test, test_data]

    recog(matW, matTest, labels_test)

    plt.show()
    sys.exit()
