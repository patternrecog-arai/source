#!/usr/bin/env python3
#coding: utf-8
#------------------------------------------------
#
# 関数fitting
#
#------------------------------------------------
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import argparse
import time
import neuralnet as nn

def heviside(x):
    """Heviside 関数"""
    return 0.5 * (np.sign(x) + 1)


def show(loop, trainIn, trainOut, g2, g3, error):
    """
    関数fitting状態の表示
        訓練データ: 青丸で表示
        NN出力    : 赤線で表示
        NN中間層  : 破線で表示
    """
    # グラフをクリア
    ax.cla()
    ax.plot(trainIn, trainOut, 'bo')
    # 訓練データの各x に対するニューラルネットの出力を赤線で表示
    ax.plot(trainIn, g3, 'r-')
    # 各重みの出力を点線で表示
    for i in range(NUM_HIDDEN):
        ax.plot(trainIn, g2[:,i], 'k--')
    ax.set_title("loop: %d squared error: %f" % (loop, error))
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_aspect('equal', 'datalim')
    ax.grid()
    plt.pause(0.01)



if __name__ == '__main__':
    """
    Main function
        -s (--signal)   <sin|cos|square|abs|heviside> (default: sin)
        -n (--num)      <# of training data>          (default: 50)
        -l (--loop)     <loop count>                  (default: 1000)
        -r (--rho)      <learning rate>               (default: 0.1)
        -u (--num_unit) <unit number of middle layer> (default: 3)
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-s', '--signal', type=str, default='sin', help='input signal type (sin|cos|square|abs|heviside)')
    parser.add_argument('-n', '--num', type=int, default=50, help='data number')
    parser.add_argument('-l', '--loop', type=int, default=1000, help='loop count')
    parser.add_argument('-r', '--rho', type=float, default=0.1, help='learning parameter rho')
    parser.add_argument('-u', '--num_unit', type=int, default=3, help='unit number of middle layer')
    args = parser.parse_args()   
    N = args.num
    rho = args.rho
    NUM_HIDDEN = args.num_unit

    print("Loop回数:     ", args.loop)
    print("訓練データ数: ", args.num)
    print("中間層Unit数: ", args.num_unit)
    print("学習率:       ", args.rho)

    # 訓練データ作成
    trainIn = np.linspace(-1.5, 1.5, N).reshape(N, 1)
    if args.signal == 'sin':
        trainOut = np.sin(trainIn)   # sin(x)
    elif args.signal == 'cos':
        trainOut = np.cos(trainIn)   # cos(x)
    elif args.signal == 'square':
        trainOut = trainIn * trainIn # x^2
    elif args.signal == 'abs':
        trainOut = np.abs(trainIn)   # |x|
    else:
        trainOut = heviside(trainIn) # H(x)

    # 入力ベクトルを1次元拡張
    ones = np.ones(len(trainIn))
    trainInE = np.insert(trainIn, 0, 1, axis=1)

    # グラフ描画準備
    fig = plt.figure(figsize=(7,7))
    title = "Function approximation by Multi-layer perceptron (" + args.signal + ")"
    fig.suptitle(title, fontsize=15)
    ax = fig.add_subplot(111)

    # Neural Network の生成 (入力層数, 中間層数, 出力層数, 中間層活性化関数, 出力層活性化関数)
    mlp = nn.MultiLayerPerceptron(1, args.num_unit, 1, "tanh", "identity")

    # Neural Network の学習
    ts = time.time() # 開始時刻
    for loop in range(args.loop): # loop回数
        mlp.fit_local(trainInE, trainOut, args.rho)   # 1 episode分のデータでWを更新
        g2, g3 = mlp.predict(trainInE)                # 1 episode分のデータに対する g2, g3を予測
        error = mlp.error_mse(g3, trainOut)           # 1 episode分の予測誤差を計算
        show(loop, trainIn, trainOut, g2, g3, error ) # グラフ表示
    te = time.time() # 終了時刻
    print ("compute : %.3f sec" % (te-ts))

    plt.show()

