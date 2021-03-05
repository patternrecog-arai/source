#!/usr/bin/env python3
#coding: utf-8
#------------------------------------------------
#
# 2 class識別 (XORの学習)
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


if __name__ == "__main__":
    """
    Main function
        -l (--loop)     <loop count>                  (default: 10000)
        -r (--rho)      <learning rate>               (default: 0.1)
        -u (--num_unit) <unit number of middle layer> (default: 3)
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-l', '--loop', type=int, default=10000, help='loop count')
    parser.add_argument('-r', '--rho', type=float, default=0.1, help='learning parameter rho')
    parser.add_argument('-u', '--num_unit', type=int, default=3, help='unit number of middle layer')
    args = parser.parse_args()   

    rho = args.rho
    """ XOR の学習 """
    mlp = nn.MultiLayerPerceptron(2, args.num_unit, 1, "tanh", "sigmoid")
    print("ネットワーク構造(入力，中間，出力, 中間層活性化関数, 出力層活性化関数): ", mlp.get_arch())
    #mlp = MultiLayerPerceptron(2, 2, 1, "sigmoid", "sigmoid")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    g3 = np.array([[0], [1], [1], [0]])

    # 入力データの最初の列にバイアスユニットの入力1を追加
    X = np.hstack([np.ones([X.shape[0], 1]), X])

    # Learning
    mlp.fit(X, g3, args.rho, args.loop)

    # Recognition
    g2, g3 = mlp.predict(X)
    for i in range(4):
        print(X[i, 1:], g3[i])

