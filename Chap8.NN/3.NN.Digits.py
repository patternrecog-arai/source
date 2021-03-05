#!/usr/bin/env python3
#coding: utf-8
#------------------------------------------------
#
# 多class識別 (Digitsの学習) 
#    (7x5)画像データから数字を推定
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

def disp_digit( img, h, w ):
    """ 数字の表示 """
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


if __name__ == '__main__':
    """ 
    Main function
        -l (--loop)     <loop count>                  (default: 1000)
        -r (--rho)      <learning rate>               (default: 0.1)
        -u (--num_unit) <unit number of middle layer> (default: 50)
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-l', '--loop', type=int, default=100, help='loop number')
    parser.add_argument('-r', '--rho', type=float, default=0.1, help='learning rate rho')
    parser.add_argument('-u', '--num_unit', type=int, default=20, help='unit number of middle layer')
    parser.add_argument('-s', '--seed', type=int, default=-1, help='random seed')
    args = parser.parse_args()    

    # Neural Network の生成 (入力層数, 中間層数, 出力層数, 中間層活性化関数, 出力層活性化関数)
    mlp = nn.MultiLayerPerceptron(35, args.num_unit, 10, "tanh", "softmax", args.seed)

    # 学習データ
    df_train      = pd.read_csv(filepath_or_buffer='./data.digits/pattern2learn.dat', encoding='ms932', sep=',', header=None)
    train_values  = df_train.values
    train_labels   = train_values[:,0]     # ラベルはレコードの先頭に入っている
    train_data    = train_values[:,1:]     # パターンはレコードの2カラム目から入っている
    num_train     = len(train_data)        # 学習データ数
    x0_train      = np.ones(num_train)     # x0は常に1
    train_dataE   = np.c_[x0_train, train_data] # 拡張特徴行列
    train_supervised      = mlp.conv_onehot(train_labels, 10)
    print('学習パターン数 = ', num_train)

    # テストデータ
    test        = pd.read_csv(filepath_or_buffer='./data.digits/pattern2recog.dat', encoding='ms932', sep=',', header=None)
    test_values = test.values
    test_labels = test_values[:,0]      # ラベルはレコードの先頭に入っている
    test_data   = test_values[:,1:]     # パターンはレコードの2カラム目から入っている
    num_test    = test.values.shape[0]  # テストデータ数
    x0_test     = np.ones(num_test)     # x0は常に1
    test_dataE  = np.c_[x0_test, test_data] # 拡張特徴行列
    test_supervised = mlp.conv_onehot(test_labels, 10) # one hot表現

    # Neural Network の学習
    ts = time.time() # 開始時刻
    for loop in range(args.loop): # loop回数
        mlp.fit_local(train_dataE, train_supervised, args.rho)         # 1 episode分のデータでWを更新
        g2, g3 = mlp.predict(train_dataE)                              # 1 episode分のデータに対する g2, g3を予測
        entropy = mlp.error_Nclass_cross_entropy(g3, train_supervised) # 1 episode分のcross entropyを計算
        print("[%d] cross entropy=%7.3f" % (loop, entropy))
        
        if loop%10 == 0:
            # 未知テストデータの識別
            print("======== テストデータに対する評価 ========")
            mlp.infer_Nclass(test_dataE, test_supervised, 10)
    te = time.time() # 終了時刻
    print ("compute : %.3f sec" % (te-ts))

    # 学習データの識別
    print("======== 学習データに対する評価 ========")
    mlp.infer_Nclass(train_dataE, train_supervised, 10)


    # 未知テストデータの識別
    print("======== テストデータに対する評価 ========")
    result_infer, cMat = mlp.infer_Nclass(test_dataE, test_supervised, 10)


    # 誤認識パターンの表示
    fig = plt.figure(figsize=(8,8))
    
    no = 0
    for ans, est, vecX in zip(test_labels, result_infer, test_data):
        if ans != est:
            img = vecX.reshape(7,5)
            sp = fig.add_subplot(8, 8, (no%64)+1)
            sp.axis('off')
            no += 1
            disp_digit(img, 7, 5) # 端末に表示
            sp.imshow(img)        # 画像として表示
            plt.pause(0.01)

    print(cMat)
    print("認識率= %6.3f %% (%d/%d)" % (np.trace(cMat) / len(test_data) * 100.0, np.trace(cMat), len(test_data)))

    g2, g3 = mlp.predict(test_dataE)
    print("cross entropy = %7.3f" % (mlp.error_Nclass_cross_entropy(g3, test_supervised)))

    plt.show()

