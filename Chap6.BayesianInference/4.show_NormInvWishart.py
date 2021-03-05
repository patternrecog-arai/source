#!/usr/bin/env python3
#coding:utf-8
#
# 正規逆ウィシャート分布の表示 (は無理なので，それからサンプリングされる多変量正規分布を表示)
#
# 	パターンが2次元連続変数で
# 	多変量正規分布(Multivariate Normal Distribution) でモデル化できる場合の
# 	事前，事後分布である正規逆ウィシャート分布の振る舞い
#
#	p(μ,Σ | α,ψ,γ,δ)
#		正規逆Wishart分布を定めるパラメタは (α,ψ,γ,δ)の4つ
#		ψはdxd行列, δはd次元ベクトル
#		分布の変数は, μ(ベクトル)とΣ(行列)
#
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import functools
import scipy.stats as st
import scipy.special as sp
import math
import sys
import distribution as ds
from matplotlib.widgets import Button

x_min, x_max = -6, 6
y_min, y_max = -6, 6
numGrid = 100

x, y = np.meshgrid(np.linspace(x_min,x_max,numGrid), np.linspace(y_min,y_max,numGrid))
xy = np.c_[x.flatten(),y.flatten()]

fig = plt.figure(figsize=(18,6))
ax00 = plt.subplot2grid((2,6), (0,0), colspan=2, rowspan=2)
ax02 = plt.subplot2grid((2,6), (0,2))
ax12 = plt.subplot2grid((2,6), (1,2))
ax03 = plt.subplot2grid((2,6), (0,3))
ax13 = plt.subplot2grid((2,6), (1,3))
ax04 = plt.subplot2grid((2,6), (0,4))
ax14 = plt.subplot2grid((2,6), (1,4))
ax05 = plt.subplot2grid((2,6), (0,5))
ax15 = plt.subplot2grid((2,6), (1,5))

def init_graph():
    """
        Initialize all graphs
    """
    ax00.grid()
    ax02.grid()
    ax12.grid()
    ax03.grid()
    ax13.grid()
    ax04.grid()
    ax14.grid()
    ax05.grid()
    ax15.grid()

    ax00.set_xlim(x_min, x_max)
    ax02.set_xlim(x_min, x_max)
    ax12.set_xlim(x_min, x_max)
    ax03.set_xlim(x_min, x_max)
    ax13.set_xlim(x_min, x_max)
    ax04.set_xlim(x_min, x_max)
    ax14.set_xlim(x_min, x_max)
    ax05.set_xlim(x_min, x_max)
    ax15.set_xlim(x_min, x_max)

    ax00.set_ylim(x_min, x_max)
    ax02.set_ylim(x_min, x_max)
    ax12.set_ylim(x_min, x_max)
    ax03.set_ylim(x_min, x_max)
    ax13.set_ylim(x_min, x_max)
    ax04.set_ylim(x_min, x_max)
    ax14.set_ylim(x_min, x_max)
    ax05.set_ylim(x_min, x_max)
    ax15.set_ylim(x_min, x_max)


init_graph()


# 乱数の種
randSeed = 1
randSeed = 47
#randSeed = 24
#randSeed = 28

#        a)左  b)2上 b)2下  c)3上 c)3下 d)4上 d)4下 e)5上 e)5下
#alpha = [50.0,  7.0, 200.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0]
alpha = [30.0,  5.0, 700.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0]
#gamma = [0.35, 0.35,  0.35, 0.35, 0.35, 0.10, 1.50, 0.35, 0.35]
gamma = [0.35, 0.35,  0.35, 0.35, 0.35, 0.05, 3.00, 0.35, 0.35]
Psi = [np.array([[ 0.8,  0.7],[ 0.7,  0.9]]), # a)
       np.array([[ 0.8,  0.7],[ 0.7,  0.9]]), # b)
       np.array([[ 0.8,  0.7],[ 0.7,  0.9]]), # b)
#       np.array([[ 3.9, -0.5],[-0.5,  1.1]]), # c)
       np.array([[ 1.2, -0.8],[-0.8,  1.1]]), # c)
       np.array([[ 0.3,  0.0],[ 0.0,  1.8]]), # c)
       np.array([[ 0.8,  0.7],[ 0.7,  0.9]]), # d)
       np.array([[ 0.8,  0.7],[ 0.7,  0.9]]), # d)
       np.array([[ 0.8,  0.7],[ 0.7,  0.9]]), # e)
       np.array([[ 0.8,  0.7],[ 0.7,  0.9]])] # e)
Delta = [np.array([ 0.0,  0.0]), # a)
         np.array([ 0.0,  0.0]), # b)
         np.array([ 0.0,  0.0]), # b)
         np.array([ 0.0,  0.0]), # c)
         np.array([ 0.0,  0.0]), # c)
         np.array([ 0.0,  0.0]), # d)
         np.array([ 0.0,  0.0]), # d)
         np.array([ 1.0, -1.0]), # e)
         np.array([-2.0,  0.0])] # e)

colors = ['red', 'green', 'blue', 'magenta', 'brown']
num_sample = 5


def drawAll():
    """
        Draw All graphs
    """
    for p in range(9):
        np.random.seed(randSeed)
        # generate a distribution
        normInvWis = ds.NormInvWishart(alpha[p], alpha[p]*Psi[p], gamma[p], Delta[p])

        # sampling based on the distribution
        sampleMu, sampleSigma = normInvWis.sampling(num_sample)

        pr = np.zeros(len(xy))
        for d in range(num_sample):
            norm = ds.MultivariateNorm(sampleMu[d], sampleSigma[d])
            #print('Mu=', sampleMu[d])
            #print('== Sigma ==\n', sampleSigma[d])
            pr = norm.pdf(xy)
            q = pr.reshape(x.shape)
            if p == 0: ax = ax00
            elif p == 1: ax = ax02
            elif p == 2: ax = ax12
            elif p == 3: ax = ax03
            elif p == 4: ax = ax13
            elif p == 5: ax = ax04
            elif p == 6: ax = ax14
            elif p == 7: ax = ax05
            elif p == 8: ax = ax15
            ax.contour(x,y,q, levels=[0.01,1000], colors=[colors[d],'black']) 
            #ax.contour(x,y,q) 

drawAll() # Draw All graphs

# Slider, Button, RadioButton のfaceの色
axcolor = 'lightgoldenrodyellow'

#
# Next Button
#                  Left  Btm   Wid   Hight 
axButton = plt.axes([0.02, 0.25, 0.05, 0.04])
button = Button(axButton, 'Next', color=axcolor, hovercolor='0.975')

def cls_all():
    """
        Clear All graphs
    """
    ax00.cla()
    ax02.cla()
    ax12.cla()
    ax03.cla()
    ax13.cla()
    ax04.cla()
    ax14.cla()
    ax05.cla()
    ax15.cla()

def cb_next(event):
    """
        'Next' Button Event Callback Func.
    """
    global randSeed
    randSeed += 1
    print('Seed=', randSeed)
    
    cls_all()
    init_graph()
    drawAll()
    fig.canvas.draw_idle()

button.on_clicked(cb_next) # Event callback func. のセット


#
# Save Button
#                  Left  Btm   Wid   Hight 
axSave = plt.axes([0.02, 0.15, 0.05, 0.04])
buttonSave = Button(axSave, 'Save', color=axcolor, hovercolor='0.975')

def cb_save(event):
    """
        'Save' Button Event Callback Func.
    """
    plt.savefig('show_NormInvWishart.py.svg', format='svg')

buttonSave.on_clicked(cb_save) # Event callback func. のセット



#
# Quit Button
#                  Left  Btm   Wid   Hight 
axQuit = plt.axes([0.02, 0.05, 0.05, 0.04])
buttonQuit = Button(axQuit, 'Quit', color=axcolor, hovercolor='0.975')

def cb_quit(event):
    """
        'Quit' Button Event Callback Func.
    """
    sys.exit()

buttonQuit.on_clicked(cb_quit) # Event callback func. のセット

# Plot
plt.show()
