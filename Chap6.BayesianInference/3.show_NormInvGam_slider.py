#!/usr/bin/env python3
#coding:utf-8
#
# 正規逆ガンマ分布の表示
#
#	パターンが1変量連続変数で
#	1次元正規分布(Univariate Normal Distribution)でモデル化できる場合の
#	事前，事後分布である正規逆ガンマ分布の振る舞い
#
# 	p(μ,σ2 | α,β,γ,δ)
#		正規逆ガンマ分布を定めるパラメタは(α,β,γ,δ)の4つ
#		分布の変数は(μ,σ2)の2つで定義域は -∞ < μ < ∞,  σ2 > 0
#
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import functools
import scipy.stats as st
import math
import distribution as ds
import sys
import distribution as ds
from matplotlib.widgets import Slider, Button, RadioButtons

# NormInvGam distribution parameters
alpha_min, alpha_max, alpha0 =  0.0, 5.0, 1.0
beta_min,  beta_max,  beta0  =  0.0, 5.0, 1.0
gamma_min, gamma_max, gamma0 =  0.0, 5.0, 1.0
delta_min, delta_max, delta0 = -2.0, 2.0, 0.0
    
rangeMu = 2
rangeVar = 4
numGrid = 100
x, y = np.meshgrid(np.linspace(-rangeMu,rangeMu,numGrid), np.linspace(rangeVar/numGrid,rangeVar,numGrid))

# グラフの準備
fig, ax = plt.subplots(figsize=(6,6))
plt.subplots_adjust(left=0.1, bottom=0.35)

# 初期分布の描画
d = ds.NormInvGam(alpha0, beta0, gamma0, delta0) 
p = d.pdf(x, y)
ax.pcolormesh(x, y, p, cmap=plt.cm.hot)		# fast

# Slider, Button, RadioButton のfaceの色
axcolor = 'lightgoldenrodyellow'

#
# Slider
#                  Left  Btm   Wid   Hight 
axAlpha = plt.axes([0.15, 0.25, 0.7, 0.03], facecolor=axcolor)
axBeta  = plt.axes([0.15, 0.20, 0.7, 0.03], facecolor=axcolor)
axGamma = plt.axes([0.15, 0.15, 0.7, 0.03], facecolor=axcolor)
axDelta = plt.axes([0.15, 0.10, 0.7, 0.03], facecolor=axcolor)

sAlpha = Slider(axAlpha, r'$\alpha$', alpha_min, alpha_max, valinit=alpha0)
sBeta  = Slider(axBeta , r'$\beta$' , beta_min,  beta_max,  valinit=beta0)
sGamma = Slider(axGamma, r'$\gamma$', gamma_min, gamma_max, valinit=gamma0)
sDelta = Slider(axDelta, r'$\delta$', delta_min, delta_max, valinit=delta0)

def cb_update(val):
    """
        Slider Event Callback Func.
    """
    d = ds.NormInvGam(sAlpha.val, sBeta.val, sGamma.val, sDelta.val) 
    p = d.pdf(x, y)
    ax.pcolormesh(x, y, p, cmap=plt.cm.hot)		# fast
    #print('>>> # of figures = ', plt.get_fignums())
    fig.canvas.draw_idle()

sAlpha.on_changed(cb_update) # Event callback func. のセット
sBeta.on_changed(cb_update)  # Event callback func. のセット
sGamma.on_changed(cb_update) # Event callback func. のセット
sDelta.on_changed(cb_update) # Event callback func. のセット


#
# Reset Button
#                  Left  Btm   Wid   Hight 
resetax = plt.axes([0.1, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

def cb_reset(event):
    """
        Reset Button Event Callback Func.
    """
    # Reset Sliders
    sAlpha.reset()
    sBeta.reset()
    sGamma.reset()
    sDelta.reset()

button.on_clicked(cb_reset) # Event callback func. のセット


#
# Quit Button
#                 Left  Btm   Wid   Hight 
quitax = plt.axes([0.8, 0.025, 0.1, 0.04])
button2 = Button(quitax, 'Quit', color=axcolor, hovercolor='0.975')

def cb_quit(event):
    """
        Quit Button Event Callback Func.
    """
    sys.exit()

button2.on_clicked(cb_quit) # Event callback func. のセット

# Plot
plt.show()
