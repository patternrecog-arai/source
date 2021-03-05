#!/usr/bin/env python3
#coding:utf-8
#
# Beta分布の表示
#
#	パターンが離散2値変数で
#	ベルヌーイ分布(Bernoulli Distribution)でモデル化できる場合の
#	事前，事後分布であるBeta分布の振る舞い
#
# 	p(λ|α,β)
#		Beta分布を定めるパラメタは(α，β)の2つ
#		分布の変数はλでその定義域は[0,1]
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
from matplotlib.widgets import Slider, Button, RadioButtons

# Beta distribution parameters
#max_param = 12
max_param = 100
alpha_min, alpha_max, alpha0 = 0, max_param, 1.0
beta_min,  beta_max,  beta0  = 0, max_param, 1.0
    
# Drawing range of Beta distribution
x_min, x_max = 0,  1
y_min, y_max = 0, 12
x_step = 0.01
x = np.arange(x_min, x_max, x_step)

fig, ax = plt.subplots(figsize=(6,6))
plt.subplots_adjust(left=0.1, bottom=0.25)

ax.grid()
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

dist = ds.Beta(alpha0, beta0)
line2dObj, = plt.plot(x, dist.pdf(x), lw=2, color='red')

# Slider, Button, RadioButton のfaceの色
axcolor = 'lightgoldenrodyellow'

#
# Slider
#                  Left  Btm   Wid   Hight 
axAlpha = plt.axes([0.1, 0.15, 0.8, 0.03], facecolor=axcolor)
axBeta  = plt.axes([0.1, 0.10, 0.8, 0.03], facecolor=axcolor)

#sAlpha = Slider(axAlpha, 'Alpha', alpha_min, alpha_max, valinit=alpha0)
#sBeta  = Slider(axBeta , 'Beta',  beta_min,  beta_max,  valinit=beta0)
sAlpha = Slider(axAlpha, r'$\alpha$', alpha_min, alpha_max, valinit=alpha0)
sBeta  = Slider(axBeta , r'$\beta$',  beta_min,  beta_max,  valinit=beta0)

def cb_update(val):
    """
        Slider Event Callback Func.
    """
    dist = ds.Beta(sAlpha.val, sBeta.val)
    line2dObj.set_ydata(dist.pdf(x))
    fig.canvas.draw_idle()

sAlpha.on_changed(cb_update) # Event callback func. のセット
sBeta.on_changed(cb_update)  # Event callback func. のセット


#
# Reset Button
#                  Left  Btm   Wid   Hight 
resetax = plt.axes([0.1, 0.025, 0.1, 0.04])
buttonR = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def cb_reset(event):
    """
        Reset Button Event Callback Func.
    """
    sAlpha.reset()
    sBeta.reset()

buttonR.on_clicked(cb_reset) # Event callback func. のセット


#
# Quit Button
#                 Left  Btm   Wid   Hight 
quitax = plt.axes([0.8, 0.025, 0.1, 0.04])
buttonQ = Button(quitax, 'Quit', color=axcolor, hovercolor='0.975')

def cb_quit(event):
    """
        Quit Button Event Callback Func.
    """
    sys.exit()

buttonQ.on_clicked(cb_quit) # Event callback func. のセット

# Plot
plt.show()
