#!/usr/bin/env python3
#coding:utf-8
#
# Dirichlet分布の表示
#
#	パターンが離散多値変数で
#	カテゴリカル分布(Categorical Distribution)でモデル化できる場合の
#	事前，事後分布であるDirichlet分布の振る舞い
#
# 	p(λ1,..λK | α1,...αK)
#		Dirichlet分布を定めるパラメタは(α1,...αK)のK個
#		分布の変数はλ1,..λKのK個でその定義域は 0 <= λi <= 1
#		Σ λi = 1 を満たす.
#
#	ここでは表示が可能なK=3で実装
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

# A(1,0,0), B(0,1,0), C(0,0,1) で構成される三角形の
# 辺ABを長さ1で規格化し，点Aを(0,0), 点Bを(1,0)とする座標系を構成すると，
# 点Cは(0.5, √0.75)になる．
# [証明] 三次元座標系で，ABの中点をH(0.5,0.5.0)とすると，辺CHの長さは
# 		√(0.5*0.5 + 0.5*0.5 + 1)=√(3/2)
# 新たな二次元座標系ではABの長さを1とするので，倍率は1/√2倍．
# よって三角形に乗る二次元座標系での点Cのy座標は，
# √(3/2) * 1/√2 = √(3/4) = √0.75
#
eps = 1.e-3
corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
corners_mesh = np.array([[eps, eps], [1-eps, eps], [0.5, 0.75**0.5-eps]])

# ドロネー三角分割を実行
triangle = tri.Triangulation(corners_mesh[:, 0], corners_mesh[:, 1])

# 辺の中点を求める
# Mid-points of triangle sides opposite of each corner
midpoints = [(corners[(i + 1) % 3] + corners[(i + 2) % 3]) / 2.0 for i in range(3)]
midpoints = np.array(midpoints)
#print('midpoints', midpoints.shape)

def xy2bc(xy, tol=1.e-3):
    """ Converts 2D Cartesian coordinates to barycentric.
        (2次元座標で与えた点を3次元座標に変換) 重心座標系??
        この3次元座標がDirichlet分布のλなので，λが0や1になると分布が定義できない.
        よって，scipyのdirichlet.pdf()を呼ぶと落ちる
        [対策] mesh点をcornersで示した三角形よりも内側に定義(corners_mesh)
        これでclipは不要．
    """
    s = [(corners[i] - midpoints[i]).dot(xy - midpoints[i].reshape(2,1)) / 0.75 for i in range(3)]
    #return np.clip(s, tol, 1.0 - tol)
    return s

def bc2xy(bc):
    " 3次元座標を2次元座標に変換 "
    x = np.dot(bc, corners[:,0])
    y = np.dot(bc, corners[:,1])
    xy = np.array([x,y])
    return xy


#def draw_pdf_contours(axis, dist, nlevels=200, subdiv=8, **kwargs):
def draw_pdf_contours(axis, dist, nlevels=32, subdiv=5, **kwargs):
    """ 重心座標系での三角メッシュ上にpdfをheatmap表示 """
    import math

    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)
    #pvals = [dist.pdf(xy2bc(xy)) for xy in zip(trimesh.x, trimesh.y)]
    xy = np.array([trimesh.x, trimesh.y])
    pvals = dist.pdf(xy2bc(xy))

    # 分布の最大箇所を探索
    max_idx = np.argmax(pvals)
    xy_max = np.array(xy[:,max_idx])
    #print('xy_max.shape = ', xy_max.shape)
    lambda_max = xy2bc(xy_max.reshape(2,1))
    lambda_max = np.array(lambda_max).flatten()
    #print('max_idx=', max_idx, 'xy_max = ', xy_max, 'lambda=', lambda_max )
   

    axis.tricontourf(trimesh, pvals, nlevels, **kwargs, cmap=plt.cm.hot)
    axis.scatter(xy_max[0], xy_max[1], c='red', marker='o')
    axis.axis('equal')
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 0.75**0.5)
    axis.axis('off')
    #print('>>> # of figures = ', plt.get_fignums())


# Dirichlet distribution parameters
alpha0 = [2.0, 2.0, 2.0]
alpha_min = [eps, eps, eps]
alpha_max = [10, 10, 10]
    
fig, ax = plt.subplots(figsize=(6,6))
plt.subplots_adjust(left=0.1, bottom=0.3)

dirichlet = ds.Dirichlet(alpha0)

draw_pdf_contours(ax, dirichlet)



# Slider, Button, RadioButton のfaceの色
axcolor = 'lightgoldenrodyellow'

#
# Slider
#                  Left  Btm   Wid   Hight 
axAlpha0 = plt.axes([0.15, 0.20, 0.7, 0.03], facecolor=axcolor)
axAlpha1 = plt.axes([0.15, 0.15, 0.7, 0.03], facecolor=axcolor)
axAlpha2 = plt.axes([0.15, 0.10, 0.7, 0.03], facecolor=axcolor)

sAlpha0 = Slider(axAlpha0, r'$\alpha$[0]', alpha_min[0], alpha_max[0], valinit=alpha0[0])
sAlpha1 = Slider(axAlpha1, r'$\alpha$[1]', alpha_min[1], alpha_max[1], valinit=alpha0[1])
sAlpha2 = Slider(axAlpha2, r'$\alpha$[2]', alpha_min[2], alpha_max[2], valinit=alpha0[2])

def cb_update(val):
    """
        Slider Event Callback Func.
    """
    alpha_update = [sAlpha0.val, sAlpha1.val, sAlpha2.val]
    dirichlet.set_param(alpha_update)
    draw_pdf_contours(ax, dirichlet)
    fig.canvas.draw_idle()

sAlpha0.on_changed(cb_update) # Event callback func. のセット
sAlpha1.on_changed(cb_update) # Event callback func. のセット
sAlpha2.on_changed(cb_update) # Event callback func. のセット


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
    sAlpha0.reset()
    sAlpha1.reset()
    sAlpha2.reset()

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
