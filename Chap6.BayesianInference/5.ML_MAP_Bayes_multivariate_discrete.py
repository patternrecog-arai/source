#!/usr/bin/env python3
#coding:utf-8
#
# パターンが1次元離散変数で
# カテゴリカル分布(Categorical Distribution) でモデル化できる場合の
# 最尤推定，MAP推定, Bayes推定
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
print('midpoints', midpoints.shape)

def xy2bc(xy, tol=1.e-3):
    """ Converts 2D Cartesian coordinates to barycentric.
        (2次元座標で与えた点を3次元座標に変換) 重心座標系
        この3次元座標がDirichlet分布のλなので，λが0や1になると分布が定義できない.
        よって，scipyのdirichlet.pdf()を呼ぶと落ちる
        [対策] mesh点をcornersで示した三角形よりも内側に定義(corners_mesh)
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


def draw_pdf_contours(axis, dist, disp_max=False, nlevels=32, subdiv=5, **kwargs):
    """ 重心座標系での三角メッシュ上にpdfをheatmap表示 """
    import math

    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)
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

    if disp_max == True:
        axis.scatter(xy_max[0], xy_max[1], c='red', marker='o')

    axis.axis('equal')
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 0.75**0.5)
    axis.axis('off')

    # 凡例
    if disp_max == True:
        axis.text(0.7, 0.80, 'Prior', ha='left', va='center')
        axis.text(0.7, 0.75, 'ML', ha='left', va='center')
        axis.scatter(0.86, 0.805, c='red',     marker='o')
        axis.scatter(0.86, 0.755, c='green',   marker='o')

    axis.text(0.7, 0.70, 'MAP', ha='left', va='center')
    axis.text(0.7, 0.65, 'Bayes', ha='left', va='center')
    axis.scatter(0.86, 0.705, c='cyan',    marker='o')
    axis.scatter(0.86, 0.655, c='magenta', marker='o')


def draw_likelihood_contours(axis, x_cat, dist, disp_max=False, nlevels=32, subdiv=5, **kwargs):
    """ 重心座標系での三角メッシュ上にpdfをheatmap表示 """
    import math

    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)
    xy = np.array([trimesh.x, trimesh.y])
    pvals = dist.likelihood(x_cat, xy2bc(xy))

    # 分布の最大箇所を探索
    max_idx = np.argmax(pvals)
    xy_max = np.array(xy[:,max_idx])
    #print('xy_max.shape = ', xy_max.shape)
    lambda_max = xy2bc(xy_max.reshape(2,1))
    lambda_max = np.array(lambda_max).flatten()
    #print('max_idx=', max_idx, 'xy_max = ', xy_max, 'lambda=', lambda_max )
   

    axis.tricontourf(trimesh, pvals, nlevels, **kwargs, cmap=plt.cm.hot)

    if disp_max == True:
        axis.scatter(xy_max[0], xy_max[1], c='red', marker='o')

    axis.axis('equal')
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 0.75**0.5)
    axis.axis('off')

    # 凡例
    #axis.text(0.7, 0.80, 'Prior', ha='left', va='center')
    axis.text(0.7, 0.75, 'ML', ha='left', va='center')
    #axis.text(0.7, 0.70, 'MAP', ha='left', va='center')
    #axis.text(0.7, 0.65, 'Bayes', ha='left', va='center')
    #axis.scatter(0.86, 0.805, c='red',     marker='o')
    axis.scatter(0.86, 0.755, c='green',   marker='o')
    #axis.scatter(0.86, 0.705, c='cyan',    marker='o')
    #axis.scatter(0.86, 0.655, c='magenta', marker='o')

    axis.text(0.5, 0.9, 'Likelihood', ha='center', va='center', fontsize=14)


def draw_point(axis, pnt3d, color):
    """ Draw Point on graph """
    xy = bc2xy(pnt3d)
    axis.scatter(xy[0], xy[1], c=color, marker='o')


def drawBarGraph( axis, title, lambda_, draw_y_max, col ):
    """
    Bar Graph の描画
        axis       : グラフ
        title      : タイトル文字列
        lambda_    : lambda_[3]の要素数の確率
        draw_y_max : 表示するbar graph確率の最大値
    """
    xClass = np.arange(3)          # 次元数は３に固定(それ以上のグラフは書けない)
    label = ["x=1", "x=2", "x=3"]  # 横軸のラベル
    axis.set_ylim(0, draw_y_max)   # 縦軸の表示最大確率
    axis.set_title(title)          # 棒グラフのタイトル
    # Bar graph
    axis.bar(xClass,  lambda_,  tick_label=label, align="center", color=col)
    # Bar graph内に数値を書く
    for x, y in zip(xClass, lambda_): axis.text(x, y, '{:.3f}'.format(y), ha='center', va='bottom')



###
### ここからMain
###

# 図の準備
fig = plt.figure(figsize=(17,5))


# Dirichlet distribution parameters
alpha0 = [2.0, 2.0, 2.0]

alpha_min = [eps, eps, eps]
alpha_max = [10, 10, 10]
#                       Left  Btm   Wid   Hight 
axDirichlet = plt.axes([0.01, 0.30, 0.21, 0.6])
dirichlet = ds.Dirichlet(alpha0)
draw_pdf_contours(axDirichlet, dirichlet, True) # Draw Dirichlet
axDirichlet.text(0.5, 0.9, 'Prior', ha='center', va='center', fontsize=14)

col_ML    = 'green'
col_MAP   = 'cyan'
col_Bayes = 'magenta'


# Categorical distribution
num_samples = 10
lambda0 = np.array([1/3, 1/3, 1/3]) # training dataはこの分布からサンプリング

# ML
CatML  = ds.Categorical(lambda0)
x_cat = CatML.sampling(num_samples) # Sampling
CatML.likelihood(x_cat, lambda0)
lambda_ML  = CatML.MLinfer(x_cat)
print('lambda_ML   =', lambda_ML)
#                        Left  Btm   Wid   Hight 
axLikelihood = plt.axes([0.22, 0.30, 0.21, 0.6])
axLikelihood.cla()
draw_likelihood_contours(axLikelihood, x_cat, CatML)
axLikelihood.text(0.5, 0.9, 'Likelihood', ha='center', va='center', fontsize=14)

# MAP
CatMAP = ds.Categorical(lambda0)
lambda_MAP = CatMAP.MAPinfer(x_cat, dirichlet)
print('lambda_MAP  =', lambda_MAP)

# Bayes
posteriorDirichlet = ds.Dirichlet(alpha0)
posteriorDirichlet.calcPosterior(x_cat)
#                                Left  Btm   Wid   Hight 
axPosteriorDirichlet = plt.axes([0.43, 0.30, 0.21, 0.6])
draw_pdf_contours(axPosteriorDirichlet, posteriorDirichlet) # Draw Posterior Dirichlet
axPosteriorDirichlet.text(0.5, 0.9, 'Posterior', ha='center', va='center', fontsize=14)

lambda_Bayes = np.zeros(3)
for k in range(3):
    lambda_Bayes[k] = posteriorDirichlet.BayesInfer(k)
print('lambda_Bayes=', lambda_Bayes)

# Draw Points
draw_point(axDirichlet, lambda_ML, col_ML)
draw_point(axDirichlet, lambda_MAP, col_MAP)
draw_point(axDirichlet, lambda_Bayes, col_Bayes)
draw_point(axLikelihood, lambda_ML, col_ML)
draw_point(axPosteriorDirichlet, lambda_MAP, col_MAP)
draw_point(axPosteriorDirichlet, lambda_Bayes, col_Bayes)


#                  Left   Btm   Wid   Hight 
axML    = plt.axes([0.66, 0.30, 0.08, 0.6])
#                  Left   Btm   Wid   Hight 
axMAP   = plt.axes([0.77, 0.30, 0.08, 0.6])
#                  Left   Btm   Wid   Hight 
axBayes = plt.axes([0.88, 0.30, 0.08, 0.6])

# Draw Bar graph
bar_y_max = 0.6
drawBarGraph( axML,    "ML",    lambda_ML,    bar_y_max, col_ML )
drawBarGraph( axMAP,   "MAP",   lambda_MAP,   bar_y_max, col_MAP )
drawBarGraph( axBayes, "Bayes", lambda_Bayes, bar_y_max, col_Bayes )

# Slider, Button, RadioButton のfaceの色
axcolor = 'lightgoldenrodyellow'

#
# Slider (Distribution Parameters)
#                   Left  Btm   Wid   Hight 
axAlpha0 = plt.axes([0.1, 0.20, 0.6, 0.03], facecolor=axcolor)
axAlpha1 = plt.axes([0.1, 0.15, 0.6, 0.03], facecolor=axcolor)
axAlpha2 = plt.axes([0.1, 0.10, 0.6, 0.03], facecolor=axcolor)

sAlpha0 = Slider(axAlpha0, r'$\alpha$[0]', alpha_min[0], alpha_max[0], valinit=alpha0[0])
sAlpha1 = Slider(axAlpha1, r'$\alpha$[1]', alpha_min[1], alpha_max[1], valinit=alpha0[1])
sAlpha2 = Slider(axAlpha2, r'$\alpha$[2]', alpha_min[2], alpha_max[2], valinit=alpha0[2])

def cb_update(val):
    """
        Slider Event Callback Func. (Distribution Parameters)
    """
    alpha_update = [sAlpha0.val, sAlpha1.val, sAlpha2.val]

    # update Dirichlet's parameters alpha
    dirichlet.set_param(alpha_update)
    draw_pdf_contours(axDirichlet, dirichlet, True) # Draw Dirichlet

    # MAP
    lambda_MAP = CatMAP.MAPinfer(x_cat, dirichlet)
    axMAP.cla()
    drawBarGraph( axMAP,   "MAP",   lambda_MAP,   bar_y_max, col_MAP ) # Draw Bar graph

    # Bayes
    posteriorDirichlet.set_param(alpha_update)
    posteriorDirichlet.calcPosterior(x_cat)
    draw_pdf_contours(axPosteriorDirichlet, posteriorDirichlet) # Draw Posterior Dirichlet
    lambda_Bayes = np.zeros(3)
    for k in range(3):
        lambda_Bayes[k] = posteriorDirichlet.BayesInfer(k)

    axBayes.cla()
    drawBarGraph( axBayes, "Bayes", lambda_Bayes, bar_y_max, col_Bayes ) # Draw Bar graph

    print('Update')
    print('lambda_ML   =', lambda_ML)
    print('lambda_MAP  =', lambda_MAP)
    print('lambda_Bayes=', lambda_Bayes)
    draw_point(axDirichlet, lambda_ML, col_ML)
    draw_point(axDirichlet, lambda_MAP, col_MAP)
    draw_point(axDirichlet, lambda_Bayes, col_Bayes)
    draw_point(axPosteriorDirichlet, lambda_MAP, col_MAP)
    draw_point(axPosteriorDirichlet, lambda_Bayes, col_Bayes)

    fig.canvas.draw_idle()

sAlpha0.on_changed(cb_update) # Event callback func. のセット
sAlpha1.on_changed(cb_update) # Event callback func. のセット
sAlpha2.on_changed(cb_update) # Event callback func. のセット

#
# Slider (Change the # of sampled data)
#                   Left  Btm   Wid   Hight 
axN      = plt.axes([0.1, 0.05, 0.6, 0.03], facecolor=axcolor)

sN       = Slider(axN,      'N'       , 2, 50, valinit=num_samples, valfmt="%i")

def cb_changeN(val):
    """
        Slider Event Callback Func. (Change N)
    """
    num_samples = int(sN.val)
    alpha_update = [sAlpha0.val, sAlpha1.val, sAlpha2.val]

    # ML
    CatML.set_param(lambda0)
    global x_cat
    x_cat = CatML.sampling(num_samples) # Samplingし直し
    global lambda_ML
    lambda_ML  = CatML.MLinfer(x_cat)

    axLikelihood.cla()
    draw_likelihood_contours(axLikelihood, x_cat, CatML)

    axML.cla()
    drawBarGraph( axML,    "ML",    lambda_ML,    bar_y_max, col_ML ) # Draw Bar graph


    # MAP
    CatMAP.set_param(lambda0)
    global lambda_MAP
    lambda_MAP = CatMAP.MAPinfer(x_cat, dirichlet)

    axMAP.cla()
    drawBarGraph( axMAP,   "MAP",   lambda_MAP,   bar_y_max, col_MAP ) # Draw Bar Graph

    # Bayes
    posteriorDirichlet.set_param(alpha_update)
    posteriorDirichlet.calcPosterior(x_cat)
    draw_pdf_contours(axPosteriorDirichlet, posteriorDirichlet) # Draw Posterior Dirichlet
    lambda_Bayes = np.zeros(3)
    for k in range(3):
        lambda_Bayes[k] = posteriorDirichlet.BayesInfer(k)
    axBayes.cla()
    drawBarGraph( axBayes, "Bayes", lambda_Bayes, bar_y_max, col_Bayes ) # Draw Bar Graph

    print('Change N')
    print('lambda_ML   =', lambda_ML)
    print('lambda_MAP  =', lambda_MAP)
    print('lambda_Bayes=', lambda_Bayes)

    axDirichlet.cla()
    draw_pdf_contours(axDirichlet, dirichlet, True) # Draw Dirichlet
    axDirichlet.text(0.5, 0.9, 'Prior', ha='center', va='center', fontsize=14)

    draw_point(axDirichlet, lambda_ML, col_ML)
    draw_point(axDirichlet, lambda_MAP, col_MAP)
    draw_point(axDirichlet, lambda_Bayes, col_Bayes)
    draw_point(axLikelihood, lambda_ML, col_ML)
    draw_point(axPosteriorDirichlet, lambda_MAP, col_MAP)
    draw_point(axPosteriorDirichlet, lambda_Bayes, col_Bayes)

    fig.canvas.draw_idle()

sN.on_changed(cb_changeN) # Event callback func. のセット


#
# Reset Button
#                   Left  Btm   Wid   Hight 
#resetax = plt.axes([0.75, 0.12, 0.07, 0.05])
resetax = plt.axes([0.85, 0.19, 0.14, 0.05])
button_reset = Button(resetax, 'Reset Param.', color=axcolor, hovercolor='0.975')

def cb_reset(event):
    """
        Reset Button Event Callback Func.
    """
    axDirichlet.cla()
    # Reset Sliders
    sAlpha0.reset()  # resetが駄目！一番最初に戻ってしまう
    sAlpha1.reset()
    sAlpha2.reset()
    alpha_update = [sAlpha0.val, sAlpha1.val, sAlpha2.val]
    print('alpha_update=', alpha_update)

    # ML
    lambda_ML  = CatML.MLinfer(x_cat)

    axML.cla()
    drawBarGraph( axML,    "ML",    lambda_ML,    bar_y_max, col_ML ) # Draw Bar graph


    # MAP
    dirichlet.set_param(alpha_update)
    lambda_MAP = CatMAP.MAPinfer(x_cat, dirichlet)

    axMAP.cla()
    drawBarGraph( axMAP,   "MAP",   lambda_MAP,   bar_y_max, col_MAP ) # Draw Bar Graph

    # Bayes
    posteriorDirichlet.set_param(alpha_update)
    posteriorDirichlet.calcPosterior(x_cat)
    lambda_Bayes = np.zeros(3)
    for k in range(3):
        lambda_Bayes[k] = posteriorDirichlet.BayesInfer(k)

    axBayes.cla()
    drawBarGraph( axBayes, "Bayes", lambda_Bayes, bar_y_max, col_Bayes ) # Draw Bar Graph

    draw_pdf_contours(axDirichlet, dirichlet, True) # Draw Dirichlet

    print('Reset')
    print('lambda_ML   =', lambda_ML)
    print('lambda_MAP  =', lambda_MAP)
    print('lambda_Bayes=', lambda_Bayes)
    draw_point(axDirichlet, lambda_ML, col_ML)
    draw_point(axDirichlet, lambda_MAP, col_MAP)
    draw_point(axDirichlet, lambda_Bayes, col_Bayes)
    draw_point(axLikelihood, lambda_ML, col_ML)
    draw_point(axPosteriorDirichlet, lambda_MAP, col_MAP)
    draw_point(axPosteriorDirichlet, lambda_Bayes, col_Bayes)

    fig.canvas.draw_idle()

button_reset.on_clicked(cb_reset) # Event callback func. のセット


#
# Plus/Munus Rate RadioButton
#
#                   Left  Btm   Wid   Hight 
radioax = plt.axes([0.75, 0.05, 0.05, 0.12])
radio = RadioButtons(radioax, ('0.2 Step', '1.0 Step'), active=0)

pm_rate = 0.2
def cb_radio(label):
    """
        Radio Button Event Callback Func.
    """
    global pm_rate
    rate_dict = {'0.2 Step': 0.2, '1.0 Step': 1.0}
    pm_rate = rate_dict[label]

radio.on_clicked(cb_radio) # Event callback func. のセット


#
# Plus Button
#                  Left  Btm   Wid   Hight 
plusax = plt.axes([0.75, 0.19, 0.04, 0.05])
button_plus = Button(plusax, '+', color=axcolor, hovercolor='0.975')

def cb_plus(event):
    """
        Plus Button Event Callback Func.
    """
    delta_alpha = pm_rate
    # Increase Alpha 
    sAlpha0.set_val( np.clip(sAlpha0.val + delta_alpha, alpha_min[0], alpha_max[0]) )
    sAlpha1.set_val( np.clip(sAlpha1.val + delta_alpha, alpha_min[1], alpha_max[1]) )
    sAlpha2.set_val( np.clip(sAlpha2.val + delta_alpha, alpha_min[2], alpha_max[2]) )
    print("+++")

button_plus.on_clicked(cb_plus) # Event callback func. のセット

#
# Minus Button
#                   Left  Btm   Wid   Hight 
minusax = plt.axes([0.80, 0.19, 0.04, 0.05])
button_minus = Button(minusax, '-', color=axcolor, hovercolor='0.975')

def cb_minus(event):
    """
        Plus Button Event Callback Func.
    """
    delta_alpha = pm_rate
    # Decrease Alpha 
    sAlpha0.set_val( np.clip(sAlpha0.val - delta_alpha, alpha_min[0], alpha_max[0]) )
    sAlpha1.set_val( np.clip(sAlpha1.val - delta_alpha, alpha_min[1], alpha_max[1]) )
    sAlpha2.set_val( np.clip(sAlpha2.val - delta_alpha, alpha_min[2], alpha_max[2]) )
    print("---")

button_minus.on_clicked(cb_minus) # Event callback func. のセット


#
# Quit Button
#                  Left  Btm   Wid   Hight 
quitax = plt.axes([0.85, 0.05, 0.1, 0.05])
button_quit = Button(quitax, 'Quit', color=axcolor, hovercolor='0.975')

def cb_quit(event):
    """
        Quit Button Event Callback Func.
    """
    sys.exit()

button_quit.on_clicked(cb_quit) # Event callback func. のセット

#
# Save Button
#                  Left  Btm   Wid   Hight 
saveax = plt.axes([0.85, 0.12, 0.07, 0.05])
button_save = Button(saveax, 'Save Fig.', color=axcolor, hovercolor='0.975')

def cb_save(event):
    """
        Save Button Event Callback Func.
    """
    fig.savefig('sample.univariate_discrete.py.png', dpi=300, format='png', transparent=True)

button_save.on_clicked(cb_save) # Event callback func. のセット

# Plot
plt.show()
