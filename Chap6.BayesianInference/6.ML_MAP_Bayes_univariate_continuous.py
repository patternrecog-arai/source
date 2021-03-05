#!/usr/bin/env python3
#coding:utf-8
#
# パターンが1次元連続変数で
# 1変数正規分布(Univariate Normal Distribution) でモデル化できる場合の
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


def draw_pdf_heatmap(axis, distribution, rangeMu, rangeVar, numGrid, title):
    """ Draw a Probability density with a heatmap """
    # グラフタイトル
    axis.set_title(title)
    # 表示範囲
    axis.set_xlim(-rangeMu, rangeMu)
    axis.set_ylim(rangeVar/numGrid,rangeVar)
    # (x,y)座標群
    x, y = np.meshgrid(np.linspace(-rangeMu,rangeMu,numGrid), np.linspace(rangeVar/numGrid,rangeVar,numGrid))
    # 確率密度
    p = distribution.pdf(x, y)
    # 分布表示
    axis.pcolormesh(x, y, p, cmap=plt.cm.hot)


def draw_max_point(axis, distribution, rangeMu, rangeVar, numGrid, color):
    """ Draw a Maximum Point of Probability density """
    # (x,y)座標群
    x, y = np.meshgrid(np.linspace(-rangeMu,rangeMu,numGrid), np.linspace(rangeVar/numGrid,rangeVar,numGrid))
    # 確率密度
    p = distribution.pdf(x, y)
    ## 最大点探索
    pos = np.argmax(p)
    max_x = 2*rangeMu * (pos%numGrid) / numGrid - rangeMu
    max_y = rangeVar * (pos//numGrid) / numGrid
    print('最大点=', max_x, max_y, pos) 
    # 最大点描画
    axis.scatter(max_x, max_y, c=color , marker="o")

    return max_x, max_y
    

def draw_likelihood_heatmap(axis, distribution, x_cat, rangeMu, rangeVar, numGrid, title):
    """ Draw a Probability density of Likelihood with a heatmap """
    # グラフタイトル
    axis.set_title(title)
    # 表示範囲
    axis.set_xlim(-rangeMu, rangeMu)
    axis.set_ylim(rangeVar/numGrid,rangeVar)
    # (x,y)座標群
    x, y = np.meshgrid(np.linspace(-rangeMu,rangeMu,numGrid), np.linspace(rangeVar/numGrid,rangeVar,numGrid))
    # 確率密度
    p = distribution.likelihood(x_cat, x, y)
    # 分布表示
    axis.pcolormesh(x, y, p, cmap=plt.cm.hot)
    


def draw_legend(axis):
    """ 凡例の表示 """
    axis.text(1.1, 3.80, 'Prior', ha='left', va='center', color='white')
    axis.text(1.1, 3.60, 'ML', ha='left', va='center', color='white')
    axis.text(1.1, 3.40, 'MAP', ha='left', va='center', color='white')
    #axis.text(1.1, 3.20, 'Bayes', ha='left', va='center', color='white')
    axis.scatter(1.76, 3.805, c='red',     marker='o')
    axis.scatter(1.76, 3.605, c='green',   marker='o')
    axis.scatter(1.76, 3.405, c='cyan',    marker='o')
    #axis.scatter(1.76, 3.205, c='magenta', marker='o')
    

def draw_point(axis, point2D, color):
    """ Draw Point on graph """
    axis.scatter(point2D[0], point2D[1], c=color, marker='o')



###
### ここからMain
###

# 描画範囲と解像度
rangeMu  = 2
rangeVar = 4
#numGrid  = 200
numGrid  = 128
#numGrid  = 100
#numGrid  = 50
eps      = 1.e-2

# 2D 分布表示用
x_min, x_max = -6.0,  6.0 # 推定分布(Norm)表示用 (X軸)
y_min, y_max = -0.02, 0.7 # 推定分布(Norm)表示用 (Y軸)

# 推定点の色
col_MP    = 'red'
col_ML    = 'green'
col_MAP   = 'cyan'
col_Bayes = 'magenta'

# 乱数の種
np.random.seed(1)

# 図の準備
fig = plt.figure(figsize=(12,8))

# Normal Inverse Gamma distribution parameters
alpha_min, alpha_max, alpha0 =  0.0, 5.0, 1.0 
beta_min,  beta_max,  beta0  =  0.0, 5.0, 1.0 
gamma_min, gamma_max, gamma0 =  0.0, 5.0, 1.0 
delta_min, delta_max, delta0 = -2.0, 2.0,-1.0 


#
# Prior (NormInvGam) Heatmap Drawing Area
#                   Left  Btm   Wid   Hight 
axPrior = plt.axes([0.05, 0.50, 0.27, 0.4])

def drawPrior(alpha, beta, gamma, delta):
    """
        事前確率分布の表示
    """
    priorNormInvGam = ds.NormInvGam(alpha, beta, gamma, delta)
    draw_pdf_heatmap(axPrior, priorNormInvGam, rangeMu, rangeVar, numGrid, 'Prior') # Draw Prior
    maxMP_x, maxMP_y = draw_max_point(axPrior, priorNormInvGam, rangeMu, rangeVar, numGrid, col_MP) # Draw Prior
    return priorNormInvGam, maxMP_x, maxMP_y

priorNormInvGam, maxMP_x, maxMP_y = drawPrior(alpha0, beta0, gamma0, delta0)


#
# Univariate Normal distribution
#                        Left  Btm   Wid   Hight 
axLikelihood = plt.axes([0.37, 0.50, 0.27, 0.4])

def updateSamples(mu, var, num_samples):
    """
        学習データの再サンプリング
    """
    normML  = ds.Norm(mu, var)
    x_samples = normML.sampling(num_samples) # Sampling
    mu_ML, var_ML  = normML.MLinfer(x_samples) # ML
    print('mu_ML =', mu_ML, 'var_ML = ', var_ML)
    return normML, x_samples, mu_ML, var_ML

def drawLikelihood(x_samples):
    """
        尤度分布の表示
    """
    draw_likelihood_heatmap(axLikelihood, normML, x_samples, rangeMu, rangeVar, numGrid, 'Likelihood') # Draw ML
    draw_point(axLikelihood, [mu_ML,var_ML], col_ML)

mu0, var0 = 1.0, 2.0
num_samples = 10
normML, x_cat, mu_ML, var_ML = updateSamples(mu0, var0, num_samples)
drawLikelihood(x_cat)

#
# Posterior (NormInvGam) Heatmap Drawing Area
#                       Left  Btm   Wid   Hight 
axPosterior = plt.axes([0.69, 0.50, 0.27, 0.4])

def drawPosterior(priorNormInvGam, mu, var, x_samples, alpha, beta, gamma, delta):
    """
        事後分布の表示
    """
    # MAP
    normMAP = ds.Norm(mu, var)
    mu_MAP, var_MAP = normMAP.MAPinfer(x_samples, priorNormInvGam)
    print('mu_MAP =', mu_MAP, 'var_MAP = ', var_MAP)

    # Bayes
    posteriorNormInvGam = ds.NormInvGam(alpha, beta, gamma, delta)
    posteriorNormInvGam.calcPosterior(x_samples)

    draw_pdf_heatmap(axPosterior, posteriorNormInvGam, rangeMu, rangeVar, numGrid, 'Posterior') # Draw Posterior

    draw_point(axPosterior, [maxMP_x,maxMP_y], col_MP)
    draw_point(axPosterior, [mu_ML,var_ML], col_ML)
    draw_point(axPosterior, [mu_MAP, var_MAP], col_MAP)

    draw_legend(axPosterior)

    return normMAP, posteriorNormInvGam

normMAP, posteriorNormInvGam = drawPosterior(priorNormInvGam, mu0, var0, x_cat, alpha0, beta0, gamma0, delta0)

#
# 2D distribution and Data Drawing Area
#                Left  Btm   Wid   Hight 
ax2D = plt.axes([0.69, 0.05, 0.27, 0.4])

def draw2Ddistribution(axis):
    """
        推定したクラス依存確率密度の表示
    """
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)
    axis.grid()

    # 入力パターン表示
    posY = np.zeros(num_samples)
    axis.scatter(x_cat, posY, c='red', marker="o") # 入力パターン表示
    x_data = np.arange(x_min, x_max, 0.01) # x軸用データ

    # 事前確率分布最大点が表す正規分布
    normMP = ds.Norm(maxMP_x, maxMP_y)
    axis.plot(x_data, normMP.pdf(x_data), c=col_MP)

    # 最尤推定の分布
    axis.plot(x_data, normML.pdf(x_data), c=col_ML)

    # 事後確率分布最大点が表す正規分布
    axis.plot(x_data, normMAP.pdf(x_data), c=col_MAP)

    # Bayes推定の事後分布
    axis.plot(x_data, posteriorNormInvGam.BayesInfer(x_data), c=col_Bayes)

    axis.legend(('MP', 'ML', 'MAP', 'Bayes'))

draw2Ddistribution(ax2D)

# Slider, Button, RadioButton のfaceの色
axcolor = 'lightgoldenrodyellow'

#
# Slider (Prior Distribution Parameters)
#                   Left  Btm   Wid   Hight 
axAlpha = plt.axes([0.10, 0.40, 0.5, 0.03], facecolor=axcolor)
axBeta  = plt.axes([0.10, 0.35, 0.5, 0.03], facecolor=axcolor)
axGamma = plt.axes([0.10, 0.30, 0.5, 0.03], facecolor=axcolor)
axDelta = plt.axes([0.10, 0.25, 0.5, 0.03], facecolor=axcolor)

sAlpha = Slider(axAlpha, r'$\alpha$', alpha_min, alpha_max, valinit=alpha0)
sBeta  = Slider(axBeta,  r'$\beta$' , beta_min,  beta_max,  valinit=beta0)
sGamma = Slider(axGamma, r'$\gamma$', gamma_min, gamma_max, valinit=gamma0)
sDelta = Slider(axDelta, r'$\delta$', delta_min, delta_max, valinit=delta0)

def cb_NormInvGam(val):
    """
        Slider (Prior Distribution Parameters) Event Callback Func.
    """
    alpha  = sAlpha.val
    beta   = sBeta.val
    gamma  = sGamma.val
    delta  = sDelta.val
    
    global priorNormInvGam
    global maxMP_x
    global maxMP_y
    global normML
    global normMAP
    global posteriorNormInvGam
    global x_cat
    global mu_ML
    global var_ML
    global num_samples
    mu  = sMu.val
    var = sVar.val
    # Redraw Prior (Left graph)
    priorNormInvGam, maxMP_x, maxMP_y = drawPrior(alpha, beta, gamma, delta)
    # Redraw Likelihood (Middle graph)
    drawLikelihood(x_cat)
    # Redraw Posterior (Right graph)
    normMAP, posteriorNormInvGam = drawPosterior(priorNormInvGam, mu, var, x_cat, alpha, beta, gamma, delta)
    # Redraw 2D distribution
    ax2D.cla()
    draw2Ddistribution(ax2D)

sAlpha.on_changed(cb_NormInvGam) # Event callback func. のセット
sBeta.on_changed(cb_NormInvGam)  # Event callback func. のセット
sGamma.on_changed(cb_NormInvGam) # Event callback func. のセット
sDelta.on_changed(cb_NormInvGam) # Event callback func. のセット


#
# Slider (# of sampled data and Training data distribution parameters)
#                   Left  Btm   Wid   Hight 
axN     = plt.axes([0.10, 0.10, 0.5, 0.03], facecolor=axcolor)
axMu    = plt.axes([0.10, 0.20, 0.5, 0.03], facecolor=axcolor)
axVar   = plt.axes([0.10, 0.15, 0.5, 0.03], facecolor=axcolor)

sN     = Slider(axN,     'N'           , 2, 100, valinit=num_samples, valfmt="%i")
sMu    = Slider(axMu,    r'$\mu$'      , -rangeMu, rangeMu, valinit=mu0)
sVar   = Slider(axVar,   r'$\sigma^2$' , eps, rangeVar, valinit=var0)

def cb_changeSampleData(val):
    """
        Callback Func. if
        Change the # of Sampled data
        or 
        Change Parameters of Training data distribution
    """
    global normML
    global normMAP
    global posteriorNormInvGam
    global x_cat
    global mu_ML
    global var_ML
    global num_samples
    alpha  = sAlpha.val
    beta   = sBeta.val
    gamma  = sGamma.val
    delta  = sDelta.val
    mu  = sMu.val
    var = sVar.val
    num_samples   = int(sN.val)
    normML, x_cat, mu_ML, var_ML = updateSamples(mu, var, num_samples)
    # Redraw Likelihood (Middle graph)
    drawLikelihood(x_cat)
    # Redraw Posterior (Right graph)
    normMAP, posteriorNormInvGam = drawPosterior(priorNormInvGam, mu, var, x_cat, alpha, beta, gamma, delta)
    # Redraw 2D distribution
    ax2D.cla()
    draw2Ddistribution(ax2D)

sN.on_changed(cb_changeSampleData)   # Event callback func. のセット
sMu.on_changed(cb_changeSampleData)  # Event callback func. のセット
sVar.on_changed(cb_changeSampleData) # Event callback func. のセット


#
# Quit Button
#                 Left  Btm   Wid   Hight 
quitax = plt.axes([0.1, 0.02, 0.1, 0.04])
button_quit = Button(quitax, 'Quit', color=axcolor, hovercolor='0.975')

def cb_quit(event):
    """
        Quit Button Event Callback Func.
    """
    sys.exit()

button_quit.on_clicked(cb_quit) # Event callback func. のセット


#
# Save Button
#
saveax = plt.axes([0.3, 0.02, 0.1, 0.04])
button_save = Button(saveax, 'Save', color=axcolor, hovercolor='0.975')

def cb_save(event):
    """
        Save Button Event Callback Func.
    """
    print('dum')
    fig.savefig('sample.univariate_continuous.py.png', dpi=300, format='png', transparent=True)
    #plt.savefig('sample.univariate_continuous.py.svg', format='svg') # 超遅いし巨大File

button_save.on_clicked(cb_save) # Event callback func. のセット

# Plot
plt.show()

