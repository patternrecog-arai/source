#!/usr/bin/env python3
#coding:utf-8
#
# パターンが1次元連続変数で
# 一変量正規分布(Univariate Normal Distribution) でモデル化できる場合の
# 最尤推定，MAP推定, Bayes推定
#
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
#import distribution as ds

################################################################################
#
# Univariate Discrete Distributions
#     class Bernoulli:
#     class Beta:
#
################################################################################
class Bernoulli:
    """
    ベルヌーイ分布 (Bernoulli distribution)
    """
    def __init__(self, lambda_):
        self.__lambda = lambda_
        self.__N      = 0

    def get_param(self):
        """ ベルヌーイパラメタの取得 """
        return self.__lambda

    def get_N(self):
        """ 学習データ数の取得 """
        return self.__N

    def pf(self, x):
        """ 与えられた分布の(x=0 or x=1)における確率値を求める"""
        return (1-self.__lambda) if x==0 else self.__lambda

    def sampling1(self):
        """ 与えられた分布に従って1点サンプリングする"""
        return 1 if np.random() < self.__lambda else 0

    def sampling(self, N):
        """ 与えられた分布に従ってN点サンプリングする"""
        data = [self.sampling1() for i in N]
        data = np.array(data) # list --> array
        return data

    def likelihood(self, x, lambda_):
        """ パターンxが与えられた時，分布パラメタlambda_の分布に対する尤度 """
        num_1 = x.sum()
        num_0 = len(x) - num_1
        print('#of0 = ', num_0, '  #of1 = ', num_1)
        #prob = (1-lambda_)**num_0 * lambda_**num_1
        log_prob = num_0 * np.log(1-lambda_) + num_1 * np.log(lambda_)
        prob = np.exp(log_plob)
        return prob

    def this_likelihood(self, x):
        """ パターンxが与えられた時の尤度を求める """
        prob = functools.reduce(mul, [self.pf(data) for data in x])
        return prob

    def this_log_likelihood(self, x):
        """ パターンxが与えられた時の対数尤度を求める """
        prob = functools.reduce(add, [np.log(self.pf(data)) for data in x])
        return prob
        
    def MLinfer(self, x):
        """ 最尤推定で学習データから尤度分布を推定する """
        self.__N      = len(x)
        num1          = x.sum()
        self.__lambda = num1 / self.__N
        return  self.__lambda
        
    def MAPinfer(self, x, priorBeta):
        """ MAP推定で事前分布(priorBeta)と学習データ(x)から尤度分布を推定する """
        alpha, beta = priorBeta.get_param()
        self.__N      = len(x)
        num1          = x.sum()
        self.__lambda = (num1 + beta - 1.0 ) / ( self.__N + alpha + beta - 2.0 )
        return self.__lambda
        

class Beta:
    """
    ベータ分布 (beta distribution)
    """
    def __init__(self, alpha, beta):
        self.__alpha = alpha
        self.__beta  = beta
        self.__N     = 0

    def get_param(self):
        """ ベータ分布パラメタの取得 """
        return self.__alpha, self.__beta

    def get_N(self):
        """ 学習データ数の取得 """
        return self.__N

    def pdf(self, lambda_):
        """ (lambda)における確率密度値を求める"""
        p = st.beta.pdf(lambda_, self.__alpha, self.__beta)
        return p

    def prob(self, bernoulli):
        """ 引数でしていしたベルヌーイ分布(bernoulli)が観測される確率密度値"""
        lambda_ = bernoulli.get_param()
        return self.pdf(lambda_)

    def sampling(self, N):
        """ N点サンプリングする """
        lambdaN  = st.beta.rvs(self.__alpha, self.__beta, size=N)
        return lambdaN

    def calcPosterior(self, x):
        """ 学習データ(x)と事前分布(self)から事後分布を推定する 
            ベータ分布とベルヌーイ分布の積はベータ分布
            推定した事後分布(ベータ分布パラメタ)をメンバ変数に保存
        """
        alpha, beta = self.get_param()
        N      = len(x)
        N_1  = x.sum()
        N_0  = N - N_1
        self.__alpha  = alpha + N_1
        self.__beta   = beta  + N_0
        self.__N = N 
        #self.__N += N # 今までの全学習データ数

        # Original definition (Overflowするので以下に変形)
        #        Γ(pα+pβ)         Γ(mα)Γ(mβ)
        # κ= ----------------- ・ -----------------
        #      Γ(pα)Γ(pβ)         Γ(mα+mβ)
        # mKappa = ( tgamma(pAlpha+pBeta) / (tgamma(pAlpha) * tgamma(pBeta)) )
        #        / ( tgamma(mAlpha+mBeta) / (tgamma(mAlpha) * tgamma(mBeta)) ); 
        #
        # log κ = logΓ(pα+pβ) - logΓ(mα+mβ) + logΓ(mα) + logΓ(mβ) - logΓ(pα) - logΓ(pβ) 
        lKappa = sp.gammaln(alpha+beta) - sp.gammaln(self.__alpha + self.__beta) \
               + sp.gammaln(self.__alpha) + sp.gammaln(self.__beta) \
               - sp.gammaln(alpha) - sp.gammaln(beta) 
        kappa = np.exp( lKappa )

        return kappa

    def BayesInfer(self, x):
        """ 1つの未知データ(x)から事後確率p(x*|D)を推定する
        """
        # Original definition (Overflowするので以下に変形)
        #        Γ(pα+pβ)         Γ(mα)Γ(mβ)
        # κ= ----------------- ・ -----------------
        #      Γ(pα)Γ(pβ)         Γ(mα+mβ)
        # kappa = ( tgamma(mAlpha+mBeta) / (tgamma(mAlpha) * tgamma(mBeta)) )
        #       / ( tgamma(postAlpha+postBeta) / (tgamma(postAlpha) * tgamma(postBeta)) ); 
        #
        # logκ = logΓ(pα+pβ) - logΓ(mα+mβ) + logΓ(mα) + logΓ(mβ) - logΓ(pα) - logΓ(pβ) 

        alpha, beta = self.get_param()
        alphaN = alpha
        betaN  = beta
        if x == 1:
            alphaN += 1
        else:
            betaN  += 1

        lKappa = sp.gammaln(alpha+beta) - sp.gammaln(alphaN+betaN) \
               + sp.gammaln(alphaN) + sp.gammaln(betaN) \
               - sp.gammaln(alpha)  - sp.gammaln(beta)
        kappa = np.exp( lKappa )

        return kappa


################################################################################
#
# Univariate Continuous Distributions
#     class Norm:
#     class NormInvGam:
#
################################################################################
class Norm:
    """
    1変量正規分布 (univariate normal distribution)
    """
    def __init__(self, mu=0.0, var=1.0):
        self.__mu    = mu
        self.__var   = var
        self.__sigma = np.sqrt(self.__var)
        self.__N     = 0

    def get_param(self):
        """ 正規分布パラメタの取得 """
        return self.__mu, self.__var

    def get_N(self):
        """ 学習データ数の取得 """
        return self.__N

    def pdf(self, x):
        """ 与えられた分布の(x)における確率密度値を求める"""
        return st.norm.pdf(x, self.__mu, np.sqrt(self.__var))

    def sampling(self, N):
        """ 与えられた分布に従ってN点サンプリングする"""
        return st.norm.rvs(size=N)*self.__sigma + self.__mu

    def likelihood(self, x, m, v):
        """ パターンxが与えられた時の(m,v)での尤度を求める[式(6.47)] """
        N = len(x)
        m1 = m.flatten()
        v1 = v.flatten()
        p1 = np.zeros(m1.shape)
        idx = 0
        for mu, var in zip(m1,v1):
            sum_x2 = ((x-mu)**2).sum() / var
            p = 1 / (2*np.pi*var)**(N/2) * np.exp( -0.5 * sum_x2 )
            p1[idx] = p
            idx += 1
        prob = np.reshape(p1, m.shape)
        return prob

    def log_likelihood(self, x, m, v):
        """ パターンxが与えられた時の(m,v)での対数尤度を求める[式(6.47)] """
        N = len(x)
        m1 = m.flatten()
        v1 = v.flatten()
        p1 = np.zeros(m1.shape)
        idx = 0
        for mu, var in zip(m1,v1):
            sum_x2 = ((x-mu)**2).sum() / var
            p = -0.5 * N * np.log(2*np.pi) - 0.5 * N * np.log(var) - 0.5 * sum_x2 
            #p1[idx] = np.exp(p)
            p1[idx] = p
            idx += 1
        prob = np.reshape(p1, m.shape)
        #print(np.argmax(prob))
        return prob
        
    def this_likelihood(self, x):
        """ 与えられた1次元正規分布に対するパターンxの尤度を求める """
        prob = 1 / np.sqrt(2*np.pi*self.__var) * np.exp( -(x-self.__mu)**2 / (2*self.__var) )
        return prob

    def this_log_likelihood_value(self, x):
        """ 与えられた1次元正規分布に対するパターンxの対数尤度を求める """
        x2 = (x-self.__mu)**2 / self.__var
        prob = -0.5 * ( np.log(2*np.pi) + np.log(self.__var) + x2 )
        return prob

    def MLinfer(self, x):
        """ 最尤推定で学習データから尤度分布を推定する """
        self.__N     = len(x)
        self.__mu    = x.sum() / self.__N
        self.__var   = ((x - self.__mu)**2).sum() / self.__N
        self.__sigma = np.sqrt(self.__var)
        return  self.__mu, self.__var
        
    def MAPinfer(self, x, priorNormInvGamma):
        """ MAP推定で事前分布(priorNormInvGamma)と学習データ(x)から尤度分布を推定する [式(6.55)] """
        alpha, beta, gamma, delta = priorNormInvGamma.get_param()
        self.__N  = len(x)
        sum_x = x.sum()
        self.__mu = 1/(self.__N+gamma) * (sum_x + gamma*delta)
        sum_x2 = ((x-self.__mu)**2).sum()
        self.__var = 1/(self.__N+2*alpha+3) * (sum_x2 + 2*beta + gamma*(delta-self.__mu)**2)
        self.__sigma = np.sqrt(self.__var)
        return self.__mu, self.__var
        


class InvGam:
    """
    逆ガンマ分布 (inverse gamma distribution)
    """
    def __init__(self, alpha, beta):
        self.__alpha = alpha
        self.__beta  = beta

    def get_param(self):
        return self.__alpha, self.__beta

    def pdf(self, x):
        """ (x)における確率密度値を求める"""
        return st.invgamma.pdf(x, self.__alpha, 0, self.__beta)

    def sampling(self, N):
        """ N点サンプリングする"""
        return st.invgamma.rvs(self.__alpha, size=N)*self.__beta



class NormInvGam:
    """
    正規逆ガンマ分布 (normal-scaled inverse gamma distribution)
    """
    def __init__(self, alpha, beta, gamma, delta):
        self.__alpha = alpha
        self.__beta  = beta
        self.__gamma = gamma
        self.__delta = delta
        self.__N     = 0

    def get_param(self):
        """ 正規逆ガンマ分布パラメタの取得 """
        return self.__alpha, self.__beta, self.__gamma, self.__delta

    def get_N(self):
        """ 学習データ数の取得 """
        return self.__N

    def norm(self, mu, var, gamma, delta):
        """ 正規分布(正規逆ガンマ分布用) """
        normal = 1 / np.sqrt(2*np.pi*var/gamma) * np.exp( -(gamma*(mu-delta)**2)/(2*var))
        return normal

    def invGamma(self, var, alpha, beta):
        """ 逆ガンマ分布(正規逆ガンマ分布用) """
        Gamma = math.gamma(alpha)
        inv_gamma = (beta**alpha) /Gamma * (1/var)**(alpha+1) * np.exp(-beta/var)
        return inv_gamma

    def pdf(self, mu, var):
        """ (mu, var)における確率密度値を求める"""
        p = self.norm(mu, var, self.__gamma, self.__delta) * self.invGamma(var, self.__alpha, self.__beta)
        return p

    def prob(self, norm):
        """ 1変量正規分布(norm)が観測される確率密度値"""
        mu, var = norm.get_param()
        return self.pdf(mu, var)

    def sampling(self, N):
        """ N点サンプリングする
            正規逆ガンマ分布に従うサンプル(正規分布パラメタ:μ,σ2)を生成
            (1) Sample σ^2 from an inverse gamma distribution with parameters α and β.
            (2) Sample μ from a normal distribution with mean δ and variance σ^2/γ
        """
        var = st.invgamma.rvs(self.__alpha, size=N)*self.__beta
        mu  = st.norm.rvs(size=N)*np.sqrt(var/self.__gamma) + self.__delta
        return mu, var

    def calcPosterior(self, x):
        """ 学習データ(x)と事前分布(self)から事後分布を推定する [式(6.57)]
            1変量正規分布と正規逆ガンマ分布の積は正規逆ガンマ分布
            推定した事後分布(正規逆ガンマ分布パラメタ)をメンバ変数に保存
        """
        alpha, beta, gamma, delta = self.get_param()
        N      = len(x)
        sum_x  = x.sum()
        sum_x2 = (x**2).sum()
        self.__alpha  = alpha + N/2
        self.__beta   = sum_x2/2 + beta + gamma*delta**2/2 - (gamma*delta + sum_x)**2 / (2*(gamma+N))
        self.__gamma  = gamma + N
        self.__delta  = (gamma*delta + sum_x) / (gamma+N)

        self.__N = N 
        #self.__N += N # 今までの全学習データ数

        kappa = 1.0 / (2.0*np.pi)**(N/2.0) * np.sqrt(gamma/self.__gamma) * np.exp( alpha*np.log(beta) - self.__alpha*np.log(self.__beta) ) * np.exp( sp.gammaln(self.__alpha) - sp.gammaln(alpha) )
        return kappa

    def BayesInfer(self, x):
        """ 1つの未知データ(x)から事後確率p(x*|D)を推定する
        """
        alpha, beta, gamma, delta = self.get_param()
        x2     = x**2
        alphaN = alpha + 1/2
        betaN  = x2/2 + beta + gamma*delta**2/2 - (gamma*delta + x)**2 / (2*(gamma+1))
        gammaN = gamma + 1
        deltaN = (gamma*delta + x) / (gamma+1)

        kappa = 1.0 / np.sqrt(2.0*np.pi) * np.sqrt(gamma/gammaN) * np.exp( alpha*np.log(beta) \
              - alphaN*np.log(betaN) ) * np.exp( sp.gammaln(alphaN) - sp.gammaln(alpha) )
        return kappa


################################################################################
#
# Multivariate Discrete Distributions
#     class Categorical:
#     class Dirichlet:
#
################################################################################
class Categorical:
    """
    カテゴリカル分布 (Categorical distribution)
    """
    def __init__(self, lambda_):
        self.__lambda = lambda_.copy()  # K次元ベクトル
        self.__K      = len(lambda_)
        self.__N      = 0

    def get_param(self):
        """ Categorical分布パラメタの取得 """
        return self.__lambda

    def set_param(self, lambda_):
        """ Categorical分布パラメタの再設定 """
        self.__lambda = lambda_.copy()
        self.__K     = len(lambda_)     # K次元ベクトル
        self.__N     = 0

    def get_K(self):
        """ カテゴリ数の取得 """
        return self.__K

    def get_N(self):
        """ 学習データ数の取得 """
        return self.__N

    def pf(self, x):
        """ 与えられた分布で観測値xに対する確率値を求める"""
        return self.__lambda[x]

    def sampling1(self):
        """ 与えられた分布に従って1点サンプリングする
            ルーレット選択
        """
        thUpper = np.zeros(self.__K)
        sum_ = 0.0
        for k in range(self.__K):
            thUpper[k] = sum_ + self.__lambda[k]
            sum_ += self.__lambda[k]

        number = np.random.rand() # Generate random [0,1]

        X = 0
        while number > thUpper[X]: X += 1

        return X


    def sampling(self, N):
        """ 与えられた分布に従ってN点サンプリングする"""
        data = np.zeros(N)
        for i in range(N):
            data[i] = self.sampling1() 
        data = np.array(data) # list --> array
        return data

    def likelihood(self, x, lambda_):
        """ 分布パラメタ(lambda_)での分布にパターン群xを与えた時の尤度 """
        numK = np.zeros(self.__K)
        for n in range(len(x)):
            numK[int(x[n])] += 1
        #for k in range(self.__K):
        #    print('[%d] %d' % (k, numK[k]))

        log_prob = 0
        for k in range(self.__K):
            log_prob += numK[k] * np.log(lambda_[k])
        prob = np.exp(log_prob)
        #print('prob = ', prob)

        return prob
        

    def this_likelihood(self, x):
        """ パターンxが与えられた時の尤度を求める """
        prob = functools.reduce(mul, [self.pf(data) for data in x])
        return prob

    def this_log_likelihood(self, x):
        """ パターンxが与えられた時の対数尤度を求める """
        prob = functools.reduce(add, [np.log(self.pf(data)) for data in x])
        return prob
        

    def MLinfer(self, x):
        """ 最尤推定で学習データから尤度分布を推定する [式(6.39)]"""
        self.__N      = len(x)
        Nk = np.zeros(self.__K)
        for n in range(self.__N):
            if x[n] < 0 or self.__K <= x[n]:
                eprint("Out of Range")
                sys.exit()
            Nk[int(x[n])] += 1

        for k in range(self.__K):
            self.__lambda[k] = Nk[k] / self.__N

        return  self.__lambda

        
    def MAPinfer(self, x, priorDirichlet):
        """ MAP推定で事前分布(priorDirichlet)と学習データ(x)から尤度分布を推定する[式(6.41)] """
        alpha = priorDirichlet.get_param()

        sumAlpha = 0.0
        for k in range(self.__K):
            sumAlpha += alpha[k]
        self.__N      = len(x)
        Nk = np.zeros(self.__K)
        for n in range(self.__N):
            if x[n] < 0 or self.__K <= x[n]:
                eprint("Out of Range")
                sys.exit()
            Nk[int(x[n])] += 1

        for k in range(self.__K):
            self.__lambda[k] = (Nk[k] + alpha[k] - 1) / (self.__N + sumAlpha - self.__K)
        
        return self.__lambda
        

class Dirichlet:
    """
    ディリクレ分布 (Dirichlet distribution)
    """
    def __init__(self, alpha):
        self.__alpha = alpha.copy()
        self.__K     = len(alpha)     # Dirichlet分布の次元数
        self.__N     = 0

    def get_param(self):
        """ ディリクレ分布パラメタの取得 """
        return self.__alpha

    def set_param(self, alpha):
        """ ディリクレ分布パラメタの再設定 """
        self.__alpha = alpha.copy()
        self.__K     = len(alpha)     # Dirichlet分布の次元数
        self.__N     = 0

    def get_K(self):
        """ Dirkcklet分布次元数の取得 """
        return self.__K

    def get_N(self):
        """ 学習データ数の取得 """
        return self.__N

    def pdf(self, lambda_):
        """ (lambda)における確率密度値を求める"""
        p = st.dirichlet.pdf(lambda_, self.__alpha)
        return p

    def prob(self, categorical):
        """ 引数でしていしたカテゴリカル分布(categorical)が観測される確率密度値"""
        lambda_ = categorical.get_param()
        return self.pdf(lambda_)

    def sampling(self, N):
        """ N点サンプリングする """
        lambdaN  = st.dirichlet.rvs(self.__alpha, size=N)
        return lambdaN

    def calcPosterior(self, x):
        """ 学習データ(x)と事前分布(self)から事後分布を推定する 
            Dirichlet分布とCategorical分布の積はDirichlet分布
            推定した事後分布(Dirichlet分布パラメタ)をメンバ変数に保存
        """
        alphaPrior = self.get_param()	# 事前分布のパラメタ
        N          = len(x)
        self.__N   = N

        N_k = np.zeros(self.__K)
        for k in x:
            N_k[int(k)] += 1

        for k in range(self.__K):
            self.__alpha[k] = alphaPrior[k] + N_k[k]  # 事後分布のパラメタ

        sumAlphaPrior = 0
        for alpha in alphaPrior:
            sumAlphaPrior += alpha
      
        sumAlphaPosterior = 0
        for alpha in self.__alpha:
            sumAlphaPosterior += alpha

        # Original definition (Overflowするので以下に変形)
        #         Γ[Σαj]        Π Γ[αj + Nj]
        # κ = --------------- ・ -----------------
        #       Γ[N + Σαj]        Π Γ[αj]
        #
        # mKappa  = ( tgamma(sumAlpha) / tgamma(N+sumAlpha) )
        #         * ( prodGammaPosterior / prodGammaPrior );
        #
        # logκ = logΓ[Σαj] - logΓ[N+Σαj] + Σ logΓ[αj + Nj] - Σ logΓ[αj]
        sumGammaPrior = 0
        sumGammaPosterior = 0
        for k in range(self.__K):
            sumGammaPrior     += sp.gammaln(alphaPrior[k])
            sumGammaPosterior += sp.gammaln(self.__alpha[k])

        lKappa = sp.gammaln(sumAlphaPrior) - sp.gammaln(sumAlphaPosterior) + sumGammaPosterior - sumGammaPrior
        kappa  = np.exp( lKappa )

        return kappa


    def BayesInfer(self, x):
        """ 1つの未知データ(x)から事後確率p(x*|D)を推定する
        """
        #                           Nk + αk        α'k
        #  p(X'=k|D) = κ(X',α') = ------------ = -------
        #                          Σ(Nm + αm)     Σ α'm
 
        alphaPrior = self.get_param()	# 事前分布のパラメタ
        sumAlpha = 0
        for alpha in alphaPrior:
            sumAlpha += alpha

        kappa = alphaPrior[x] / sumAlpha

        return kappa


################################################################################
#
# Multivariate Continuous Distributions
#     class MultivariateNorm:
#     class NormInvWishart:
#
################################################################################
Mu0 = np.zeros(2)
Sigma0 = np.identity(2, dtype=float)
class MultivariateNorm:
    """
    多変量正規分布 (multivariate normal distribution)
    """
    def __init__(self, Mu=Mu0, Sigma=Sigma0):
        self.__Mu    = np.copy(Mu)
        self.__Sigma = np.copy(Sigma)
        self.__dim   = len(Mu)
        self.__N     = 0

    def get_param(self):
        """ 正規分布パラメタの取得 """
        return self.__Mu, self.__Sigma

    def get_N(self):
        """ 学習データ数の取得 """
        return self.__N

    def pdf(self, x):
        """ 与えられた分布の(x)における確率密度値を求める"""
        return st.multivariate_normal.pdf(x, mean=self.__Mu, cov=self.__Sigma)

    def sampling(self, N):
        """ 与えられた分布に従ってN点サンプリングする"""
        return st.multivariate_normal.rvs(mean=self.__Mu, cov=self.__Sigma, size=N)

    def this_likelihood(self, x):
        """ パターンxが与えられた時の与えられた分布での尤度を求める """
        L = 1.0
        for n in range(len(x)):
            L *= self.pdf(x[n])
        return L

    def this_log_likelihood(self, x):
        """ パターンxが与えられた時の与えられた分布での対数尤度を求める """
        logL = 0.0
        for n in range(len(x)):
            logL += log(self.pdf(x[n]))
        return logL

    def MLinfer(self, x):
        """ 最尤推定で学習データから尤度分布を推定する """
        self.__Mu = np.mean(x, axis=0)
        self.__Sigma = np.cov(x, rowvar=False, bias=True)
        print('Mu.shape=', self.__Mu.shape, 'Sigma.shape=', self.__Sigma.shape)
        print('-- Mu --\n', self.__Mu)
        print('-- Sigma --\n', self.__Sigma)
        
    def MAPinfer(self, x, priorNormInvWishart):
        """ MAP推定で事前分布(priorNormInvWishart)と学習データ(x)から尤度分布を推定する """
        #
        # 未実装
        #    事後分布の最大点を見つければ良い
        #
        


Delta0 = np.zeros(2)
Psi0 = np.identity(2, dtype=float)
class NormInvWishart:
    """
    正規逆ウィシャート分布 (normal-scaled inverse Wishart distribution)
    alpha : スカラー ( > 次元数 )
    Psi   : 行列
    gamma : スカラー
    Delta : ベクトル
    """
    def __init__(self, alpha=3, Psi=Psi0, gamma=1, Delta=Delta0):
        self.__alpha = alpha
        self.__Psi   = Psi
        self.__gamma = gamma
        self.__Delta = Delta
        self.__dim   = len(Delta)
        self.__N     = 0

    def get_param(self):
        """ 正規逆ウィシャート分布パラメタの取得 """
        return self.__alpha, self.__Psi, self.__gamma, self.__Delta

    def get_N(self):
        """ 学習データ数の取得 """
        return self.__N

    def pdf(self, muX, SigmaX):
        """ (muX, SigmaX)における確率密度値を求める
            NIW(μ,Σ|α,Ψ,γ,δ) = N(μ|δ, Σ/γ)・W^{-1}(Σ|Ψ,α)
        """
        pNorm = st.multivariate_normal.pdf(muX, mean=self.__Delta, cov=SigmaX/self.__gamma)
        pInvWis = st.invwishart.pdf(SigmaX, self.__alpha, self.__Psi)
        return pNorm * pInvWis

    def prob(self, multivariateNorm):
        """ 1変量正規分布(norm)が観測される確率密度値"""
        Mu, Sigma = multivariateNorm.get_param()
        return self.pdf(Mu, Sigma)

#    def sampling(self, N):
#        """ N点サンプリングする
#            正規逆ウィシャート分布に従うサンプル(多変量正規分布パラメタ:μ,Σ)を生成
#            Generation of random variates is straightforward:
#            (1) Sample Sigma from an inverse Wishart distribution with parameters Psi and alpha
#            (2) Sample Mu from a multivariate normal distribution with mean Delta and variance Sigma/gamma
#        """
#        # このようにαでΨを補正しないと，αの増大に比例して分散(広がり)が小さくなってしまう
#        # 式から判断して追加した．
#        fPsi = self.__Psi / self.__alpha  # 補正
#        # (1) Sampling Sigma
#        sampleSigma = st.invwishart.rvs( self.__alpha, fPsi, size=N )
#        # (2) Sampling Mu
#        fSigma = sampleSigma/self.__gamma
#        print('fSigma.shape=', fSigma.shape)
#        sampleMu = st.multivariate_normal.rvs(mean=self.__Delta , cov=fSigma, size=N)
#        return sampleMu, sampleSigma

    def sampling1(self):
        """ 与えられた分布に従って1点サンプリングする
            正規逆ウィシャート分布に従うサンプル(多変量正規分布パラメタ:μ,Σ)を生成
            Generation of random variates is straightforward:
            (1) Sample Sigma from an inverse Wishart distribution with parameters Psi and alpha
            (2) Sample Mu from a multivariate normal distribution with mean Delta and variance Sigma/gamma
        """
        # このようにαでΨを補正しないと，αの増大に比例して分散(広がり)が小さくなってしまう
        # 式から判断して追加した．
        #fPsi = self.__Psi / self.__alpha  # 補正
        fPsi = self.__Psi   # 補正なし定義通り
        # (1) Sampling Sigma
        sampleSigma = st.invwishart.rvs( self.__alpha, fPsi, size=1 )
        # (2) Sampling Mu
        fSigma = sampleSigma/self.__gamma
        sampleMu = st.multivariate_normal.rvs(mean=self.__Delta , cov=fSigma, size=1)
        return sampleMu, sampleSigma

    def sampling(self, N):
        """ 与えられた分布に従ってN点サンプリングする"""
        sampleMu = np.zeros((N,self.__dim))
        sampleSigma = np.zeros((N,self.__dim,self.__dim))
        for i in range(N):
            sampleMu[i], sampleSigma[i] = self.sampling1()
        return sampleMu, sampleSigma

    def logMultivariateGamma( x, d ):
        val = (d*(d-1))/4.0 * np.log(np.pi);
        for j in range(d):
            val += sp.gammaln(x + (1.0-j)/2.0) 
        return val

    def calcPosterior(self, x):
        """ 学習データ(x)と事前分布(self)から事後分布を推定する 
            多変量正規分布と正規逆ウィシャート分布の積は正規逆ウィシャート分布
            推定した事後分布(正規逆ウィシャート分布パラメタ)をメンバ変数に保存
        """
        alpha, Psi, gamma, Delta = self.get_param()
        N      = len(x)
        dim    = self.__dim
        sum_x  = x.sum()

        self.__alpha  = alpha + N
        self.__gamma  = gamma + N
        deltaMat = np.outer(Delta, Delta)
        xMat = np.zeros((dim,dim), dtype=float)
        for xVec in x:
            xMat += np.outer(xVec, xVec)
        termVec = gamma*Delta + sum_x
        termMat = np.outer(termVec, termVec)
        self.__Psi    = Psi + deltaMat + xMat + termMat
        self.__Delta  = (gamma*Delta + sum_x) / (gamma+N)

        self.__N = N 
        #self.__N += N # 今までの全学習データ数

        kappa = 1.0 / (np.pi)**(N*dim/2.0) \
              * np.exp( (np.log(np.linalg.det(Psi)) * alpha/2.0) - (np.log(np.linalg.det(self.__Psi)) * self.__alpha/2.0) ) \
              * np.exp( logMultivariateGamma(self.__alpha/2.0, dim) - logMultivariateGamma(alpha/2.0, dim) ) \
              * (gamma/self.__gamma)**(dim/2.0)
        return kappa

    def BayesInfer(self, x):
        """ 1つの未知データ(x)から事後確率p(x*|D)を推定する
        """
        alpha, Psi, gamma, Delta = self.get_param()
        dim    = self.__dim

        alphaPost = alpha + 1
        gammaPost = gamma + 1
        deltaMat  = np.outer(Delta, Delta)
        xMat      = np.outer(x, x)
        termVec   = gamma*Delta + x
        termMat   = np.outer(termVec, termVec)
        PsiPost   = Psi + deltaMat + xMat + termMat
        DeltaPost = (gamma*Delta + x) / (gamma+1)

        kappa = 1.0 / (np.pi)**(dim/2.0) \
              * np.exp( (np.log(np.linalg.det(Psi)) * alpha/2.0) - (np.log(np.linalg.det(PsiPost)) * alphaPost/2.0) ) \
              * np.exp( logMultivariateGamma(alphaPost/2.0, dim) - logMultivariateGamma(alpha/2.0, dim) ) \
              * (gamma/gammaPost)**(dim/2.0)
        return kappa
