import sympy
import numpy as np


def get_coef(expr,vari, trun=30, period=1.0):
    """
    Calculate Fourier coefficients of symbolic expression of some function
    :param expr: Symbolic expression of a function on which Fourier expansion is performed
    :param vari: variable of the symbolic expression
    :param trun: Truncation terms
    :param period: Period
    :return: Fourier coefficients from 0 to trun
    """
    an=([2/period*sympy.Integral(expr*sympy.cos(2*i*sympy.pi*vari/period),(vari,0,period)).evalf() for i in range(trun+1)])
    bn=([2/period*sympy.Integral(expr*sympy.sin(2*i*sympy.pi*vari/period),(vari,0,period)).evalf() for i in range(trun+1)])
    return np.asarray(an, dtype=float), np.asarray(bn, dtype=float)


def fourier_exp(t,an,bn,trun=10, period=1.0):
    '''
    :param t: Vector where the function need to be evaluated
    :param an: List/np.ndarray of coefficients of sin in Lambda, from a0 to an
    :param bn: List/np.ndarray of coefficients of cos in Lambda, from b0 to bn
    :param trun: Number of truncation terms
    :param period: Period
    :return:  function value
    '''
    trunAn=trunBn=trun
    if trunAn>len(an)-1:
        trunAn=len(an)-1
    if trunBn>len(bn)-1:
        trunBn=len(bn)-1
    an0=an[0]*0.5
    an_coe=np.asarray(an[1:])
    bn_coe=np.asarray(bn[1:])
    an_ran=np.arange(1,trunAn+1,dtype=float)
    bn_ran=np.arange(1,trunBn+1,dtype=float)
    an_val=an_coe*np.cos(2*np.pi*an_ran*t/period)
    bn_val=bn_coe*np.sin(2*np.pi*bn_ran*t/period)
    return an0+np.sum(an_val,axis=0)+np.sum(bn_val,axis=0)

def halfcube(random_start=0,random_end=32,halfwidth0=1,pow=-1):
    """
    produce a halfcube with given dimension and shrinkage
    :param random_start: decay starting parameter
    :param random_end: decay ending parameter
    :param halfwidth0: base halfwidth
    :param pow: decaying power
    :return: A (random_end-random_start,) array
    """
    ran=np.arange(random_start,random_end,dtype=float)
    ran[0]=1.0
    return ran**pow*halfwidth0

def coef_domain(an_mid, bn_mid, random_start=0, halfwidth0=1, pow=-1):
    r"""
    return the parameter domain centered at the an and bn points
    """
    # override random_end with size of an_mid
    random_len =  an_mid.shape[0]
    random_end = random_start+random_len
    # get half lengths of intervals in each dimension
    half = halfcube(random_start, random_end, halfwidth0, pow)

    # define lower and upper bounds for the an and bn coefficients
    an_lower = half*(-1)*np.ones((random_len,))+an_mid[:]
    bn_lower = half*(-1)*np.ones((random_len,))+bn_mid[:]
    an_upper = half*(1)*np.ones((random_len,))+an_mid[:]
    bn_upper = half*(1)*np.ones((random_len,))+bn_mid[:]

    domain = np.zeros((2*(random_len),2))

    # Define lower limits of domain
    domain[0::2,0] = an_lower[:]
    domain[1::2,0] = bn_lower[:]
    # Define upper limits of domain
    domain[0::2,1] = an_upper[:]
    domain[1::2,1] = bn_upper[:]

    return domain


def coef_domain2(an1_mid, bn1_mid, an2_mid, bn2_mid, random_start=1, halfwidth0=1, pow=-1):
    r"""
    return the parameter domain centered at the an1, bn1, an2, bn2
    """
    domain1 = coef_domain(an1_mid, bn1_mid, random_start=random_start, halfwidth0=halfwidth0, pow=pow)
    domain2 = coef_domain(an2_mid, bn2_mid, random_start=random_start, halfwidth0=halfwidth0, pow=pow)
    random_len =  an1_mid.shape[0]
    domain = np.zeros((4*(random_len),2))
    domain[0::4,:] = domain1[0::2,:]
    domain[1::4,:] = domain1[1::2,:]
    domain[2::4,:] = domain2[0::2,:]
    domain[3::4,:] = domain2[1::2,:]
    return domain


