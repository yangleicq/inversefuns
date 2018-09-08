"""
This module contains functions for linear ODE
"""

import numpy as np
import scipy.integrate as integrate
from utilities import halfcube

def linear_ode_fourier(t,an,bn,trun=30, period=1,y0=5):
    r"""
    Compute the exact solution of a linear differential equation:
     y'=lambda(t)y
     y(0)=y0
    when the lambda(t)=1/2a0+\sum_{i=1}^trun aicos(2\pi/T it)+\sum_{i=1}^trun bisin(2\pi/T it)

    :param t: vector of points in [0,T] where the function need to be evaluated
    :param an: List/np.ndarray of coefficients of sin in Lambda, from a0 to an
    :param bn: List/np.ndarray of coefficients of cos in Lambda, from b0 to bn, b0 is always set to 0 for easier programming
    :param trun: Number of truncation terms
    :param period: Period
    :param y0: y(0)
    :return: function value of y(t)
    """
    trunAn=trunBn=trun
    t = np.array(t)
    if trunAn>len(an)-1:
        trunAn=len(an)-1

    if trunBn>len(bn)-1:
        trunBn=len(bn)-1

    an0=an[0]*0.5*t
    an_coe=np.asarray(an[1:])
    bn_coe=np.asarray(bn[1:])
    an_ran=np.arange(1,trunAn+1,dtype=float)
    bn_ran=np.arange(1,trunBn+1,dtype=float)
    if t.shape:
        length = t.shape[0]
    else:
        length = 1
    an_val=np.reshape(np.repeat(an_coe/(2*np.pi*an_ran/period),length),
                      (trunAn,length))*np.sin(2*np.pi*np.outer(an_ran,t)/period)
    bn_val=np.reshape(np.repeat(bn_coe/(2*np.pi*bn_ran/period),length),
                      (trunBn,length))*(1-np.cos(2*np.pi*np.outer(bn_ran,t)/period))
    return y0*np.exp(an0+np.sum(an_val,axis=0)+np.sum(bn_val,axis=0))








def origin_fun(x,case=1):
    """
    Some functions defined on [0,1] to be expanded by Fourier basis
    :param x:The point where the function need to be evaluated
    :param case: Considered case
    :return: Corresponding function value
    """
    if case==1:
        return 32*x*x*(1-x)*(1-x)
    if case==2:
        return 20*x*(1-x)*(0.5-x)
    if case==3:
        return (1-x)*(x)
    if case==4:
        return x**3



def linear_ode_origin(t,y0=5,case=1):
    """
    Compute the numerical solution of a linear differential equation of one of the original function
    :param t: The point where the function need to be evaluated
    :param y0: y(0)
    :param case: Considered case
    :return:Function value
    """
    return y0*np.exp(integrate.quad(origin_fun, 0, t,args=(case))[0])



def coef_random(an,bn,random_trun_start=0,random_start=1,random_end= 32, halfwidth0=1,pow=-1):
    """
    Takes Fourier coefficients and return a random sample of
    uniform distribution on a hypercube around the input coefficients.
    :param an, bn: Base Fourier coefficient
    :param random_trun_start: Until which term the random perturbation will be added
    :param random_start: decay starting parameter
    :param random_end: decay endding parameter
    :param halfwidth: Half width of interval
    :param pow: Decay power
    :return: random coefficients an, bn
    """

    an=np.asarray(an)
    bn=np.asarray(bn)
    half=halfcube(random_start,random_end,halfwidth0,pow)
    an_random=half*np.random.uniform(-1,1,(random_end-random_start,))
    bn_random=half*np.random.uniform(-1,1,(random_end-random_start,))

    an_random=np.append(np.zeros(random_trun_start-0),an_random)
    bn_random=np.append(np.zeros(random_trun_start-0),bn_random)

    if an.shape[0]>an_random.shape[0]:
        an_random.resize(an.shape)
        bn_random.resize(bn.shape)
    else:
        an.resize(an_random.shape)
        bn.resize(bn_random.shape)
    an_random=an+an_random
    bn_random=bn+bn_random

    return an_random,bn_random





def int_random_fourier(an,bn,random_trun_start=0,random_start=1,random_end= 32,
                  halfwidth0=1,pow=-1,lower=0,upper=1):
    """
    Takes Fourier coefficients and return a random integration value
    :param an, bn: Base Fourier coefficients
    :param random_trun_start: until which term the perturbation will be added
    :param random_start
    :param halfwidth: Half width of interval
    :param pow: Decay power
    :param lower: Lower limit of integration
    :param upper: Upper limit of integration
    :return: Integration value
    """
    an_random,bn_random=coef_random(an,bn,random_trun_start=random_trun_start,random_start=random_start,
                                    random_end= random_end, halfwidth0=halfwidth0,pow=pow)
    int_random=integrate.quad(linear_ode_fourier, lower, upper, args=(an_random,bn_random,len(an_random)-1))[0]
    return int_random

