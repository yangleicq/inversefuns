"""
This module contains functions for two dimensional(dual lambda example) and three dimensional(Lorentz example) ODE
"""


import numpy as np
from utilities import fourier_exp
import sympy





def lambda10(t):
    return 1/(2*(1+t**2))

def lambda2(t,period=1.0):
    k=sympy.floor(t/period)
    return (t-k*period)*((k+1)*period-t)

def dual_lambda_fourier_gen(an1,bn1,an2,bn2,period=1.0):
    def dual_lambda_fourier(y,t):
        y1p=fourier_exp(t,an1,bn1,period=period)*y[0]-fourier_exp(t,an2,bn2,period=period)*y[1]
        y2p=fourier_exp(t,an1,bn1,period=period)*y[1]+fourier_exp(t,an2,bn2,period=period)*y[0]
        return y1p,y2p
    return dual_lambda_fourier


def lambda1(t,period=1.0):
    k=sympy.floor(t/period)
    return lambda10(t-k*period)

def dual_lambda_origin(y,t,period=1.0):
    y1p=lambda1(t,period=period)*y[0]-lambda2(t,period=period)*y[1]
    y2p=lambda1(t,period=period)*y[1]+lambda2(t,period=period)*y[0]
    return y1p,y2p






###   3RD EXAMPLE (lorentz example)
def lambda0(t, period=1.0):
    k=sympy.floor(t/period)
    return (t-k*period)*((k+1)*period-t)



def lorenz_fourier_gen(an,bn,period=1.0):
    def lorenz_fourier(y, t, s=10.0, A=5.6, w=np.pi*2.0,r=0,b=28.0, c=2.666667):
        y0 = s*(y[1]-y[0])
        y1 = -y[1]-y[0]*y[2]+y[0]*b  + (A+r)*fourier_exp(t,an,bn,period=period)
        y2 = y[0]*y[1]-c*y[2]
        return y0, y1, y2
    return lorenz_fourier

def lorenz_origin(y, t, s=10.0, A=5.6, w=np.pi*2.0,r=0,b=28.0, c=2.666667):
    y0 = s*(y[1]-y[0])
    y1 = -y[1]-y[0]*y[2]+y[0]*b  + (A+r)*lambda0(t)
    y2 = y[0]*y[1]-c*y[2]
    return y0, y1, y2



