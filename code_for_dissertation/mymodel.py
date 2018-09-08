"""
Evaluate the true Fourier coefficients of a given function x(1-x),
generate the domain based on that and define the model Q:\Lambda \to D
"""

import sympy
from inversefuns.utilities import get_coef, coef_domain, fourier_exp_vec
import numpy as np

param_len = 5
t=np.array((0.1,0.2,0.4,0.5,0.7))
period0 = 1.0
def true_param():
    x = sympy.symbols('x')
    # This will take some time because we are evaluating oscillatory function integration
    an, bn = get_coef(expr=(1-x)*(x), vari=x, trun=(param_len-1), period = period0)
    return an, bn


def my_model_domain(pow=-1,halfwidth0=0.5):
    an = bn = np.zeros(param_len)
    domain = coef_domain(an, bn, pow=pow, halfwidth0=halfwidth0)
    return domain

def my_model(parameter_samples):
    num_samples = parameter_samples.shape[0]
    if t.shape:
        QoI_samples = np.zeros((num_samples, t.shape[0]))
    else:
        QoI_samples = np.zeros((num_samples, 1))
    an = parameter_samples[:, 0::2]
    bn = parameter_samples[:, 1::2]
    for i in range(0, num_samples):
        QoI_samples[i, :] = fourier_exp_vec(t,an[i,:],bn[i,:])

    return QoI_samples




