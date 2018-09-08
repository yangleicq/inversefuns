# import bet.calculateP as calculateP
# import bet.postProcess as postProcess
import bet.calculateP.simpleFunP as simpleFunP
import bet.calculateP.calculateP as calculateP
import bet.postProcess.plotP as plotP
import bet.postProcess.plotDomains as plotD
import bet.sample as samp
import bet.sampling.basicSampling as bsam
import numpy.random as nprand
import numpy as np
import pickle
import myModel_linearode as mm
import sys
import os
decaynum = int(sys.argv[1])
figurepath = "figures_linearode/decay-"+str(decaynum)
if not os.path.isdir(figurepath):
    os.makedirs(figurepath)


createNewOutputData = sys.argv[2].lower() == 'true'
an, bn = mm.true_param()
coef_true= np.zeros((1,10))
coef_true[0,0::2] = an
coef_true[0,1::2] = bn
# Initialize 2*trunc_term input parameter sample set object
# trunc_term is defined in myModel
input_samples = samp.sample_set(2*mm.param_len)
# Set parameter domain
parameter_domain = mm.my_model_domain(pow = -decaynum,halfwidth0=0.5)
input_samples.set_domain(parameter_domain)

# Define the sampler that will be used to create the discretization
# object, which is the fundamental object used by BET to compute
# solutions to the stochastic inverse problem
sampler = bsam.sampler(mm.my_model)


# Generate samples on the parameter space
randomSampling = True
if randomSampling is True:
    input_samples = sampler.random_sample_set('random', input_samples, num_samples=200000)
else:
    num_samples_per_dim = 5
    input_samples = sampler.regular_sample_set(input_samples, num_samples_per_dim=num_samples_per_dim)


'''
A standard Monte Carlo (MC) assumption is that every Voronoi cell
has the same volume.
'''
MC_assumption = True
# Estimate volumes of Voronoi cells associated with the parameter samples
if MC_assumption is False:
    input_samples.estimate_volume(n_mc_points=1E5)
#TODO add non-MC method for volume estimation
else:
    input_samples.estimate_volume_mc()



# Create the discretization object using the input samples
my_discretization = sampler.compute_QoI_and_create_discretization(input_samples)


# read discretization

#my_discretization = samp.load_discretization(file_name='linearode_IVP_discretization.txt.gz')
#input_samples = my_discretization._input_sample_set



# Create a reference parameter simulating the scenario where a true parameter
# is responsible for an observed QoI datum. We model the uncertainty in the
# recorded reference QoI datum below.


ref_samples = samp.sample_set(2*mm.param_len)
ref_samples.set_domain(parameter_domain)

Random_ref = False
# Using Random_ref or the true parameter
if Random_ref is True:
    nprand.seed(1)
    ref_discretization = sampler.create_random_discretization('random', ref_samples,
                                                          num_samples=1)
else:
    ref_samples.set_values(coef_true)
    ref_discretization = sampler.compute_QoI_and_create_discretization(ref_samples)






##the create_random_discretization function has 2 steps
##first: generate the input_sample_set using random_sample_set
##second: generate output_sample_set using compute_QoI_and_create_discretization

param_ref = ref_discretization._input_sample_set.get_values()

Q_ref = ref_discretization._output_sample_set.get_values()[0,:]

'''
Suggested changes for user:

Try different ways of discretizing the probability measure on D
defined as a uniform probability measure on an interval centered
at Q_ref whose size is determined by scaling the interval
'''




randomDataDiscretization = False

if createNewOutputData is False:
    output_probability_set = pickle.load( open( "output_probability_set.p", "rb" ) )
    my_discretization._output_probability_set=output_probability_set
else:
    if randomDataDiscretization is False:
        simpleFunP.regular_partition_uniform_distribution_rectangle_scaled(
        data_set=my_discretization, Q_ref=Q_ref, rect_scale=0.08,
        cells_per_dimension = 3)
    else:
        simpleFunP.uniform_partition_uniform_distribution_rectangle_scaled(
        data_set=my_discretization, Q_ref=Q_ref, rect_scale=0.05,
        M=50, num_d_emulate=1E5)
    output_probability_set = my_discretization._output_probability_set
    pickle.dump(output_probability_set, open( "output_probability_set.p", "wb" ) )



# calculate probablities
calculateP.prob(my_discretization)


########################################
# Post-process the results
########################################

#calculate 2d marginal probs
(bins, marginals2D) = plotP.calculate_2D_marginal_probs(input_samples,
                                                        nbins = 10)
plotP.plot_2D_marginal_probs(marginals2D, bins, input_samples, filename = "figures_linearode/decay-"+str(decaynum)+"/linearode_IVP_reg",
                             lam_ref=param_ref[0,:], file_extension = ".eps", plot_surface=False)

# smooth 2d marginals probs (optional)
#marginals2D = plotP.smooth_marginals_2D(marginals2D, bins,
#                                        sigma=0.2*(parameter_domain[:,1]-parameter_domain[:,0]))

# plot 2d marginals probs
#plotP.plot_2D_marginal_probs(marginals2D, bins, input_samples, filename = "linear_ODE_IVP",
#                             lam_ref=param_ref[0,:], file_extension = ".eps", plot_surface=False)

# # calculate 1d marginal probs
# (bins, marginals1D) = plotP.calculate_1D_marginal_probs(input_samples,
#                                                         nbins = 10)
# # smooth 1d marginal probs (optional)
# marginals1D = plotP.smooth_marginals_1D(marginals1D, bins,
#                                         sigma=0.2 * (parameter_domain[:, 1] - parameter_domain[:, 0]))
# # plot 1d marginal probs
# plotP.plot_1D_marginal_probs(marginals1D, bins, input_samples, filename = "figures_linearode/linearode_IVP_reg",
#                              lam_ref=param_ref[0,:], file_extension = ".eps")

## post process
#np.sum(my_discretization._input_sample_set._probabilities_local)

arghigh=np.argsort(-my_discretization._input_sample_set._probabilities_local)

#uniprob = np.unique(np.round(-my_discretization._input_sample_set._probabilities_local,decimals=10))





import inversefuns.linearode as lr

t = np.linspace(0,1,1001)

y=lr.linear_ode_fourier(t,coef_true[0,0::2],coef_true[0,1::2])

sol_all = np.zeros((1001,10))
for i in range(0,10):
    sol=lr.linear_ode_fourier(t,input_samples._values_local[arghigh[i],0::2],input_samples._values_local[arghigh[i],1::2])
    sol_all[:,i]=sol


from matplotlib import pyplot as plt




grid = [100,200,400,500,700]
plt.figure()
plt.plot(t,y,lw=2)
plt.plot(t[grid], y[grid], 'o')
for i in range(0,10):
    plt.plot(t,sol_all[:,i],lw=2,alpha=0.4)
plt.savefig("figures_linearode/decay-"+str(decaynum)+"/Figure.pdf")

# for i in range(0,20):
#     print(arghigh[i])
#     print(input_samples._values_local[arghigh[i],0::2])
#     print(input_samples._values_local[arghigh[i],1::2])
#reload(lr)



orin0=lr.fourier_exp_vec(t,coef_true[0,0::2],coef_true[0,1::2])
orin1=(1-t)*t
orin_all = np.zeros((1001,10))
for i in range(0,10):
    orin=lr.fourier_exp_vec(t,input_samples._values_local[arghigh[i],0::2],input_samples._values_local[arghigh[i],1::2])
    orin_all[:,i]=orin

plt.figure()
plt.plot(t,orin0,lw=2)
plt.plot(t,orin1)
plt.plot(t[grid], orin0[grid], 'o',color='blue')
plt.ylim([-0.4,0.8])

#plt.plot(t,orin1,lw=2)
for i in range(0,10):
    plt.plot(t,orin_all[:,i],lw=2,alpha=0.3,color='black')
#plt.ylim( (-0.2, 0.3) )
plt.savefig("figures_linearode/decay-"+str(decaynum)+"/fundecay"+str(decaynum)+".pdf")


