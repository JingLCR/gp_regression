#!/usr/bin/env python
# coding: utf-8

# # Modeling spawning salmon
# 
# The plot below shows the relationship between the number of spawning salmon in a particular stream and the number of fry that are recruited into the population in the spring.
# 
# We would like to model this relationship, which appears to be non-linear (we have biological knowledge that suggests it should be non-linear too).
# 
# ![](images/spawn.jpg)

# In[1]:


import os

os.environ['MKL_THREADING_LAYER'] = 'sequential'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'GNU'

# from os import chdir, getcwd
# from os.path import join
# wd=getcwd()
# chdir(join(wd, r'Documents\GitHub\gp_regression'))
# In[2]:


# get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter('ignore')


# In[3]:


salmon_data = pd.read_table(r'data\salmon.txt', sep='\s+', index_col=0)
salmon_data.plot.scatter(x='spawners', y='recruits', s=50)


# # Parametric approaches
# 
# Simple linear regression:
# 
# $$ y_i = \beta_0 + \beta_1 x_i + \epsilon_i $$
# 
# $$ \epsilon_i \sim N(0, \sigma) $$
# 
# There are three unknowns, each of which need to be given a prior:
# 
# $$\beta_0, \beta_1 \sim \text{Normal}(0, 50)$$
# 
# $$\sigma \sim \text{HalfNormal}(50)$$

# In[4]:


a_normal_distribution = pm.Normal.dist(mu=0, sigma=50)
plt.hist(a_normal_distribution.random(size=1000))


# In[5]:


a_half_normal_distribution = pm.HalfNormal.dist(sigma=50)
plt.hist(a_half_normal_distribution.random(size=1000))


# Extract our predictor and response variables.

# In[6]:


x, y = salmon_data[['spawners', 'recruits']].values.T


# Construct a model in PyMC3.

# In[7]:


with pm.Model() as linear_salmon_model:
    
    β = pm.Normal('β', mu=0, sigma=50, shape=2)
    σ = pm.HalfNormal('σ', sigma=50)
    
    μ = β[0] + β[1] * x
    
    recruits = pm.Normal('recruits', mu=μ, sigma=σ, observed=y)


# Fit the model using Markov chain Monte Carlo (MCMC).

# In[8]:


with linear_salmon_model:
    
    linear_trace = pm.sample(1000, tune=2000, cores=2)


# Examine the posterior distributions of the unknown parameters.

# In[9]:


az.plot_trace(linear_trace);


# Draw posterior samples of the regression line.

# In[10]:


X_pred = np.linspace(0, 500, 100)

ax = salmon_data.plot.scatter(x='spawners', y='recruits', c='k', s=50)
ax.set_ylim(0, None)
for b0,b1 in linear_trace['β'][:20]:
    ax.plot(X_pred, b0 + b1*X_pred, alpha=0.3, color='seagreen');


# ### Quadratic Model
# 
# It appears the linear model is a poor fit to the data. Let's make a polynomial (quadratic) model by adding a parameter for $x^2$.

# In[11]:


with pm.Model() as quad_salmon_model:
    
    β = pm.Normal('β', mu=0, sigma=50, shape=3)
    σ = pm.HalfNormal('σ', sigma=50)
    
    μ = β[0] + β[1] * x + β[2] * x**2
    
    recruits = pm.Normal('recruits', mu=μ, sigma=σ, observed=y)


# In[12]:


with quad_salmon_model:
    
    quad_trace = pm.sample(1000, tune=2000, cores=2, random_seed=1)


# In[13]:


ax = salmon_data.plot.scatter(x='spawners', y='recruits', c='k', s=50)
ax.set_ylim(0, None)
for b0,b1,b2 in quad_trace['β'][:20]:
    ax.plot(X_pred, b0 + b1*X_pred + b2*X_pred**2, alpha=0.3, color='seagreen');


# ![](images/stop.jpg)
# 
# Let's go back to the slides ...
# 
# ---




# ## Gaussian Process

# In[14]:


with pm.Model() as gp_salmon_model:

    # Lengthscale
    ρ = pm.HalfCauchy('ρ', 5)
    η = pm.HalfCauchy('η', 5)
    
    M = pm.gp.mean.Linear(coeffs=(salmon_data.recruits/salmon_data.spawners).mean())
    K = (η**2) * pm.gp.cov.ExpQuad(1, ρ) 
    
    σ = pm.HalfNormal('σ', 50)
    
    recruit_gp = pm.gp.Marginal(mean_func=M, cov_func=K)
    recruit_gp.marginal_likelihood('recruits', X=salmon_data.spawners.values.reshape(-1,1), 
                           y=salmon_data.recruits.values, noise=σ)
    


# In[15]:


ρ


# In[ ]:


with gp_salmon_model:
    gp_trace = pm.sample(1000, tune=2000, cores=2, random_seed=42)


# In[ ]:


az.plot_trace(gp_trace, var_names=['ρ', 'η', 'σ']);


# In[ ]:


with gp_salmon_model:
    salmon_pred = recruit_gp.conditional("salmon_pred", X_pred.reshape(-1, 1))
    gp_salmon_samples = pm.sample_posterior_predictive(gp_trace, vars=[salmon_pred], samples=3, random_seed=42)


# In[ ]:


ax = salmon_data.plot.scatter(x='spawners', y='recruits', c='k', s=50)
ax.set_ylim(0, None)
for x in gp_salmon_samples['salmon_pred']:
    ax.plot(X_pred, x);


# In[ ]:


with gp_salmon_model:
    gp_salmon_samples = pm.sample_posterior_predictive(gp_trace, vars=[salmon_pred], samples=100, random_seed=42)


# In[ ]:


from pymc3.gp.util import plot_gp_dist
fig, ax = plt.subplots(figsize=(8,6))
plot_gp_dist(ax, gp_salmon_samples['salmon_pred'], X_pred)
salmon_data.plot.scatter(x='spawners', y='recruits', c='k', s=50, ax=ax)
ax.set_ylim(0, 350);


# In[ ]:


with gp_salmon_model:
    salmon_pred_noise = recruit_gp.conditional("salmon_pred_noise", X_pred.reshape(-1,1), pred_noise=True)
    gp_salmon_samples = pm.sample_posterior_predictive(gp_trace, vars=[salmon_pred_noise], samples=500, random_seed=42)


# In[ ]:


from pymc3.gp.util import plot_gp_dist
fig, ax = plt.subplots(figsize=(8,6))
plot_gp_dist(ax, gp_salmon_samples['salmon_pred_noise'], X_pred)
salmon_data.plot.scatter(x='spawners', y='recruits', c='k', s=50, ax=ax)
ax.set_ylim(0, 350);


# ### Exercise
# 
# We might be interested in what may happen if the population gets very large -- say, 600 or 800 spawners. We can predict this, though it goes well outside the range of data that we have observed. Generate predictions from the posterior predictive distribution that covers this range of spawners.
# 
# *Hint: you need to add a new `conditional` variable.*

# In[ ]:


# Write answer here

