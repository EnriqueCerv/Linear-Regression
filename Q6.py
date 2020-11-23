#%%
import numpy as np
import matplotlib.pyplot as plt 
import math
# %% Getting data

data = np.loadtxt('PCB.txt')
ages = data[:,0]
PCB = data[:,1]
PCB_log = np.log(data[:,1])
ages, PCB_log

# %% Lin regression algorithm q:1-3

# Function only works when input is 1d
def regression_params(input, label):
    X_mat = np.hstack((input[:,np.newaxis], np.ones((input.shape[0],1), dtype=input.dtype)))
    W = np.dot(np.linalg.inv(np.dot(X_mat.T, X_mat)), np.dot(X_mat.T, label))
    return W[0], W[1]

a,b = regression_params(ages, PCB_log)

#X = np.hstack((ages[:,np.newaxis], np.ones((ages.shape[0],1), dtype=ages.dtype)))
#w = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, PCB_log))
#a = w[0]
#b = w[1]

x = np.linspace(0, np.max(ages),120)
#exp_regression = np.exp(a*x+b)
exp_regression = (a*x+b)

plt.figure()
plt.plot(x, exp_regression, 'r-', label='Linear model')
plt.plot(ages, np.log(PCB), 'k.', label='Data')
plt.title('ln(PCB) Concentration in fish')
plt.xlabel('age (years)')
plt.ylabel('ln(PCB) concentration (ln(ppm))')
plt.legend()
#plt.show()


# %% MSE for linear model, q3

PCB_lin_reg = a*ages+b
MSE_lin = ((PCB_log-PCB_lin_reg)**2).sum()/PCB.shape[0]
MSE_lin

# %% R squared value for linear model, q4

# To double check the above
#plt.figure()
#plt.plot(ages,np.exp(PCB_lin_reg), 'r.', ages, PCB, 'k.', x, regression)

numerator = ((PCB_log-PCB_lin_reg)**2).sum()
denominator = ((PCB_log-PCB_lin_reg.mean())**2).sum()
R_sqr_lin = 1-numerator/denominator
R_sqr_lin

# %% Non linear model q5

ages_sqrt = np.sqrt(ages)
a2,b2 = regression_params(ages_sqrt, PCB_log)

x = np.linspace(0, np.max(ages),120)
#nlin_regression = np.exp(a2*np.sqrt(x)+b2)
nlin_regression = a2*np.sqrt(x)+b2

plt.figure()
plt.plot(x, nlin_regression, 'b-', label='Non-Linear model')
plt.plot(ages, np.log(PCB), 'k.', label='Data')
plt.title('ln(PCB) Concentration in fish')
plt.xlabel('age (years)')
plt.ylabel('ln(PCB) concentration (ln(ppm))')
plt.legend()
#plt.show() 

# %% MSE for non linear model, q5
PCB_nlin_reg = a2*np.sqrt(ages)+b2
MSE_nlin = ((PCB_log-PCB_nlin_reg)**2).sum()/PCB.shape[0]
MSE_nlin

# %% R squared value for non linear model, q5

# To double check the above
#plt.figure()
#plt.plot(ages,np.exp(PCB_nlin_reg), 'r.', ages, PCB, 'k.', x, regression)

numerator = ((PCB_log-PCB_nlin_reg)**2).sum()
denominator = ((PCB_log-PCB_nlin_reg.mean())**2).sum()
R_sqr_nlin = 1-numerator/denominator
R_sqr_nlin

# %% Final results
print('For the linear model we have \n parameters ', a, b, '\n the MSE is ', MSE_lin, '\n and the R^2 is ', R_sqr_lin)
print('For the non linear model we have \n parameters ', a2, b2, '\n the MSE is ', MSE_nlin, '\n and the R^2 is ', R_sqr_nlin)
plt.show()