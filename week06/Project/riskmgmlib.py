import pandas as pd
import numpy as np
import scipy
import copy
#1. Covariance estimation techniques


# Generating expoentially weighted weights
def weight_gen(n, lambd = 0.94):
    weight = np.zeros(n)
    for i in range(n):
        weight[i] = (1-lambd) * (lambd) ** i
    normalized_weight = weight / np.sum(weight)
    return normalized_weight


def ewcov_gen(data, weight):
    data = data - data.mean(axis=0)
    weight = np.diag(weight)
    data_left = weight@data
    data_right = np.dot(data.T,data_left)
    return data_right
#2. Non PSD fixes for correlation matrices

# chol_psd function, return the lower half matrix
def chol_psd(a):
    n = a.shape[1]
    #Initialize the root matrix with 0 values
    root = np.zeros((n,n))
    #loop over columns
    for j in range(n):
        s = 0
        #if we are not on the first column, calculate the dot product of the preceeding row values.
        if j>0:
            s = root[j,0:j].T@root[j,0:j]
        temp = a[j,j] - s
        # here temp is the critical value, when temp>=-1e-3, there is no nan but still invalid answer, but it is close
        if temp<=0 and temp>=-1e-3:
            temp = 0
        root[j,j] = np.sqrt(temp)
        #Check for the 0 eigan value.  Just set the column to 0 if we have one
        if root[j,j] == 0:
            for i in range(j,n):
                root[j,i] = 0
        else:
            ir = 1/root[j,j]
            for i in range(j+1,n):
                s = root[i,0:j].T@root[j,0:j]
                root[i,j] = (a[i,j] - s) * ir
    return root

# fixing psd matrix
def near_psd(a, epsilon = 0.0):
    is_cov = False
    for i in np.diag(a):
        if abs(i-1)>1e-8:
            is_cov = True
        else:
            is_cov = False
            break
    if is_cov:
        invSD = np.diag(1/np.sqrt(np.diag(a)))
        a = invSD@a@invSD
    vals, vecs = np.linalg.eigh(a)
    vals = np.array([max(i,epsilon) for i in vals])
    T = 1/(np.square(vecs) @ vals)
    T = np.diag(np.sqrt(T))
    l = np.diag(np.sqrt(vals))
    B  = T @ vecs @ l
    out = B @ B.T
    if is_cov:
        invSD = np.diag(1/np.diag(invSD))
        out = invSD @ out @ invSD
    return out

#Implement Higham’s 2002 nearest psd correlation function
def Frobenius_Norm(a):
    return np.sqrt(np.sum(np.square(a)))

def projection_u(a):
    np.fill_diagonal(a, 1.0)
    return a

# A note here, epsilon is the smallest eigenvalue, 0 does not work well here, will still generate very small negativa values, so I set it to 1e-7
def projection_s(a, epsilon = 1e-7):
    vals, vecs = np.linalg.eigh(a)
    vals = np.array([max(i,epsilon) for i in vals])
    return vecs@np.diag(vals)@vecs.T

def Higham_method(a, tol = 1e-8):
    s = 0
    gamma = np.inf
    y = a
    # iteration
    while True:
        r = y - s
        x = projection_s(r) 
        s = x - r
        y = projection_u(x)
        gamma_next = Frobenius_Norm(y-a)
        if abs(gamma - gamma_next) < tol:
            break
        gamma = gamma_next
    return y

# if a matrix is psd
def is_psd(matrix):
    eigenvalues = np.linalg.eigvals(matrix)
    return np.all(eigenvalues >= 0)
#3. Simulation Methods

def sim_mvn_from_cov(cov, num_of_simulation=25000):
    return chol_psd(cov) @ np.random.normal(size=(cov.shape[0], num_of_simulation))

# variance matrix
def var(cov):
    return np.diag(cov)
# Correlation matrix
def corr(cov):
    return np.diag(1/np.sqrt(var(cov))) @ cov @ np.diag(1/np.sqrt(var(cov))).T
# Covariance matrix
def cov(var, cor):
    std = np.sqrt(var)
    return np.diag(std) @ cor @ np.diag(std).T


# using PCA with an optional parameter for % variance explained.
# return the simulation result
def PCA_with_percent(cov, percent = 0.95, num_of_simulation = 25000):
    eigenvalue, eigenvector = np.linalg.eigh(cov)
    total = np.sum(eigenvalue)
    for i in range(cov.shape[0]):
        i = len(eigenvalue)-i-1
        if eigenvalue[i]<0:
            eigenvalue = eigenvalue[i+1:]
            eigenvector = eigenvector[:,i+1:]
            break
        if sum(eigenvalue[i:])/total > percent:
            eigenvalue = eigenvalue[i:]
            eigenvector = eigenvector[:,i:]
            break
    simulate = np.random.normal(size = (len(eigenvalue),num_of_simulation))
    return eigenvector @ np.diag(np.sqrt(eigenvalue)) @ simulate

# direct simulation
def direct_simulation(cov, n_samples=25000):
    B = chol_psd(cov)
    r = scipy.random.randn(len(B[0]), n_samples)
    return B @ r

#4. VaR calculation methods

# Given data and alpha, return the VaR
def calculate_var(data, mean=0, alpha=0.05):
    return mean-np.quantile(data, alpha)


def normal_var(data, mean=0, alpha=0.05, nsamples=10000):
    sigma = np.std(data)
    simulation_norm = np.random.normal(mean, sigma, nsamples)
    var_norm = calculate_var(simulation_norm, mean, alpha)
    return var_norm

def ewcov_normal_var(data, mean=0, alpha=0.05, nsamples=10000, lambd = 0.94):
    ew_cov = ewcov_gen(data, weight_gen(len(data), lambd))
    ew_variance = ew_cov
    sigma = np.sqrt(ew_variance)
    simulation_ew = np.random.normal(mean, sigma, nsamples)
    var_ew = calculate_var(simulation_ew, mean, alpha)
    return var_ew

def t_var(data, mean=0, alpha=0.05, nsamples=10000):
    params = scipy.stats.t.fit(data, method="MLE")
    df, loc, scale = params
    simulation_t = scipy.stats.t(df, loc, scale).rvs(nsamples)
    var_t = calculate_var(simulation_t, mean, alpha)
    return var_t

def historic_var(data, mean=0, alpha=0.05):
    return calculate_var(data, mean, alpha)


#5. ES calculation
def calculate_es(data, mean = 0, alpha=0.05):
    return -np.mean(data[data<-calculate_var(data, mean, alpha)])

#6. Other functions

# Given a price series, return the returns
def return_calculate(price, method = 'discrete'):
    returns = []
    for i in range(len(price)-1):
        returns.append(price[i+1]/price[i])
    returns = np.array(returns)
    if method == 'discrete':
        return returns - 1
    if method == 'log':
        return np.log(returns)