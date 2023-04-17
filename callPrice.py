from math import log, sqrt, pi, exp
from scipy.stats import norm
from datetime import datetime, date
import numpy as np
import pandas as pd
from pandas import DataFrame
import concurrent.futures
import os
def generate_heston_paths(S, T, r, kappa, theta, v_0, rho, xi, 
                          steps, Npaths, return_vol=False):
    dt = T/steps
    size = (Npaths, steps)
    prices = np.zeros(size)
    sigs = np.zeros(size)
    S_t = S
    v_t = v_0
    for t in range(steps):
        WT = np.random.multivariate_normal(np.array([0,0]), 
                                           cov = np.array([[1,rho],
                                                          [rho,1]]), 
                                           size=Npaths) * np.sqrt(dt) 
        
        S_t = S_t*(np.exp( (r- 0.5*v_t)*dt+ np.sqrt(v_t) *WT[:,0] ) ) 
        v_t = np.abs(v_t + kappa*(theta-v_t)*dt + xi*np.sqrt(v_t)*WT[:,1])
        prices[:, t] = S_t
        sigs[:, t] = v_t
    
    if return_vol:
        return prices, sigs
    
    return prices

def heston_call(S, K, T, r, kappa, theta, v_0, rho, xi):
    prices_pos = generate_heston_paths(S, T, r, kappa, theta,
                                    v_0, rho, xi, steps=1000, Npaths=10000,
                                    return_vol=False)[:,-1]
    return np.max(np.mean(prices_pos) - K,0) * np.exp(-r*T)


S = np.linspace(10, 500, num = 1000)
T = np.linspace(1/12, 3, num= 1000)
r = np.linspace(0.01, 0.1, num=100)
v_0 = np.linspace(0.05, 0.9, num=100)
theta = np.linspace(0.01, 0.8, num = 100)
kappa = np.linspace(0, 10, num = 100)
# xi < 2 * theta * kappa
l = [S, T, r, v_0, theta, kappa]

param_List = []
for i in range(30000):
    params = []
    for p in l:
        params.append(np.random.choice(p))
    param_List.append(params)



np.random.seed(42)
for params in param_List:
    S = params[0]
    T = params[1]
    t = np.random.choice(np.linspace(1/12, T, num= 1000))
    q = np.random.choice(np.linspace(-0.2, 0.2, num= 1000))
    K = np.exp(q * t) * S
    params.insert(1, K)
    theta = params[-2]
    kappa = params[-1]
    xi = np.random.choice(np.linspace(0, np.sqrt(2 * theta * kappa), num= 1000))
    params.append(xi)



def calculate_call_price(params):
    S = params[0]
    K = params[1]
    T = params[2]
    r = params[3]
    v_0 = params[4]
    theta = params[5]
    kappa = params[6]
    xi = params[7]
    call_price = heston_call(S, K, T, r, kappa, theta, v_0, rho, xi)
    params.append(call_price)
    return params

def parallel_for_loop(param_List):
    num_processes = os.cpu_count()  # Use all available CPU cores
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(executor.map(calculate_call_price, param_List))
    return results
  # Your list of parameters
  
updated_param_List = parallel_for_loop(param_List)

df = pd.DataFrame(updated_param_List )
df.columns = ["S", "K", "T", "r", "v_0", "theta", "kappa", "xi", 'call_price']
df.to_csv("SimulationOfHeston.csv")