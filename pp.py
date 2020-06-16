import pymc3 as pm
import numpy as np
import csv
import warnings
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd

# Get data
confirmed = dict()
death = dict()
released = dict()
candidate = dict()
negative = dict()

I_data = []
with open('corona_data_daily.csv', 'r') as rf:
    reader = csv.reader(rf, delimiter=',')
    for row in reader:
        date = int(row[0])
        if 20200301 <= date and date <= 20200331:
            confirmed[date] = int(row[1])
            death[date] = int(row[2])
            released[date] = int(row[3])
            candidate[date] = int(row[4])
            negative[date] = int(row[5])
        I_data.append(int(row[1]))



data = dict()
data['confirmed'] = confirmed
data['death'] = death
data['released'] = released
data['candidate'] = candidate
data['negative'] = negative

def SEIR_model(
    total_duration = -1,
    # constants
    N = 51844627,
    c0 = 6.6,
    data = data,
    S0 = -1,
    E0 = -1,
    I0 = -1,
    R0 = -1,
):
    # Initials
    S0 = N if S0 == -1 else S0
    E0 = 0 if E0 == -1 else E0
    I0 = 1 if I0 == -1 else I0
    R0 = 0 if R0 == -1 else R0

    # Population accumulation of each cases
    month_confirmed = sum(data['confirmed'].values())
    month_death = sum(data['death'].values())
    month_released = sum(data['released'].values())
    month_candidate = sum(data['candidate'].values())

    # Observation (y)
    b0 = month_confirmed / month_candidate
    s0 = 1/7
    g0 = 0.08 #month_released/month_confirmed
    a0 = month_death/month_confirmed

    # Do posterior inference
    with pm.Model() as model:

        # Normalized value
        # Prior (p(x))
        b_mu = pm.Normal('b_mu', mu = 0, sigma = 1)
        s_mu = pm.Normal('s_mu', mu = 0, sigma = 1)
        g_mu = pm.Normal('g_mu', mu = 0, sigma = 1)
        a_mu = pm.Normal('a_mu', mu = 0, sigma = 1)

        # Likelihood (p(y|x))
        beta = pm.Normal('beta', mu = b_mu, sigma = 0.001, observed = b0)
        sigma = pm.Normal('sigma', mu = s_mu, sigma = 0.001, observed = s0)
        gamma = pm.Normal('gamma', mu = g_mu, sigma = 0.001, observed = g0)
        alpha = pm.Normal('alpha', mu = a_mu, sigma = 0.001, observed = a0)

        step = pm.Metropolis(vars = [b_mu, s_mu, g_mu, a_mu])
        trace = pm.sample(1000, step)

    warnings.simplefilter("ignore")

    # pm.traceplot(trace)
    # plt.show()

    # Approximate samples from posterior (p(x|y))
    ppc = pm.sample_posterior_predictive(trace, model=model, samples=100)

    # Uses the mean of samples as the approximate value
    appr_beta = ppc['beta'].mean()
    appr_sigma = ppc['sigma'].mean()
    appr_gamma = ppc['gamma'].mean()
    appr_alpha = ppc['alpha'].mean()

    # Get probability using values from posterior inference
    h = 1

    P21 = 1 - np.exp(-1 * appr_sigma * h)
    P32 = 1 - np.exp(-1 * appr_alpha * h)
    P33 = 1 - np.exp(-1 * appr_gamma * h)

    # SEIR
    def new_SEIR(S_t, E_t, I_t, R_t):
        P11 = 1 - np.exp(-1 * appr_beta.item() * (c0 * I_t)/N * h)

        B11 = pm.Poisson.dist(mu = S_t * P11).random(size = 100).mean()        
        B21 = pm.Binomial.dist(n = E_t, p = P21).random(size = 100).mean()
        B32 = pm.Binomial.dist(n = I_t, p = P32).random(size = 100).mean()
        B33 = pm.Binomial.dist(n = I_t, p = P33).random(size = 100).mean()
        
        new_S = S_t - B11
        new_E = E_t + B11 - B21
        new_I = I_t + B21 - B32 - B33
        new_R = R_t + B33

        return new_S, new_E, new_I, new_R 

    result = {'S': [S0], 'E': [E0], 'I':[I0], 'R':[R0]}
    S, E, I, R = S0, E0, I0, R0
    if total_duration == -1:
        flag = 0
        idx = 0
        while True:
            if flag == 1 and I <= 5:
                break
            if I > 100:
                flag = 1
            if idx%10 == 0:
                print(I)
            new_S, new_E, new_I, new_R = new_SEIR(S, E, I, R)
            result['S'].append(new_S)
            result['E'].append(new_E)
            result['I'].append(new_I)
            result['R'].append(new_R)
            S, E, I, R = new_S, new_E, new_I, new_R
            idx+=1
    else:
        for i in range(total_duration):
            if i%10 == 0:
                print(I)
            new_S, new_E, new_I, new_R = new_SEIR(S, E, I, R)
            result['S'].append(new_S)
            result['E'].append(new_E)
            result['I'].append(new_I)
            result['R'].append(new_R)
            S, E, I, R = new_S, new_E, new_I, new_R

    return result


def SQEIR_model(
    total_duration = -1,
    # constants
    N = 51844627,
    c0 = 6.6,
    data = data,
    S0 = -1,
    E0 = -1,
    I0 = -1,
    R0 = -1,
    S_q0 = -1,
    E_q0 = -1,
    I_q0 = -1,
):
    # Initials
    S0 = N if S0 == -1 else S0
    E0 = 0 if E0 == -1 else E0
    I0 = 1 if I0 == -1 else I0
    R0 = 0 if R0 == -1 else R0
    # Added for Quarantine
    S_q0 = 0 if S_q0 == -1 else S_q0
    E_q0 = 0 if E_q0 == -1 else E_q0
    I_q0 = 0 if I_q0 == -1 else I_q0

    # Population accumulation of each cases
    month_confirmed = sum(data['confirmed'].values())
    month_death = sum(data['death'].values())
    month_released = sum(data['released'].values())
    month_candidate = sum(data['candidate'].values())

    # Observation (y)
    b0 = month_confirmed / month_candidate
    s0 = 1/7
    g0 = 0.08 #month_released/month_confirmed
    a0 = month_death/month_confirmed
    # Added for Quarantine
    q0 = 0.9 
    l0 = 1/14
    d0 = 0.1  #논문값 #증상까지 다 나타나고선 격리되는 비율
    g_q0 = 0.17 #논문값

    # Do posterior inference
    with pm.Model() as model:

        # Normalized value
        # Prior (p(x))
        b_mu = pm.Normal('b_mu', mu = 0, sigma = 1)
        s_mu = pm.Normal('s_mu', mu = 0, sigma = 1)
        g_mu = pm.Normal('g_mu', mu = 0, sigma = 1)
        a_mu = pm.Normal('a_mu', mu = 0, sigma = 1)
        q_mu = pm.Normal('q_mu', mu = 0, sigma = 1)
        l_mu = pm.Normal('l_mu', mu = 0, sigma = 1)
        d_mu = pm.Normal('d_mu', mu = 0, sigma = 1)
        g_q_mu = pm.Normal('g_q_mu', mu = 0, sigma = 1)

        # Likelihood (p(y|x))
        beta = pm.Normal('beta', mu = b_mu, sigma = 0.001, observed = b0)
        sigma = pm.Normal('sigma', mu = s_mu, sigma = 0.001, observed = s0)
        gamma = pm.Normal('gamma', mu = g_mu, sigma = 0.001, observed = g0)
        alpha = pm.Normal('alpha', mu = a_mu, sigma = 0.001, observed = a0)
        q = pm.Normal('q', mu = q_mu, sigma = 0.001, observed = q0)
        lamda = pm.Normal('lamda', mu = l_mu, sigma = 0.001, observed = l0)
        delta = pm.Normal('delta', mu = d_mu, sigma = 0.001, observed = d0)
        gamma_q = pm.Normal('gamma_q', mu = g_q_mu, sigma = 0.001, observed = g_q0)

        step = pm.Metropolis(vars = [b_mu, s_mu, g_mu, a_mu, q_mu, l_mu, d_mu, g_q_mu])
        trace = pm.sample(1000, step)

    warnings.simplefilter("ignore")

#     pm.traceplot(trace)
#     plt.show()

    # Approximate samples from posterior (p(x|y))
    ppc = pm.sample_posterior_predictive(trace, model=model, samples=100)

    # Uses the mean of samples as the approximate value
    appr_beta = ppc['beta'].mean()
    appr_sigma = ppc['sigma'].mean()
    appr_gamma = ppc['gamma'].mean()
    appr_alpha = ppc['alpha'].mean()
    appr_q = ppc['q'].mean()
    appr_lamda = ppc['lamda'].mean()
    appr_delta = ppc['delta'].mean()
    appr_gamma_q = ppc['gamma_q'].mean()

    # Get probability using values from posterior inference
    h = 1

    P21 = 1 - np.exp(-1 * appr_sigma * h)
    P31 = 1 - np.exp(-1 * appr_delta * h)
    P32 = 1 - np.exp(-1 * appr_alpha * h)
    P33 = 1 - np.exp(-1 * appr_gamma * h)
    P41 = 1 - np.exp(-1 * appr_lamda * h)
    P61 = 1 - np.exp(-1 * appr_gamma_q * h)

    # SQEIR
    def new_SQEIR(S_t, E_t, I_t, R_t, S_q_t, E_q_t, I_q_t):
        P11 = 1 - np.exp(-1 * appr_beta.item() * (c0 * I_t)/N * h)
        P12 = 1 - np.exp(-1 * (1 - appr_beta.item()) * appr_q * (c0 * I_t)/N * h)

        #이거 샘플 몇개 뽑지? 일단 100개 해볼게
        B11 = pm.Poisson.dist(mu = S_t * P11).random(size = 100).mean()   
        B12 = pm.Poisson.dist(mu = S_t * P12).random(size = 100).mean()     
        B21 = pm.Binomial.dist(n = E_t, p = P21).random(size = 100).mean()
        B31 = pm.Binomial.dist(n = I_t, p = P31).random(size = 100).mean()
        B32 = pm.Binomial.dist(n = I_t, p = P32).random(size = 100).mean()
        B33 = pm.Binomial.dist(n = I_t, p = P33).random(size = 100).mean()
        B41 = pm.Binomial.dist(n = S_q_t, p = P41).random(size = 100).mean()
        B51 = pm.Binomial.dist(n = E_q_t, p = P21).random(size = 100).mean()
        B61 = pm.Binomial.dist(n = I_q_t, p = P61).random(size = 100).mean()
        B62 = pm.Binomial.dist(n = I_q_t, p = P32).random(size = 100).mean()
        
        new_S = S_t - B11 - B12 + B41
        new_E = E_t + (1 - appr_q.item()) * B11 - B21 
        new_I = I_t + B21 - B31 - B32 - B33
        new_R = R_t + B33 + B61
        new_S_q = S_q + B12 - B41
        new_E_q = E_q + appr_q.item() * B11 - B51
        new_I_q = I_q + B31 + B51 - B61 - B62

        return new_S, new_E, new_I, new_R, new_S_q, new_E_q, new_I_q

    result = {'S': [S0], 'E': [E0], 'I':[I0], 'R':[R0], 'S_q':[S_q0], 'E_q':[E_q0], 'I_q':[I_q0]}
    S, E, I, R, S_q, E_q, I_q = S0, E0, I0, R0, S_q0, E_q0, I_q0
    flag = 0
    idx = 0
    if total_duration == -1:
        while True:
            if flag == 1 and I <= 2:
                break
            if I > 10:
                flag = 1
            if idx%10 == 0:
                print(I + I_q)
            new_S, new_E, new_I, new_R, new_S_q, new_E_q, new_I_q = new_SQEIR(S, E, I, R, S_q, E_q, I_q)
            result['S'].append(new_S)
            result['E'].append(new_E)
            result['I'].append(new_I)
            result['R'].append(new_R)
            result['S_q'].append(new_S_q)
            result['E_q'].append(new_E_q)
            result['I_q'].append(new_I_q)
            S, E, I, R, S_q, E_q, I_q = new_S, new_E, new_I, new_R, new_S_q, new_E_q, new_I_q
            idx += 1
    else:
        for i in range(total_duration):
            if i%10 == 0:
                print(I + I_q)
            new_S, new_E, new_I, new_R, new_S_q, new_E_q, new_I_q = new_SQEIR(S, E, I, R, S_q, E_q, I_q)
            result['S'].append(new_S)
            result['E'].append(new_E)
            result['I'].append(new_I)
            result['R'].append(new_R)
            result['S_q'].append(new_S_q)
            result['E_q'].append(new_E_q)
            result['I_q'].append(new_I_q)
            S, E, I, R, S_q, E_q, I_q = new_S, new_E, new_I, new_R, new_S_q, new_E_q, new_I_q

    return result
# # 1번
# result = SEIR_model()

# # 2번
# result = SQEIR_model()

# 3번
result = dict()

# ~ 20200218 
result1 = SEIR_model(total_duration=29, c0=40)
print('first model finished')
# 20200219(신천지) ~ 20200322(사회적 거리두기)
result2 = SQEIR_model(total_duration=33, c0=40, S0=result1['S'][-1], E0=result1['E'][-1], I0=result1['I'][-1], R0=result1['R'][-1])
print('second model finished')
# 20200323 ~ 20200611
result3 = SQEIR_model(
    total_duration=81,
    S0=result2['S'][-1],
    E0=result2['E'][-1],
    I0=result2['I'][-1],
    R0=result2['R'][-1],
    S_q0=result2['S_q'][-1],
    E_q0=result2['E_q'][-1],
    I_q0=result2['I_q'][-1],
)
print('third model finished')

result['S'] = result1['S'] + result2['S'] + result3['S']
result['E'] = result1['E'] + result2['E'] + result3['E']
result['I'] = result1['I'] + result2['I'] + result3['I']
result['R'] = result1['R'] + result2['R'] + result3['R']
result['S_q'] = [0 for i in range(len(result1['S'])) ] + result2['S_q'] + result3['S_q']
result['E_q'] = [0 for i in range(len(result1['E'])) ] + result2['E_q'] + result3['E_q']
result['I_q'] = [0 for i in range(len(result1['I'])) ] + result2['I_q'] + result3['I_q']

data_I = []

if len(I_data) > len(result['S']):
    data_I = I_data[:len(result['S'])]
else:
    a = [0 for i in range(len(result['S']) - len(I_data))]
    data_I = I_data + a
    
df = pd.DataFrame({'x': list(range(len(result['S']))),
                   'S': [sum(x) for x in zip(result['S'], result['S_q'])],
                   'E': [sum(x) for x in zip(result['E'], result['E_q'])],
                   'I': [sum(x) for x in zip(result['I'], result['I_q'])],
                   'R': result['R'],
                   'Data_I': data_I})

plt.plot('x', 'S', data=df, color='red', label='Susceptible')
plt.plot('x', 'E', data=df, color='orange', label='Exposed')
plt.plot('x', 'I', data=df, color='green', label='Infectious')
plt.plot('x', 'R', data=df, color='skyblue', label='Recovered')
plt.plot('x', 'Data_I', data=df, color='black', label='Data')
plt.legend()
plt.show()
print('finish')
