import pymc3 as pm
import numpy as np
import csv
import warnings
import matplotlib
from matplotlib import pyplot as plt

# Get data
confirmed = dict()
death = dict()
released = dict()
candidate = dict()
negative = dict()

acc_confirmed = 0
acc_death = 0
acc_released = 0
acc_candidate = 0
acc_negative = 0

with open('corona_data.csv', 'r') as rf:
    reader = csv.reader(rf, delimiter=',')
    for row in reader:
        if row[1] != 'date':
            date = int(row[1])
            if 20200301 <= date and date <= 20200331:
                confirmed[date] = int(row[2]) - acc_confirmed
                death[date] = int(row[3]) - acc_death
                released[date] = int(row[4]) - acc_released
                candidate[date] = int(row[5]) - acc_candidate
                negative[date] = int(row[6]) - acc_negative
                
            acc_confirmed = int(row[2])
            acc_death = int(row[3])
            acc_released = int(row[4])
            acc_candidate = int(row[5])
            acc_negative = int(row[6])

data = dict()
data['confirmed'] = confirmed
data['death'] = death
data['released'] = released
data['candidate'] = candidate
data['negative'] = negative


def SEIR_model(
    # constants
    N = 51844627,
    c0 = 15,
    data = data,
):
    # Initials
    S0 = N
    E0 = 3317
    I0 = 3736
    R0 = 30

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

    pm.traceplot(trace)
    plt.show()

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

        #이거 샘플 몇개 뽑지? 일단 100개 해볼게
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
    for i in range(30):
        new_S, new_E, new_I, new_R = new_SEIR(S, E, I, R)
        result['S'].append(new_S)
        result['E'].append(new_E)
        result['I'].append(new_I)
        result['R'].append(new_R)
        S, E, I, R = new_S, new_E, new_I, new_R
    
    print(result)

SEIR_model()