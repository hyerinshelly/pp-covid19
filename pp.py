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

acc_confirmed = 0
acc_death = 0
acc_released = 0
acc_candidate = 0

with open('corona_data.csv', 'r') as rf:
    reader = csv.reader(rf, delimiter=',')
    for row in reader:
        if row[1] != 'date':
            date = int(row[1])
            if 20200218 <= date and date <= 20200318:
                confirmed[date] = int(row[2]) - acc_confirmed
                death[date] = int(row[3]) - acc_death
                released[date] = int(row[4]) - acc_released
                candidate[date] = int(row[5]) - acc_candidate
                
            acc_confirmed = int(row[2])
            acc_death = int(row[3])
            acc_released = int(row[4])
            acc_candidate = int(row[5])

month_confirmed = sum(confirmed.values())
month_death = sum(death.values())
month_released = sum(released.values())
month_candidate = sum(candidate.values())

# constants
c0 = 6.6
N = 51844627

# Initials 
S0 = N
E0 = 0
I0 = 0
R0 = 0

# Prior
b_mu = 0.09
s_mu = 1/7
g_mu = month_released/month_confirmed
a_mu = month_death/month_confirmed

# Do posterior inference
with pm.Model() as model:

    # Normalized value
    beta = pm.Normal('beta', mu = b_mu, sigma = 0.001)
    sigma = pm.Normal('sigma', mu = s_mu, sigma = 0.001)
    gamma = pm.Normal('gamma', mu = g_mu, sigma = 0.001)
    alpha = pm.Normal('alpha', mu = a_mu, sigma = 0.001)

    step = pm.Metropolis(vars = [beta, sigma, gamma, alpha])
    trace = pm.sample(1000, step)

warnings.simplefilter("ignore")

pm.traceplot(trace)
plt.show()

# Get probability using values from posterior inference
h = 1


P11 = 1 - np.exp(-1 * beta * (c0 * I)/N * h) #여기 I도 시간걸린당,,
P21 = 1 - np.exp(-1 * sigma * h)
P32 = 1 - np.exp(-1 * alpha * h)
P33 = 1 - np.exp(-1 * gamma * h)

# pm.Binomial #이런 함수가 있넹
