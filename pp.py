import pymc3 as pm
import csv
import warnings
import matplotlib
from matplotlib import pyplot as plt

confirmed = dict()
death = dict()
released = dict()
candidate = dict()
with open('corona_data.csv', 'r') as rf:
    reader = csv.reader(rf, delimiter=',')
    for row in reader:
        if row[1] != 'date':
            date = int(row[1])
            if 20200218 <= date and date <= 20200318:
                confirmed[date] = int(row[2])
                death[date] = int(row[3])
                released[date] = int(row[4])
                candidate[date] = int(row[5])

month_confirmed = sum(confirmed.values())
month_death = sum(death.values())
month_released = sum(released.values())
month_candidate = sum(candidate.values())

b_mu = 0.09
s_mu = 1/7
g_mu = month_released/month_confirmed
a_mu = month_death/month_confirmed

with pm.Model() as model:
    # constants
    c0 = 0
    N = 51844627

    # Normalized value
    beta = pm.Normal('beta', mu = b_mu, sigma = 0.001)
    #sigma = pm.Normal('sigma', mu = s_mu, sigma = 0.001)
    #gamma = pm.Normal('gamma', mu = g_mu, sigma = 0.001)
    #alpha = pm.Normal('alpha', mu = a_mu, sigma = 0.001)

    step = pm.Metropolis(vars = [beta])
    trace = pm.sample(1000, step)

warnings.simplefilter("ignore")

pm.traceplot(trace)
plt.show()
