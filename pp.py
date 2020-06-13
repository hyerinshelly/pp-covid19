import pymc3 as pm
import csv

confirmed = dict()
death = dict()
released = dict()
candidate = dict()
with open('corona_data.csv', 'r') as rf:
    reader = csv.reader(rf, delimiter=',')
    for row in reader:
        if isinstance(row[1], int):
            confirmed[row[1]] = row[2]
            death[row[1]] = row[3]
            released[row[1]] = row[4]
            candidate[row[1]] = row[5]

month_confirmed = sum(confirmed.values())
month_death = sum(death.values())
month_released = sum(released.values())
month_candidate = sum(candidate.values())

with pm.Model() as model:
    # constants
    c0 = 0
    N = 0

    # Normalized value
    beta = pm.Normal('beta', mu = , sigma = 0.001 )
    sigma = 0
    gamma = 0
    alpha = 0

    pm.Metropolis(vars = [beta, c0, sigma ...])