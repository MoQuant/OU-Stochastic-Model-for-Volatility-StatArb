import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

def xmedian(x):
    n = len(x)
    if n % 2 == 0:
        i = int(n / 2)
        return x[i-1]
    else:
        i = int(n / 2)
        return x[i]

def Regression(x, y):
    H = np.vstack([np.ones(len(x)), x]).T
    ixtx = np.linalg.inv(H.T.dot(H))
    beta = ixtx.dot(H.T.dot(y))
    factor = sum((y - H @ beta)**2)/(len(x) - 2)
    stderr = np.sqrt(np.diag(factor*ixtx))
    tstat = beta / stderr 
    print('Significane in Parameters')

    for i, j in zip(beta, tstat):
        print(f'Beta = {i} | TStat = {j}')
    print('\n')
    return beta, H

stockA = 'GS'
stockB = 'JPM' 

fetch = lambda a: yf.Ticker(a).history(period='1y')

dataA = fetch(stockA)
dataB = fetch(stockB)

priceA = dataA['Close']
priceB = dataB['Close']

t = 30.0 / 252.0
n = 400
dt = t / n

p = 100
window = 30

rorA = priceA.pct_change().dropna()
rorB = priceB.pct_change().dropna()

volA = rorA.rolling(window=window).std().dropna().values 
volB = rorB.rolling(window=window).std().dropna().values

B1, H1 = Regression(volA, volB)

X = volB - H1 @ B1

B2, H2 = Regression(X[:-1], X[1:])

a, b = B2

theta = -np.log(b) / dt
mu = a / (1 - b)
vx = np.std(X)*np.sqrt((-2*np.log(b))/(dt*(1-b**2)))

fig = plt.figure()
ax = fig.add_subplot(111)

grid = np.zeros((p, n))
optimal = np.zeros(n)

ax.set_title(f'Volatility StatArb with Ornstein-Uhlenbeck Model: {stockA} and {stockB}')
for i in range(p):
    Xt = X[-1]
    grid[i, 0] = Xt 
    for j in range(1, n):
        dWT = np.random.normal() 
        Xt += theta*(mu - Xt)*dt + vx*dWT
        grid[i, j] = Xt
    ax.plot(grid[i, :], color='black')
    plt.pause(0.0001)

for i in range(n):
    optimal[i] = xmedian(grid[:, i])

ax.plot(optimal, color='limegreen')

plt.show()