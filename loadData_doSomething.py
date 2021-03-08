import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#snp = pd.read_pickle('sp500_adj_close.pkl')
data = pd.read_pickle('three_stocks.pkl')


# Calculating the short-window moving average
short_rolling = data.rolling(window=20).mean()
print(short_rolling.head())


# Calculating the long-window moving average
long_rolling = data.rolling(window=100).mean()
print(long_rolling.tail())


# Relative returns
returns = data.pct_change(1)
print(returns.head())


# Log returns - First the logarithm of the prices is taken and the the difference of consecutive (log) observations
log_returns = np.log(data).diff()
print(log_returns.head())


########################################
fig = plt.figure(figsize=[16,9])

ax = fig.add_subplot(2,1,1)

for c in log_returns:
    ax.plot(log_returns.index, log_returns[c].cumsum(), label=str(c))

ax.set_ylabel('Cumulative log returns')
ax.legend(loc='best')
ax.grid()

ax = fig.add_subplot(2,1,2)

for c in log_returns:
    ax.plot(log_returns.index, 100*(np.exp(log_returns[c].cumsum()) - 1), label=str(c))

ax.set_ylabel('Total relative returns (%)')
ax.legend(loc='best')
ax.grid()

#plt.show()

###########################################

# Last day returns. Make this a column vector
r_t = log_returns.tail(1).transpose()
print(r_t)


# Weights as defined above
weights_vector = pd.DataFrame(1 / 3, index=r_t.index, columns=r_t.columns)
print(weights_vector)


# Total log_return for the portfolio is:
portfolio_log_return = weights_vector.transpose().dot(r_t)
print(portfolio_log_return)

####################################

weights_matrix = pd.DataFrame(1 / 3, index=data.index, columns=data.columns)
#weights_matrix.tail()

# Initially the two matrices are multiplied. Note that we are only interested in the diagonal, 
# which is where the dates in the row-index and the column-index match.
temp_var = weights_matrix.dot(log_returns.transpose())
print(temp_var.head().ix[:, 0:5])

# The numpy np.diag function is used to extract the diagonal and then
# a Series is constructed using the time information from the log_returns index
portfolio_log_returns = pd.Series(np.diag(temp_var), index=log_returns.index)
#portfolio_log_returns.tail()

###########################################

tot_relative_returns = np.exp(portfolio_log_returns.cumsum()) - 1

fig = plt.figure(figsize=[16,9])

ax = fig.add_subplot(2,1,1)
ax.plot(portfolio_log_returns.index, portfolio_log_returns.cumsum())
ax.set_ylabel('portfolio cumulative log returns')
ax.grid()

ax = fig.add_subplot(2,1,2)
ax.plot(tot_relative_returns.index, tot_relative_returns * 100)
ax.set_ylabel('portfolio total relative returns (%)')
ax.grid()

plt.show()


# Calculating the time-related parameters of the simulation
days_per_year = 52 * 5
total_days_in_simulation = data.shape[0]
number_of_years = total_days_in_simulation / days_per_year

# The last data point will give us the total portfolio return
total_portfolio_return = tot_relative_returns[-1]
# Average portfolio return assuming compunding of returns
average_yearly_return = (1 + total_portfolio_return)**(1 / number_of_years) - 1

print('Total portfolio return is: ' +
      '{:5.2f}'.format(100 * total_portfolio_return) + '%')
print('Average yearly return is: ' +
      '{:5.2f}'.format(100 * average_yearly_return) + '%')
