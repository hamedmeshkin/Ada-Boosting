import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.dates as mdates



my_year_month_fmt = mdates.DateFormatter('%m/%y')

data = pd.read_pickle('three_stocks.pkl')
#print(data.head())


# Calculating the short-window simple moving average
short_rolling = data.rolling(window=20).mean()
#short_rolling.head(20)

# Calculating the long-window simple moving average
long_rolling = data.rolling(window=100).mean()
#print(long_rolling.tail())


# Using Pandas to calculate a 20-days span EMA. adjust=False specifies that we are interested in the recursive calculation mode.
ema_short = data.ewm(span=20, adjust=False).mean()


start_date = '2015-01-01'
end_date = '2016-12-31'


fig = plt.figure(figsize=(15,9))
ax = fig.add_subplot(1,1,1)


ax.plot(data.ix[start_date:end_date, :].index, data.ix[start_date:end_date, 'MSFT'], label='Price')
ax.plot(long_rolling.ix[start_date:end_date, :].index, long_rolling.ix[start_date:end_date, 'MSFT'], label = '100-days SMA')
ax.plot(short_rolling.ix[start_date:end_date, :].index, short_rolling.ix[start_date:end_date, 'MSFT'], label = '20-days SMA')
ax.plot(ema_short.ix[start_date:end_date, :].index, ema_short.ix[start_date:end_date, 'MSFT'], label = 'Span 20-days EMA')

ax.legend(loc='best')
ax.set_ylabel('Price in $')
ax.xaxis.set_major_formatter(my_year_month_fmt)

plt.show()



