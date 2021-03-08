import numpy as np
from pandas_datareader import data
import pandas as pd
import talib as ta
#from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
#from sklearn.datasets import load_iris
#from sklearn.cluster import SpectralClustering
#from scipy.spatial.distance import euclidean
from project_util import *

########################
# parameters
########################
period = 10



"""
###################################
#  get data from the web
######################################
tickers = ['^GSPC']

data_source = 'yahoo'


start_date = '2000-01-01'
end_date = '2016-12-31'


panel_data = data.DataReader(tickers, data_source, start_date, end_date)
#panel_data = data.DataReader(tickers, data_source) # read all available data

data = panel_data.to_frame().unstack(level=-1)

########################################
# trim and fill NA
#########################################

all_weekdays = pd.date_range(start=start_date, end=end_date, freq='B')
data_weekdays = data.reindex(all_weekdays)
data_weekdays = data_weekdays.fillna(method='ffill')

data_weekdays.to_pickle('sp_test.pkl')
"""

######################
# load data
######################
data_weekdays = pd.read_pickle('sp_test.pkl')
open = data_weekdays['Open'].values[:,0]
high = data_weekdays['High'].values[:,0]
low = data_weekdays['Low'].values[:,0]
close = data_weekdays['Close'].values[:,0]
adj_close = data_weekdays['Adj Close'].values[:,0]
volume = data_weekdays['Volume'].values[:,0]

timeIndex = data_weekdays.index

#data_matrix = data_weekdays.values

#print(data_weekdays.tail())
#adj_close = data_weekdays['Adj Close']
#print(adj_close.values[0,:])
#print(data_weekdays.axes)

################################
# indicators
#####################################
#
# ATR, STOCH, ADX, AROON, (BBANDS ???), ADOSC, MACD, MFI, SAR,
#
#
# overlap studies -> BBANDS, SAR
# momentum indicators -> STOCH, ADX, AROON, MACD, MFI
# volume indicators -> ADOSC
# cycle indicators -> HT_DCPERIOD
# price transform -> WCLPRICE
# volatility indicators -> ATR
# pattern recognition -> CDLBELTHOLD

atr = ta.ATR(high, low, close, timeperiod = period)
stoch_k, stoch_d = ta.STOCH(high, low, close) # use deafault
adx = ta.ADX(high, low, close, timeperiod = period)

aroon_up, aroon_dn = ta.AROON(high, low, timeperiod = period)
aroon = aroon_up - aroon_dn

adosc = ta.ADOSC(high, low, close, volume)
macd, macdsignal, macdhist = ta.MACD(close) # use default
mfi = ta.MFI(high, low, close, volume, timeperiod=period)
sar = ta.SAR(high, low) # use default
ht = ta.HT_DCPERIOD(close)
wcl = ta.WCLPRICE(high, low, close)

#print(len(atr))
#print(len(stoch_d))
#print(len(adx))
#print(len(aroon))
#print(len(ad))
#print(len(macd))
#print(len(mfi))
#print(len(sar))
#print(len(ht))
#print(len(wcl))
attrNames = np.array(['atr', 'stoch', 'adx', 'aroon', 'adosc', 'macd', 'mfl', 'sar', 'ht', 'wcl'])
F = np.column_stack((atr, stoch_d, adx, aroon, adosc, macd, mfi, sar, ht, wcl)) 
#print(F)
indexSet = pd.DataFrame(F, index=timeIndex, columns=attrNames)
fSet = indexSet.fillna(indexSet.ix[-1]) # fill NaN with the last row of the dataframe
#print(fSet.head())
#fData = fSet.values

#print(fSet.describe())

#print(fSet.tail())
#print(fData)
#print(fSet.mean())
#print(fSet.std())
fScaled = (fSet - fSet.mean()) / fSet.std()
fData = fScaled.values
#print(fScaled.tail())
#spc = SpectralClustering(n_clusters=3, random_state=123, affinity='rbf', gamma=0.1)

#spc.fit(fData)
#print(spc.affinity_matrix_)

w, v = solvLap(fData)
#print(w)
phi = lambda x : calcPhi(x, w, v)
fScore = fScaled.apply(phi, axis=0)
#print(type(fScore))
#print(fScore.index)
#print(fScore.values)

#print(w)
#print(v)


################################################
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
a = fScore.values.argsort()
x = fScore.values[a]
y = np.arange(1, len(fScore.values)+1, 1)
z = fScore.index[a]
ax.scatter(x, y)
#ax.plot(fSet.index, close, label='close')
ax.set_xlabel('Date')
ax.set_ylabel('Index')
ax.set_yticks(y)
ax.set_yticklabels(z)
#ax.legend()
plt.show()





###############
# labels
#####################
#T_ind =  T(high, low, close, threshold=0.025, timeperiod=period)
