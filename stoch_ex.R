library(xts)
library(quantmod)
library(TTR)


GSPC <- getSymbols("^GSPC", from="1970-01-01", to="2016-12-31", auto.assign=FALSE)


T.ind <- function(quotes, tgt.margin=0.025, n.days=10) {
  v <- apply(HLC(quotes), 1, mean)
  c <- Cl(quotes)
  r <- matrix(NA, ncol = n.days, nrow = NROW(quotes))
  for (x in 1:n.days) r[,x] <- Next(Delt(c, v, k=x), x)
  x <- apply(r, 1, function(p) sum(p[p > tgt.margin | p < -tgt.margin]))
  return(x)
}




#candleChart(last(GSPC, '3 months'), theme='white', TA=NULL)
#avgPrice <- function(x) apply(HLC(x), 1, mean)
#addAvgPrice <- newTA(FUN=avgPrice, col=1, legend='AvgPrice')
#addT.ind <- newTA(FUN=T.ind, col='red', legend='tgtRet')
#addAvgPrice(on=1)
#addT.ind()

myATR <- function(x) ATR(HLC(x))[,'atr']
myAroon <- function(x) aroon(cbind(Hi(x), Lo(x)))$oscillator

myATR(GSPC)[-5:-1]
myAroon(GSPC)[-5:-1]
