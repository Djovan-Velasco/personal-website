######### Project 170 ##########
## To what extent do asymmetric GARCH models outperform symmetric models in explaining historical variance 
## and forecasting conditional volatility of the US/UK exchange rate during geopolitical crises?
library(dplyr)
library(astsa)
library(rugarch)
# read data
ex_rate <- read.csv("EXUSUK.csv")

# convert to ts object
ex_rate_ts <- ts(ex_rate$EXUSUK, start = c(1971, 1), frequency = 12)

# plot data 
tsplot(ex_rate_ts)
acf2(ex_rate_ts)
## acf shows autocorrelation

## add dummy variable regressor for geopolitical events that may have caused spikes
shock_dates <- c(
  "1971-08-01", # End of Bretton Woods 
  "1985-09-01", # The Plaza Accord
  "1992-09-01", # Black Wednesday
  "2008-09-01", # Global Financial Crisis (Lehman collapse)
  "2016-06-01", # Brexit 
  "2020-03-01", # COVID-19
  "2022-09-01"  # UK "Mini-Budget" Crash
)

ex_rate$Crisis_Flag <- ifelse(ex_rate$observation_date %in% shock_dates, 1, 0)

dummy <- ex_rate$Crisis_Flag[-1]
dummy_matrix <- as.matrix(dummy)

## apply transformation and differencing and plot
diff_series <- diff(log(ex_rate_ts))
tsplot(diff_series)
acf1(diff_series)
acf1(diff_series^2)
## ------------- explanation -----------------
## acf shows evidence of volatility
## differenced series shows autocorrelation
## -------------------------------------------

# test mean model
acf2(diff_series)
ex_rate_arima <- sarima(diff_series, 1, 0, 0)$fit$resid
acf2(ex_rate_arima^2) 
hist(scale(ex_rate_arima))
## ------------- explanation -----------------
## tails are fat
## squared returns still have autocorrelation
## evidence that garch model may fit best 
## -------------------------------------------

## Test GARCH with t-dist to see if volatility symmetric
ex_rate_garch <- ugarchspec(
  variance.model = list(
    model = "sGARCH", 
    garchOrder = c(1, 1), 
    external.regressors = dummy_matrix
  ),
  mean.model = list(
    armaOrder = c(1, 0),
    include.mean = TRUE
  ),
  distribution.model = "std"
)
garch_fit <- ugarchfit(spec = ex_rate_garch, data = diff_series)
print(garch_fit)

## no evidence of excess volatility
## removes autocorrletion
## alpha + beta < 1 so stationary
## shape signficant, t dist better over normal
## ---------------------------------

## Test EGARCH/GJRGARCH for better fit and asymmetric
ex_rate_egarch <- ugarchspec(
  variance.model = list (
    model = "eGARCH",
    garchOrder = c(1,1),
    external.regressors = dummy_matrix
  ),
  mean.model = list(
    armaOrder = c(1, 0),
    include.mean = TRUE
  ),
  distribution.model = "std"
)
egarch_fit <- ugarchfit(spec = ex_rate_egarch, data = diff_series)
print(egarch_fit)
  
# GJR-GARCH Specification
ex_rate_gjr <- ugarchspec(
  variance.model = list(
    model = "gjrGARCH",            
    garchOrder = c(1, 1), 
    external.regressors = dummy_matrix
  ),
  mean.model = list(
    armaOrder = c(1, 0),            
    include.mean = TRUE
  ),
  distribution.model = "std"        
)
gjr_fit <- ugarchfit(spec = ex_rate_gjr, data = diff_series)
print(gjr_fit)
## check output significance (a geopolitical event does cause volatility) then test positive vs negative shock

## -------------- Prediction -------------

## split data 80/20
## use first 80% years to predict the remaining 20% of years and use rmse to compare to original series
split <- floor(0.8 * length(diff_series))
train <- diff_series[1:split]
test <- diff_series[(split + 1):length(diff_series)]

dummy_train <- dummy_matrix[1:split, , drop=FALSE]
dummy_test  <- dummy_matrix[(split + 1):nrow(dummy_matrix), , drop=FALSE]

actual <- test^2

## train
ex_rate_garch_train <- ugarchspec(
  variance.model = list(
    model = "sGARCH", 
    garchOrder = c(1, 1), 
    external.regressors = dummy_train
  ),
  mean.model = list(
    armaOrder = c(1, 0),
    include.mean = TRUE
  ),
  distribution.model = "std"
)
garch_fit_train <- ugarchfit(spec = ex_rate_garch_train, data = train)
print(garch_fit_train)

ex_rate_egarch_train <- ugarchspec(
  variance.model = list (
    model = "eGARCH",
    garchOrder = c(1,1),
    external.regressors = dummy_train
  ),
  mean.model = list(
    armaOrder = c(1, 0),
    include.mean = TRUE
  ),
  distribution.model = "std"
)
egarch_fit_train <- ugarchfit(spec = ex_rate_egarch_train, data = train)
print(egarch_fit_train)

ex_rate_gjr_train <- ugarchspec(
  variance.model = list(
    model = "gjrGARCH",            
    garchOrder = c(1, 1), 
    external.regressors = dummy_train
  ),
  mean.model = list(
    armaOrder = c(1, 0),            
    include.mean = TRUE
  ),
  distribution.model = "std"        
)
gjr_fit_train <- ugarchfit(spec = ex_rate_gjr_train, data = train)
print(gjr_fit_train)

bic_garch  <- infocriteria(garch_fit_train)[2]
bic_egarch <- infocriteria(egarch_fit_train)[2]
bic_gjr    <- infocriteria(gjr_fit_train)[2]

bic_comparison <- data.frame(
  Model = c("Standard GARCH(1,1)", "EGARCH(1,1)", "GJR-GARCH(1,1)"),
  BIC = c(bic_garch, bic_egarch, bic_gjr)
)

bic_comparison <- bic_comparison[order(bic_comparison$BIC), ]

rownames(bic_comparison) <- NULL

print(bic_comparison)
## test
garch_test <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1), external.regressors = dummy_test),
  mean.model = list(armaOrder = c(1, 0), include.mean = TRUE),
  distribution.model = "std"
)

egarch_test <- ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1, 1), external.regressors = dummy_test),
  mean.model = list(armaOrder = c(1, 0), include.mean = TRUE),
  distribution.model = "std"
)

gjr_test <- ugarchspec(
  variance.model = list(model = "gjrGARCH", garchOrder = c(1, 1), external.regressors = dummy_test),
  mean.model = list(armaOrder = c(1, 0), include.mean = TRUE),
  distribution.model = "std"
)

setfixed(garch_test) <- as.list(coef(garch_fit_train))
setfixed(egarch_test) <- as.list(coef(egarch_fit_train))
setfixed(gjr_test) <- as.list(coef(gjr_fit_train))

filter_garch <- ugarchfilter(garch_test, data = test)
filter_egarch <- ugarchfilter(egarch_test, data = test)
filter_gjr <- ugarchfilter(gjr_test, data = test)

pred_var_garch_dynamic <- sigma(filter_garch)^2
pred_var_egarch_dynamic <- sigma(filter_egarch)^2
pred_var_gjr_dynamic <- sigma(filter_gjr)^2

rmse_garch_dyn <- sqrt(mean((actual - pred_var_garch_dynamic)^2))
rmse_egarch_dyn <- sqrt(mean((actual - pred_var_egarch_dynamic)^2))
rmse_gjr_dyn <- sqrt(mean((actual - pred_var_gjr_dynamic)^2))

print(c("Standard GARCH" = rmse_garch_dyn, "EGARCH" = rmse_egarch_dyn, "GJR-GARCH" = rmse_gjr_dyn))

## plot comparison
test_start_time <- time(diff_series)[split + 1]

ts_actual <- ts(actual, start = test_start_time, frequency = 12)
ts_garch  <- ts(pred_var_garch_dynamic, start = test_start_time, frequency = 12)
ts_egarch <- ts(pred_var_egarch_dynamic, start = test_start_time, frequency = 12)
ts_gjr    <- ts(pred_var_gjr_dynamic, start = test_start_time, frequency = 12)

tsplot(ts_actual, type = "l", col = "gray80", lwd = 2,
     ylab = "Volatility (Variance)", 
     xlab = "Year",
     main = "1-Step Ahead Dynamic Volatility Forecast vs Actual",
     ylim = c(0, max(actual))) 

# Overlay the dynamic filters
lines(ts_garch, col = "blue", lwd = 2)
lines(ts_egarch, col = "darkgreen", lwd = 2, lty = 2) 
lines(ts_gjr, col = "red", lwd = 2, lty = 3) 

legend("topleft", 
       legend = c("Actual Squared Returns", "Standard GARCH", "EGARCH", "GJR-GARCH"),
       col = c("gray80", "blue", "darkgreen", "red"),
       lty = c(1, 1, 2, 3), 
       lwd = 2, 
       bty = "n", 
       cex = 0.9)
