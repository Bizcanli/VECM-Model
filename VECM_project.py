#!/usr/bin/env python
# coding: utf-8

# In[12]:


pip install arch


# In[17]:


import warnings
import pandas as pd
from pathlib import Path
import statsmodels.api as sm
import  matplotlib.pylab as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
plt.style.use('fivethirtyeight')


# In[18]:


import arch
from arch.unitroot import ADF
from arch.unitroot import DFGLS
from arch.unitroot import PhillipsPerron
from arch.unitroot import KPSS


# In[19]:


from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss


# In[20]:


warnings.filterwarnings("ignore")
fred= pd.read_excel(r"C:\Users\07ser\OneDrive\Masaüstü\fred2.xlsx")
df= fred.set_index('DATE')
df.index


# In[ ]:


GPDI= df["GPDI"]
GDP= df["GDP"]
GE=df["GE"]
GPDI.plot()


# In[ ]:


# Augmented Dickey Fuller Test for GPDI
adf_GPDI= ADF(GPDI, trend='ct', max_lags=20)
print(adf_GPDI.summary().as_text())
reg_res = adf_GPDI.regression
residuals=reg_res.resid
print(reg_res.summary().as_text())
plot_acf(residuals,lags=60)


# In[ ]:


# p-value obtained is 0.899 which is bigger than 0.05. This suggests that GPDI is non-stationary.


# In[ ]:


dif_GPDI=GPDI.diff()
dif_GPDI=dif_GPDI.dropna()
dif_adf_GPDI = ADF(dif_GPDI, trend='ct', max_lags=10) 
print(dif_adf_GPDI.summary().as_text())
reg_res = dif_adf_GPDI.regression
residuals=reg_res.resid
print(reg_res.summary().as_text())
plot_acf(residuals,lags=60)


# In[ ]:


#p-value obtained is 0.000 which is smaller than 0.05. This suggests that dif_GPDI is stationary.


# In[ ]:


#Augmented Dickey Fuller Test for GDP
GDP.plot()
adf_GDP = ADF(GDP, trend='ct', max_lags=10) 
print(adf_GDP.summary().as_text())
reg_res = adf_GDP.regression
residuals=reg_res.resid
print(reg_res.summary().as_text())
plot_acf(residuals,lags=60)


# In[ ]:


#p-value is 0.990. So GDP is non-stationary


# In[ ]:


dif_GDP=GDP.diff()
dif_GDP=dif_GDP.dropna()
dif_adf_GDP = ADF(dif_GDP, trend='ct', max_lags=10) 
print(dif_adf_GDP.summary().as_text())
reg_res = dif_adf_GDP.regression
residuals=reg_res.resid
print(reg_res.summary().as_text())
plot_acf(residuals,lags=60)


# In[ ]:


# P-value is 0.000. So dif_GDP is stationary. 


# In[ ]:


dif2_=GDP.diff().diff()
dif2_GDP=dif2_GDP.dropna()
dif2_adf_GDP = ADF(dif2_GDP, trend='ct', max_lags=10) 
print(dif2_adf_GDP.summary().as_text())


# In[ ]:


# Augmented Dickey Fuller Test for GE
GE.plot()
adf_GE = ADF(GE, trend='c', max_lags=10) 
print(adf_GE.summary().as_text())
reg_res = adf_GE.regression
residuals=reg_res.resid
print(reg_res.summary().as_text())
plot_acf(residuals,lags=60)


# In[ ]:


# P-value is greater than 0.05, so GE is non-stationary.


# In[ ]:



dif_=GE.diff()
dif_GE=dif_GE.dropna()
dif_adf_GE = ADF(dif_GE, trend='c', max_lags=10) 
print(dif_adf_GE)
reg_res = dif_adf_GE.regression
residuals=reg_res.resid
print(reg_res.summary().as_text())
plot_acf(residuals,lags=60)


# In[ ]:


# P-value for dif_GE is 0.026, so dif_GE is stationary.


# In[ ]:


#KPSS test for GPDI
kpss_GPDI = KPSS(GPDI, trend='ct') 
print(kpss_GPDI.summary().as_text())


# In[ ]:


# P-value of KPSS test for GPDI is lower than 0.05. This suggests that GPDI is not trend stationary.


# In[ ]:


dif_kpss_GPDI=KPSS(dif_GPDI, trend='ct') 
print(dif_kpss_GPDI.summary().as_text())


# In[ ]:


# P-value of KPSS test for diff_kpss_GPDI is greater than 0.05. This suggests that dif_kpss_GPDI is trend stationary.


# In[ ]:



#KPSS test for GDP
kpss_GDP = KPSS(GDP, trend='ct') 
print(kpss_GDP.summary().as_text())


# In[ ]:


# P-value of KPSS test for GDP is lower than 0.05. This suggests that GDP is not trend stationary. 


# In[ ]:


dif_kpss_GDP=KPSS(dif_GDP, trend='ct') 
print(dif_kpss_GDP.summary().as_text())


# In[ ]:


# P-value of KPSS test for dif_GDP is greater than 0.05. This suggests that dif_GDP is trend stationary. 


# In[ ]:



#KPSS test for GE
kpss_GE = KPSS(GE, trend='ct') 
print(kpss_GE.summary().as_text())


# In[ ]:


# P-value of KPSS test for GE is lower than 0.05. This suggests that GE is not trend stationary. 


# In[ ]:


dif_kpss_GE=KPSS(dif_GE, trend='ct') 
print(dif_kpss_GE.summary().as_text())


# In[ ]:


# I could not debug this code so I will also use phillips-perron test.


# In[ ]:



#PHILLIPSPERRON test for GPDI
PhillipsPerron_GPDI = PhillipsPerron (GPDI, trend='ct') 
print(PhillipsPerron_GPDI.summary().as_text())


# In[ ]:


# P-value of Phillipis-Perron test for GPDI is greater than 0.05. This suggests that GDPI non-stationary.


# In[ ]:


dif_PhillipsPerron_GPDI= PhillipsPerron (dif_GPDI, trend='ct') 
print(dif_PhillipsPerron_GPDI.summary().as_text())


# In[ ]:


# P-value of Phillipis-Perron test for dif_GPDI is lower than 0.05. This suggests that dif_GPDI stationary.


# In[ ]:


#PHILLIPSPERRON test for GDP
PhillipsPerron_GDP = PhillipsPerron (GDP, trend='ct') 
print(PhillipsPerron_GDP.summary().as_text())


# In[ ]:


# P-value of Phillipis-Perron test for GDP is greater than 0.05. This suggests that GDP is non-stationary.


# In[ ]:


dif_PhillipsPerron_GDP= PhillipsPerron (dif_GDP, trend='ct') 
print(dif_PhillipsPerron_GDP.summary().as_text())


# In[ ]:


# P-value of Phillipis-Perron test for dif_GDP is lower than 0.05. This suggests that dif_GDP stationary.


# In[ ]:


#PHILLIPSPERRON test for GE
PhillipsPerron_GE = PhillipsPerron (GE, trend='c') 
print(PhillipsPerron_GE.summary().as_text())


# In[ ]:


# P-value of Phillipis-Perron test for GE is greater than 0.05. This suggests that GE non-stationary.


# In[ ]:


dif_PhillipsPerron_GE= PhillipsPerron (dif_GE, trend='c') 
print(dif_PhillipsPerron_GE.summary().as_text())


# In[ ]:


# P-value of Phillipis-Perron test for dif_GE is lower than 0.05. This suggests that dif_GE stationary.


# In[ ]:


# All the variables are non-stationary at levels but stationary at first differences. So we can proceed with the Johansen Cointegration Test.


# In[ ]:


import numpy as np
import statsmodels.tsa.stattools as ts 


# In[ ]:


from pathlib import Path


# In[ ]:


from statsmodels.tsa.api import VAR
from statsmodels.tsa.api import VECM
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import CointRankResults
from statsmodels.tsa.vector_ar.vecm import coint_johansen


# In[ ]:


vecmdata = df[['GPDI', 'GDP', 'GE']]
train_vecm= vecmdata.iloc[0:78]
test_vecm= vecmdata.iloc[-5:0]
vecmvalues=train_vecm.values


# In[ ]:



Test_Johansen=coint_johansen(vecmvalues,0,1)
trace_test=pd.DataFrame(Test_Johansen.lr1)
trace_test.columns=["trace test stat"]
cvt=pd.DataFrame(Test_Johansen.cvt)
cvt.columns=["0.1","0.05","0.01"]
Trace_test=pd.concat([trace_test,cvt],axis=1)
Trace_test


# In[ ]:


meigen_test=pd.DataFrame(Test_Johansen.lr2)
meigen_test.columns=["meigen test stat"]
cvm=pd.DataFrame(Test_Johansen.cvm)
cvm.columns=["0.1","0.05","0.01"]
Meigen_test=pd.concat([meigen_test,cvm],axis=1)
Meigen_test


# In[ ]:


# Both trace test statistics and max.eigen test statistics are greater than critical values, we reject H0.
# There is cointegration. VECM model is be the better option for the analysis. 


# In[ ]:


import pandas as pd
import statsmodels.api as sm


# In[ ]:


from statsmodels.tsa.api import VAR
from statsmodels.tsa.api import VECM
from statsmodels.tools.eval_measures import rmse, aic


# In[ ]:


model = VECM(train_vecm, k_ar_diff=2, coint_rank=1, deterministic='ci') 
vecm_res = model.fit()
vecm_res.summary()


# In[ ]:


vecm_res.plot_data(with_presample=True)
vecmalphabeta=vecm_res.gamma.round(4) 


# In[ ]:


## Forecast
predicted_values=pd.DataFrame(vecm_res.predict(steps=5))
predicted_values.columns=['GPDI', 'GDP', 'GE']


# In[ ]:


forecast, lower, upper = vecm_res.predict(5, 0.05)
print("lower bounds of confidence intervals:")
print(lower.round(3))
print("\npoint forecasts:")
print(forecast.round(1))
print("\nupper bounds of confidence intervals:")
print(upper.round(3))


# In[ ]:


vecm_res.plot_forecast(steps=10) #out of sample forecast


# In[ ]:


# Forecasts are shown in graphs above.


# In[ ]:


#impulse-response functions


# In[ ]:


irf = vecm_res.irf(50)
irf.plot(orth=True)


# In[ ]:


irf.plot(impulse='GPDI')
irf.plot(impulse='GDP') 
irf.plot(impulse='GE') 


# In[ ]:


irf = vecm_res.irf(80)
irf.plot(impulse='GPDI') 


# In[ ]:


# Plots above show how a shock in one variable affects other variables in subsequent periods.


# In[ ]:



## Calculation of the metrics
def forecast_accuracy(forecast, actual):
 mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
 mae = np.mean(np.abs(forecast - actual))    # MAE
 rmse = np.mean((forecast - actual)**2)**.5  # RMSE
 return({'mape':mape, 'mae': mae, 
             'rmse':rmse})


# In[ ]:


test_values=test_vecm.reset_index(drop=True)  ## in order to perform calculation, index of two DFs must match
metrics_for_GPDI=forecast_accuracy(predicted_values.GPDI, test_values.GPDI)
metrics_for_GDP=forecast_accuracy(predicted_values.GDP, test_values.GDP)
metrics_for_GE=forecast_accuracy(predicted_values.GE, test_values.GE)


# In[ ]:


metrics_for_GPDI


# In[ ]:


list(test_values.columns)


# In[ ]:


list(predicted_values.columns)

