#!/usr/bin/env python
# coding: utf-8

# In[94]:


pip install arch


# In[180]:


import warnings
import pandas as pd
from pathlib import Path
import statsmodels.api as sm
import  matplotlib.pylab as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
plt.style.use('fivethirtyeight')


# In[181]:


import arch
from arch.unitroot import ADF
from arch.unitroot import DFGLS
from arch.unitroot import PhillipsPerron
from arch.unitroot import KPSS


# In[182]:


from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss


# In[183]:


warnings.filterwarnings("ignore")
fred= pd.read_excel(r"C:\Users\07ser\OneDrive\Masaüstü\fred2.xlsx")
df= fred.set_index('DATE')
df.index


# In[184]:


GPDI= df["GPDI"]
GDP= df["GDP"]
GE=df["GE"]
GPDI.plot()


# In[185]:


# Augmented Dickey Fuller Test for GPDI
adf_GPDI= ADF(GPDI, trend='ct', max_lags=20)
print(adf_GPDI.summary().as_text())
reg_res = adf_GPDI.regression
residuals=reg_res.resid
print(reg_res.summary().as_text())
plot_acf(residuals,lags=60)


# In[186]:


# p-value obtained is 0.899 which is bigger than 0.05. This suggests that GPDI is non-stationary.


# In[187]:


dif_GPDI=GPDI.diff()
dif_GPDI=dif_GPDI.dropna()
dif_adf_GPDI = ADF(dif_GPDI, trend='ct', max_lags=10) 
print(dif_adf_GPDI.summary().as_text())
reg_res = dif_adf_GPDI.regression
residuals=reg_res.resid
print(reg_res.summary().as_text())
plot_acf(residuals,lags=60)


# In[188]:


#p-value obtained is 0.000 which is smaller than 0.05. This suggests that dif_GPDI is stationary.


# In[189]:


#Augmented Dickey Fuller Test for GDP
GDP.plot()
adf_GDP = ADF(GDP, trend='ct', max_lags=10) 
print(adf_GDP.summary().as_text())
reg_res = adf_GDP.regression
residuals=reg_res.resid
print(reg_res.summary().as_text())
plot_acf(residuals,lags=60)


# In[190]:


#p-value is 0.990. So GDP is non-stationary


# In[191]:


dif_GDP=GDP.diff()
dif_GDP=dif_GDP.dropna()
dif_adf_GDP = ADF(dif_GDP, trend='ct', max_lags=10) 
print(dif_adf_GDP.summary().as_text())
reg_res = dif_adf_GDP.regression
residuals=reg_res.resid
print(reg_res.summary().as_text())
plot_acf(residuals,lags=60)


# In[192]:


# P-value is 0.000. So dif_GDP is stationary. 


# In[193]:


dif2_=GDP.diff().diff()
dif2_GDP=dif2_GDP.dropna()
dif2_adf_GDP = ADF(dif2_GDP, trend='ct', max_lags=10) 
print(dif2_adf_GDP.summary().as_text())


# In[194]:


# Augmented Dickey Fuller Test for GE
GE.plot()
adf_GE = ADF(GE, trend='c', max_lags=10) 
print(adf_GE.summary().as_text())
reg_res = adf_GE.regression
residuals=reg_res.resid
print(reg_res.summary().as_text())
plot_acf(residuals,lags=60)


# In[195]:


# P-value is greater than 0.05, so GE is non-stationary.


# In[196]:



dif_=GE.diff()
dif_GE=dif_GE.dropna()
dif_adf_GE = ADF(dif_GE, trend='c', max_lags=10) 
print(dif_adf_GE)
reg_res = dif_adf_GE.regression
residuals=reg_res.resid
print(reg_res.summary().as_text())
plot_acf(residuals,lags=60)


# In[197]:


# P-value for dif_GE is 0.026, so dif_GE is stationary.


# In[198]:


#KPSS test for GPDI
kpss_GPDI = KPSS(GPDI, trend='ct') 
print(kpss_GPDI.summary().as_text())


# In[199]:


# P-value of KPSS test for GPDI is lower than 0.05. This suggests that GPDI is not trend stationary.


# In[200]:


dif_kpss_GPDI=KPSS(dif_GPDI, trend='ct') 
print(dif_kpss_GPDI.summary().as_text())


# In[201]:


# P-value of KPSS test for diff_kpss_GPDI is greater than 0.05. This suggests that dif_kpss_GPDI is trend stationary.


# In[202]:



#KPSS test for GDP
kpss_GDP = KPSS(GDP, trend='ct') 
print(kpss_GDP.summary().as_text())


# In[203]:


# P-value of KPSS test for GDP is lower than 0.05. This suggests that GDP is not trend stationary. 


# In[204]:


dif_kpss_GDP=KPSS(dif_GDP, trend='ct') 
print(dif_kpss_GDP.summary().as_text())


# In[205]:


# P-value of KPSS test for dif_GDP is greater than 0.05. This suggests that dif_GDP is trend stationary. 


# In[206]:



#KPSS test for GE
kpss_GE = KPSS(GE, trend='ct') 
print(kpss_GE.summary().as_text())


# In[207]:


# P-value of KPSS test for GE is lower than 0.05. This suggests that GE is not trend stationary. 


# In[254]:


dif_kpss_GE=KPSS(dif_GE, trend='ct') 
print(dif_kpss_GE.summary().as_text())


# In[268]:


# I could not debug this code so I will also use phillips-perron test.


# In[210]:



#PHILLIPSPERRON test for GPDI
PhillipsPerron_GPDI = PhillipsPerron (GPDI, trend='ct') 
print(PhillipsPerron_GPDI.summary().as_text())


# In[211]:


# P-value of Phillipis-Perron test for GPDI is greater than 0.05. This suggests that GDPI non-stationary.


# In[212]:


dif_PhillipsPerron_GPDI= PhillipsPerron (dif_GPDI, trend='ct') 
print(dif_PhillipsPerron_GPDI.summary().as_text())


# In[213]:


# P-value of Phillipis-Perron test for dif_GPDI is lower than 0.05. This suggests that dif_GPDI stationary.


# In[214]:


#PHILLIPSPERRON test for GDP
PhillipsPerron_GDP = PhillipsPerron (GDP, trend='ct') 
print(PhillipsPerron_GDP.summary().as_text())


# In[215]:


# P-value of Phillipis-Perron test for GDP is greater than 0.05. This suggests that GDP is non-stationary.


# In[216]:


dif_PhillipsPerron_GDP= PhillipsPerron (dif_GDP, trend='ct') 
print(dif_PhillipsPerron_GDP.summary().as_text())


# In[217]:


# P-value of Phillipis-Perron test for dif_GDP is lower than 0.05. This suggests that dif_GDP stationary.


# In[218]:


#PHILLIPSPERRON test for GE
PhillipsPerron_GE = PhillipsPerron (GE, trend='c') 
print(PhillipsPerron_GE.summary().as_text())


# In[219]:


# P-value of Phillipis-Perron test for GE is greater than 0.05. This suggests that GE non-stationary.


# In[220]:


dif_PhillipsPerron_GE= PhillipsPerron (dif_GE, trend='c') 
print(dif_PhillipsPerron_GE.summary().as_text())


# In[221]:


# P-value of Phillipis-Perron test for dif_GE is lower than 0.05. This suggests that dif_GE stationary.


# In[222]:


# All the variables are non-stationary at levels but stationary at first differences. So we can proceed with the Johansen Cointegration Test.


# In[223]:


import numpy as np
import statsmodels.tsa.stattools as ts 


# In[224]:


from pathlib import Path


# In[225]:


from statsmodels.tsa.api import VAR
from statsmodels.tsa.api import VECM
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import CointRankResults
from statsmodels.tsa.vector_ar.vecm import coint_johansen


# In[226]:


vecmdata = df[['GPDI', 'GDP', 'GE']]
train_vecm= vecmdata.iloc[0:78]
test_vecm= vecmdata.iloc[-5:0]
vecmvalues=train_vecm.values


# In[233]:



Test_Johansen=coint_johansen(vecmvalues,0,1)
trace_test=pd.DataFrame(Test_Johansen.lr1)
trace_test.columns=["trace test stat"]
cvt=pd.DataFrame(Test_Johansen.cvt)
cvt.columns=["0.1","0.05","0.01"]
Trace_test=pd.concat([trace_test,cvt],axis=1)
Trace_test


# In[232]:


meigen_test=pd.DataFrame(Test_Johansen.lr2)
meigen_test.columns=["meigen test stat"]
cvm=pd.DataFrame(Test_Johansen.cvm)
cvm.columns=["0.1","0.05","0.01"]
Meigen_test=pd.concat([meigen_test,cvm],axis=1)
Meigen_test


# In[55]:


# Both trace test statistics and max.eigen test statistics are greater than critical values, we reject H0.
# There is cointegration. VECM model is be the better option for the analysis. 


# In[56]:


import pandas as pd
import statsmodels.api as sm


# In[57]:


from statsmodels.tsa.api import VAR
from statsmodels.tsa.api import VECM
from statsmodels.tools.eval_measures import rmse, aic


# In[234]:


model = VECM(train_vecm, k_ar_diff=2, coint_rank=1, deterministic='ci') 
vecm_res = model.fit()
vecm_res.summary()


# In[267]:


vecm_res.plot_data(with_presample=True)
vecmalphabeta=vecm_res.gamma.round(4) 


# In[236]:


## Forecast
predicted_values=pd.DataFrame(vecm_res.predict(steps=5))
predicted_values.columns=['GPDI', 'GDP', 'GE']


# In[237]:


forecast, lower, upper = vecm_res.predict(5, 0.05)
print("lower bounds of confidence intervals:")
print(lower.round(3))
print("\npoint forecasts:")
print(forecast.round(1))
print("\nupper bounds of confidence intervals:")
print(upper.round(3))


# In[238]:


vecm_res.plot_forecast(steps=10) #out of sample forecast


# In[273]:


# Forecasts are shown in graphs above.


# In[239]:


#impulse-response functions


# In[240]:


irf = vecm_res.irf(50)
irf.plot(orth=True)


# In[248]:


irf.plot(impulse='GPDI')
irf.plot(impulse='GDP') 
irf.plot(impulse='GE') 


# In[251]:


irf = vecm_res.irf(80)
irf.plot(impulse='GPDI') 


# In[ ]:


# Plots above show how a shock in one variable affects other variables in subsequent periods.


# In[264]:



## Calculation of the metrics
def forecast_accuracy(forecast, actual):
 mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
 mae = np.mean(np.abs(forecast - actual))    # MAE
 rmse = np.mean((forecast - actual)**2)**.5  # RMSE
 return({'mape':mape, 'mae': mae, 
             'rmse':rmse})


# In[265]:


test_values=test_vecm.reset_index(drop=True)  ## in order to perform calculation, index of two DFs must match
metrics_for_GPDI=forecast_accuracy(predicted_values.GPDI, test_values.GPDI)
metrics_for_GDP=forecast_accuracy(predicted_values.GDP, test_values.GDP)
metrics_for_GE=forecast_accuracy(predicted_values.GE, test_values.GE)


# In[266]:


metrics_for_GPDI


# In[270]:


list(test_values.columns)


# In[271]:


list(predicted_values.columns)


# In[275]:


# I cannot get the values of mape,mae etc. so I could not interpret them.


# In[ ]:




