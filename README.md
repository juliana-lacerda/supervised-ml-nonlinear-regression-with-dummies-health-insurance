# MULTIPLE NON-LINEAR REGRESSION WITH DUMMIES

In this work, an OLS model is built in order to fit data composed of health insurance information. 

Several statistical tests are perfomed, such as Shapiro-Francia test for residual normality check, Breusch-Pagan test to detect heteroscedasticity on the residuals and a multicollinearity test (VIF and Tolerance) is performed on the explanatory variables. 

The Box-Cox transformation is also performed on the target variable in order to make its distribution more like the normal one. 

The categorical variable plan is transformed by the use of dummies.

The data is composed of the following columns:
* med_exp (float): average medical expenses per month
* age (int): age in years
* cron_d (int): quantity of cronical deseases
* income (float): average yearly income in thousands of dollars
* plan (str): type of health insurance plan

target $\rightarrow$ med_exp
