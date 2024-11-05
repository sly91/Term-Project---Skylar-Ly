#!/usr/bin/env python
# coding: utf-8

# In[182]:


#Import libraries needed for analysis



import yfinance as yf
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


# In[183]:


# High-rated ESG and Low-rated ESG tickers
high_esg_tickers = ["ADBE", "NKE", "ACN", "CSCO", "PXT.TO"]  
low_esg_tickers = ["VEEV", "CRAI", "ZM", "SGY.TO", "STRA"]    



# In[184]:


#Time period
start_date = "2022-01-01"
end_date = "2024-01-01"


# In[185]:


# Downloading Daily Adjusted Closed Prices for High-rated ESG stocks and Low-rated ESG stocks
high_esg_data = yf.download(high_esg_tickers, start=start_date, end=end_date)["Adj Close"]
low_esg_data = yf.download(low_esg_tickers, start=start_date, end=end_date)["Adj Close"]

print("High ESG Data")
print(high_esg_data.head())

print("\nLow ESG Data")
print(low_esg_data.head())


# In[186]:


# Calculating daily returns for High-rated and Low-rated ESG portfolios
high_esg_returns = high_esg_data.pct_change().dropna()
low_esg_returns = low_esg_data.pct_change().dropna()

print("High ESG Daily Returns")
print(high_esg_returns.head())  # Displays the first 5 rows

print("\nLow ESG Daily Returns")
print(low_esg_returns.head())  # Displays the first 5 rows


# In[187]:


# Calculating the average daily returns for the High-rated and Low-rated portfolios
high_esg_portfolio_returns = high_esg_returns.mean(axis=1)
low_esg_portfolio_returns = low_esg_returns.mean(axis=1)

print("Average Daily Returns for High ESG Portfolio:\n", high_esg_portfolio_returns)
print("\nAverage Daily Returns for Low ESG Portfolio:\n", low_esg_portfolio_returns)


# In[188]:


# Calculating cumulative returns for High-rated and Low-rated portfolio
high_cum_returns = (1 + high_esg_portfolio_returns).cumprod()
low_cum_returns = (1 + low_esg_portfolio_returns).cumprod()

print("Cumulative Returns for High ESG Portfolio:\n", high_cum_returns)
print("\nCumulative Returns for Low ESG Portfolio:\n", low_cum_returns)


# In[189]:


# Plot cumulative returns
plt.figure(figsize=(10, 6))
plt.plot(high_cum_returns, label="High-rated ESG Portfolio", color='green')
plt.plot(low_cum_returns, label="Low-rated ESG Portfolio", color='red')
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.title("Cumulative Returns of High-rated vs. Low-rated ESG Portfolios")
plt.legend()
plt.show()


# In[190]:


# Regression Analysis: Testing ESG Ratings as Predictors of Performance
#  independent variable (time)
dates = np.arange(len(high_esg_portfolio_returns))


# In[191]:


# Add constant term for regression model
X = sm.add_constant(dates)


# In[192]:


# Run regressions on high-rated and low-rated ESG portfolios
model_high = sm.OLS(high_esg_portfolio_returns, X).fit()
model_low = sm.OLS(low_esg_portfolio_returns, X).fit()


# In[193]:


# Print regression models
print("High ESG Portfolio Regression Results:\n", model_high.summary())
print("\nLow ESG Portfolio Regression Results:\n", model_low.summary())



