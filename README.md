This is an introduction to my term project. My topic was to determine if high-rated ESG scores correlate with high stock performance over a period of time. To assess this, the python file contained in this repository is my data analysis on high-rated and low-rated ESG portfolios and how they compare against each other. I have also conducted a regression analysis on the two portfolios to determine if there were any significant relationships between portfolio performance and time regarding the selection of high-rated and low-rated ESG stocks.

Importing Libraries: The code imports necessary libraries needed for analysis (e.g., yfinance to retrieve data for high-rated and low-rated ESG stocks, numpy for numerical operations, statsmodels for regression analysis, and matplotlib for plotting).

Data Download: Historical price data for both high-rated and low-rated ESG stocks is downloaded from Yahoo Finance within the code. No specific data is stored in this repository, as all metrics are obtained directly by downloading data through yfinance each time the code runs.

Daily and Cumulative Return Calculation: The code calculates daily returns for each stock and averages them to produce portfolio-level daily returns. It also calculates cumulative returns to show the growth of an initial investment over time.

Cumulative Return Plotting: The code generates a cumulative return plot comparing the high-rated and low-rated ESG portfolios over the specified period. This graph visually represents the difference in returns between the two portfolios, helping determine if the high-rated ESG portfolio demonstrates better or more stable performance. I have included a screenshot of this figure in the repository.

Regression Analysis: A regression analysis is conducted on the daily returns for each portfolio to examine any trends over time. This analysis helps determine if time significantly impacts portfolio performance, potentially revealing long-term performance trends for each ESG group. I have included screenshots of these results in this repository.

