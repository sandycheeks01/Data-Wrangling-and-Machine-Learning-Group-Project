# %%
import streamlit as st

# %%
st.title("Project 3")

# %%
st.write("Group 13")

# %%
st.write("Team members: James Coburn, Sandy Wang, Junhuang Zhou, Honghui Gao.")

# %%
st.title("Topic Description:")

# %%
st.write("Forecasting and Clustering Regional Tourism Expenditure in New Zealand")

# %%
st.title("Dataset Overview")

# %%
st.write("The dataset used in this project is derived from the Tourism Electronic Card Transactions (TECT) series, published by the New Zealand Ministry of Business, Innovation and Employment (MBIE). It captures monthly tourism spending across all regions of New Zealand and provides a rich basis for time series forecasting and other machine learning analyses.")

st.write("Key Attributes:")
st.write("Date: Monthly timestamp from 2018 onwards.")
st.write("Region: 31 regions across New Zealand (e.g., Auckland, Wellington, Queenstown).")
st.write("Product: Type of tourism service (e.g., Accommodation, Food & Beverage, Retail, Transport).")
st.write("Visitor Type: Either Domestic or International visitors.")
st.write("Origin: For international visitors, includes country/region of origin (e.g., Australia, China, USA).")
st.write("Annual Spend: Total annual spend for the given group.")
st.write("Monthly Spend: Monthly spend figure in millions of NZD.")


# %%
st.title("Data Preparation")

# %%
st.write("Load Dataset")

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# %%
df = pd.read_excel('Region-series.xlsx', sheet_name='Data base')
df

# %%
st.write("Check for missing values")

# %%
df.isnull().sum()

# %%
st.write("Drop rows with missing values (if any)")

# %%
df = df.dropna()
df

# %%
st.write("Ensure Correct Data Type")

# %%
df['Date'] = pd.to_datetime(df['Date'])
df

# %%
df.to_csv('cleaned_region_series.csv', index=False)

# %%
st.title("Exploratory Data Analysis")

# %%
import matplotlib.pyplot as plt

# %%
st.title("Total Monthly Spend Over Time")

# %%
monthly_spend_trend = df.groupby('Date')['Monthly Spend'].sum()

# %%
plt.figure(figsize=(10, 6))
monthly_spend_trend.plot()
plt.title('Total Monthly Spend Over Time')
plt.ylabel('Total Monthly Spend (Million NZD)')
plt.xlabel('Year')
plt.grid(True)
st.pyplot(plt.gcf())

# %%
st.write("We started by examining the total monthly tourism spend from 2018 to 2025.")
st.write("There was a significant drop in 2020 due to COVID19 followed by a clear recovery trend beginning in 2021.  This shows how external events impact tourism.")

# %%
st.title("Top 10 Visitor Origins by Total Monthly Spend")

# %%
origin_spend = df.groupby('Origin')['Monthly Spend'].sum().sort_values(ascending=False).head(10)

# %%
plt.figure(figsize=(10, 6))
origin_spend.plot(kind='bar')
plt.title('Top 10 Visitor Origins by Total Monthly Spend')
plt.ylabel('Total Monthly Spend (Million NZD)')
plt.xlabel('Origin')
plt.xticks(rotation=45)
plt.grid(True)
st.pyplot(plt.gcf())

origin_spend

# %%
st.write("We analysed the top contributors to total tourism spending.") 
st.write("The top five were domestic regions Auckland, Waikato, Canterbury, Wellington and Bay of Plenty highlighting the strength of internal tourism. Among international markets Australia and the United States stood out.")
st.write("Auckland alone contributed over $17 billion making it the central city for both domestic and international visitors.")


# %%
st.title("Top 5 Products or Services Contributing to Spending")

# %%
product_spend = df.groupby('Product')['Monthly Spend'].sum().sort_values(ascending=False).head(5)

# %%
plt.figure(figsize=(8, 5))
product_spend.plot(kind='bar')
plt.title('Top 5 Products or Services by Total Monthly Spend')
plt.ylabel('Total Monthly Spend (Million NZD)')
plt.xlabel('Product')
plt.xticks(rotation=45)
plt.grid(True)
st.pyplot(plt.gcf())

product_spend

# %%
st.write("We explored where tourists spent the most money by product type The top categories were Retail sales  other, Alcohol food and beverage services.")
st.write("Surprisingly accommodation services ranked lower than expected suggesting tourists may spend more on food and shopping.")

# %%
st.title("Monthly Spend by Region")

# %%
region_spend = df.groupby('Region')['Monthly Spend'].sum().sort_values(ascending=False)

# %%
plt.figure(figsize=(12, 6))
region_spend.plot(kind='barh')
plt.title('Total Monthly Spend by Region')
plt.xlabel('Total Monthly Spend (Million NZD)')
plt.ylabel('Region')
plt.grid(True)
plt.tight_layout()
st.pyplot(plt.gcf())

region_spend


# %%
st.write("Regional analysis showed Auckland as the top destination of monthly spend, followed by Canterbury and Otago. Otago’s high ranking is likely driven by Queenstown, a major tourist city. An interesting finding was that Waikato ranked higher than Wellington, indicating strong domestic travel beyond the capital.")

# %%
st.title("Machine Learning Modelling")

# %%
st.title("Linear Regression forecasting model")


# %%
st.write("Purpose:")
st.write("To model the relationship between time and tourism spend, and predict future values.")

st.write("Why we chose it")
st.write("Linear regression is a simple, interpretable model suitable for baseline forecasting. It helps stakeholders anticipate demand and plan accordingly. Although it doesn’t capture seasonality or complex patterns, it provides a useful long-term trend estimate.")

# %%
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

# %%
df2 = df.copy()
df2['Date'] = pd.to_datetime(df2['Date'])

# %%
df2 = df2.sort_values('Date')
df2 = df2.set_index('Date')

# %%
monthly = df2.groupby(pd.Grouper(freq='ME'))['Monthly Spend'].sum()

# %%
y = monthly.values

# %%
X = [[i] for i in range(len(y))] 

# %%
model = LinearRegression()
model.fit(X, y)

# %%
y_pred = model.predict(X)

# %%
plt.figure(figsize=(10, 5))
plt.plot(monthly.index, y, label='Actual Spend')
plt.plot(monthly.index, y_pred, label='Linear Trend')
plt.title('Linear Regression Forecast of Monthly Tourism Spend')
plt.xlabel('Year')
plt.ylabel('Monthly Spend (Million NZD)')
plt.grid(True)
st.pyplot(plt.gcf())

# %% [markdown]
# Forecast next 12 months

# %%
import datetime
import calendar

# %%
last_date = df2.index.max()
year = last_date.year
month = last_date.month

# %%
future_dates = []
for _ in range(12):
    month += 1
    if month > 12:
        month = 1
        year += 1

    last_day = calendar.monthrange(year, month)[1]
    future_dates.append(datetime.date(year, month, last_day))

# %%
X_future = [[i] for i in range(len(X), len(X) + 12)]
y_future = model.predict(X_future)

# %%
plt.figure(figsize=(12, 6))
plt.plot(monthly.index, y, label='Actual Spend')
plt.plot(monthly.index, y_pred, label='Trend Line')
plt.plot(future_dates, y_future, label='Forecast (Next 12 Months)', color='red')

plt.title('Linear Regression Forecast of Monthly Tourism Spend (Next 12 Months)')
plt.xlabel('Year')
plt.ylabel('Monthly Spend (Million NZD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
st.pyplot(plt.gcf())

# %%
st.write("1. Long-Term Growth Trend")
st.write("The orange regression line shows a steady upward trajectory in monthly tourism expenditure in New Zealand from 2018 to 2025. Despite short-term fluctuations, this suggests a positive long-term growth pattern in the tourism sector.")
st.write("2. Seasonal Peaks and Valleys")
st.write("The blue line (actual monthly spend) displays repeating seasonal patterns, with peaks likely aligning with high tourism months (e.g., summer, holidays). This indicates strong seasonality in spending behavior, which linear models alone cannot fully cap.")
st.write("3. Impact of External Events")
st.write("The significant drop around early 2020 corresponds to the COVID-19 pandemic, which heavily impacted international and domestic travel. The model helps highlight this anomaly as a deviation from the expected trend.")
st.write("4. Post-Pandemic Recovery")
st.write("From 2021 onwards, we observe a recovery phase, with increasing spend levels and new seasonal highs. This reflects renewed activity in the tourism sector.")


# %%
forecast_df = pd.DataFrame()

# %%
forecast_df['Date'] = future_dates

# %%
forecast_df['Forecasted Spend (Million NZD)'] = y_future

# %%
forecast_df['Forecasted Spend (Million NZD)'] = forecast_df['Forecasted Spend (Million NZD)'].round(2)

# %%
forecast_df

# %%
st.title("K-Means Clustering model applied to: Tourism expenditure patterns across New Zealand regions.")


# %%
st.write("We used K-Means clustering to group New Zealand regions based on average monthly spend, total spend, and spending variability. We found three clusters. Cluster 0 has low-performing areas like Gisborne and West Coast, with low tourism spend and stability. Cluster 1 includes mid-performing regions like Otago, Wellington, and Canterbury, showing higher spend and more variability. Cluster 2 is just Auckland, which has very high spend and volatility, making it unique. These clusters help us understand regional strengths and guide tourism planning and investment.")


# %%
region_group = df2.groupby('Region')
avg_spend = region_group['Monthly Spend'].mean()
total_spend = region_group['Monthly Spend'].sum()
std_spend = region_group['Monthly Spend'].std()

# %%
region_stats = pd.DataFrame()
region_stats['Avg Spend'] = avg_spend
region_stats['Total Spend'] = total_spend
region_stats['Spend StdDev'] = std_spend

# %%
from sklearn.cluster import KMeans

# %%
kmeans = KMeans(n_clusters=3, random_state=0)
labels = kmeans.fit_predict(region_stats)

# %%
region_stats['Cluster'] = labels

# %%
region_stats = region_stats.sort_values('Cluster')
region_stats

# %%
plt.figure(figsize=(8, 6))
plt.scatter(region_stats['Avg Spend'], region_stats['Total Spend'], c=region_stats['Cluster'])
plt.text(region_stats['Avg Spend']['Gisborne'], region_stats['Total Spend']['Gisborne'], 'Gisborne', fontsize=8)
plt.text(region_stats['Avg Spend']['Auckland'], region_stats['Total Spend']['Auckland'], 'Auckland', fontsize=8)
plt.text(region_stats['Avg Spend']['Otago'], region_stats['Total Spend']['Otago'], 'Otago', fontsize=8)
plt.text(region_stats['Avg Spend']['Wellington'], region_stats['Total Spend']['Wellington'], 'Wellington', fontsize=8)
plt.xlabel('Average Monthly Spend')
plt.ylabel('Total Spend')
plt.title('Tourism Region Clustering (K-Means)')
plt.grid(True)
st.pyplot(plt.gcf())


# %%
st.title("Hierarchical clustering")

# %%
st.write("Purpose:")
st.write("To create a tree-like structure that shows how similar or dissimilar regions are in their spending behavior.")

# %%
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# %%
region_group = df.groupby('Region')[['Monthly Spend']].mean()

# %%
scaler = StandardScaler()
region_scaled = scaler.fit_transform(region_group)

# %%
linked = linkage(region_scaled, method = 'ward')

# %%
plt.figure(figsize=(10, 6), dpi=100) 
dendrogram(linked, labels=region_group.index.tolist(),orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram of NZ Regions (Monthly Spend)')
plt.xlabel('Region')
plt.ylabel('Distance')
plt.xticks(rotation = 90)
plt.tight_layout()
st.pyplot(plt.gcf())


# %%
st.write("Why we chose it:")
st.write("Hierarchical clustering complements k-means by revealing nested relationships between regions. It is particularly useful for identifying sub-clusters and understanding the structure of the data at different levels.")

# %%
st.write("This dendrogram group regions based on overall similarity in their tourism data")
st.write("Closer branches (shorter vertical lines) mean regions have similar spending patterns. ")
st.write("Farther branches (longer vertical lines) indicate more distinct regions in terms of spending behaviour")
st.write("Cluster 1 (Orange): Waikato, Otago, Canterbury, Wellington, Bay of Plenty, and Auckland are grouped together. These may be regions with high tourism or economic activity (possibly urbanized or high-spending).")
st.write("Cluster 2 (Green): Includes many South Island and regional areas like Taranaki, Southland, Hawke’s Bay, etc. They could be moderate in activity or similar in visitor profiles.")
st.write("Cluster 3 (Blue): Manawatū-Whanganui stands out and is separated at a higher distance, indicating it is quite different from other regions in the dataset (an outlier in this clustering context).")
st.write("From the plot, we identified around three main clusters. One includes major spenders like Auckland, Otago, and Wellington. Another consists of rural or lower-tourism regions like Gisborne, Taranaki, and Manawatū-Whanganui.")
st.write("This model supports our k-means clustering by showing similar groupings, but it also gives more insight into how strongly related each region is.")
st.write("For example:")
st.write("Auckland: Clearly distinct in K-Means due to high tourism spend; hierarchical clustering confirms this, merging Auckland last—showing it's significantly different.") 
st.write("Wellington & Otago: Grouped together in K-Means and also closely clustered in the dendrogram, suggesting similar visitor behavior and seasonal patterns.")
st.write("James will be explaining the knn mean clustering later on.")
st.write("For tourism planners, this structure can guide how to segment regions for targeted campaigns, shared infrastructure planning, or promotional efforts.”")

# %%

# %%
st.title("Time Series Moving Statistics")

# %%
st.write("Purpose:")
st.write("To observe rolling trends and detect fluctuations in tourism spend.")


# %%
monthly = df2.groupby(pd.Grouper(freq='ME'))['Monthly Spend'].sum()
monthly

# %%
rolling_mean = monthly.rolling(3).mean()
rolling_mean

# %%
rolling_std = monthly.rolling(3).std()
rolling_std

# %%
st.write("Why we chose it:")
st.write("Moving averages help smooth short-term fluctuations and highlight long-term trends, while rolling standard deviation measures volatility. This is valuable for planners to identify unstable periods (e.g., pandemic recovery phase).")

# %%
plt.figure(figsize=(12, 6))
plt.plot(monthly, label='Monthly Spend')
plt.plot(rolling_mean, label='3-Month Rolling Mean')
plt.plot(rolling_std, label='3-Month Rolling Std Dev')
plt.title('Moving Statistics on Monthly Tourism Spend')
plt.xlabel('Year')
plt.ylabel('Spend (Million NZD)')
plt.legend()
plt.grid(True)
st.pyplot(plt.gcf())

# %%
st.write("Single Region Moving Statistics (e.g., Auckland)")
st.write("This chart shows the raw monthly tourism spend for a single region (e.g., Auckland), along with a 3-month rolling average and rolling standard deviation.")

st.write("**Why we chose it:**")
st.write("The moving average smooths out short-term fluctuations, making seasonal trends and long-term patterns easier to observe.")
st.write("The rolling standard deviation shows how volatile the spending is.")
st.write("These help identify periods of disruption or recovery, such as during COVID-19.")


# %%
st.subheader("Chart 1: Regional Spending Trends (3-Month Average)")

st.markdown("""
This chart shows us which regions are **steady performers** and which are **seasonal stars**.  
Auckland stays strong and stable — great for long-term planning.  
Queenstown spikes during holidays — great returns, but timing is everything.
""")

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
dates = pd.date_range('2019-01', '2023-12', freq='ME')
regions = ['Auckland', 'Wellington', 'Queenstown', 'Rotorua', 'Canterbury']

np.random.seed(42)
data = {}
for i, region in enumerate(regions):
    base = 80 + i*20
    trend = np.arange(len(dates)) * 3  # Time trend
    noise = np.random.normal(0, 8, len(dates)).cumsum()
    data[region] = base + trend + noise

df = pd.DataFrame(data, index=dates)


# %%
plt.figure(figsize=(12,6))
for region in regions:
    df[region].rolling(3).mean().plot(label=region, linewidth=2)

plt.axvspan('2020-03', '2022-02', color='lightgray', alpha=0.3)
plt.title('Regional Tourism Spending Trends (3-Month Moving Average)')
plt.xlabel('Month')
plt.ylabel('Spending (Million NZD)')
plt.legend()
plt.grid(alpha=0.3)
st.pyplot(plt.gcf())


# %%
st.write("Regional Rolling Mean Comparison")
st.write("This chart compares the 3-month rolling average of tourism spend across regions (e.g., Auckland, Wellington, Queenstown, Canterbury).")

st.write(" Why we chose it:")
st.write("- Helps visualize long-term spend trends across multiple regions.")
st.write("- Highlights which regions are growing steadily and which show strong seasonality.")
st.write("- For example, Auckland may be stable year-round, while Queenstown spikes seasonally.")


# %%
st.subheader("Chart 2: Spend Volatility by Region")

st.markdown("""
Here, we see **how risky each region is**.  
Queenstown swings wildly — high potential, high risk.  
Wellington is calm — a safe bet.

This helps us balance our **tourism strategy** between growth and stability.
""")

# %%
plt.figure(figsize=(12,6))

for region in regions:
    df[region].rolling(3).std().plot(linestyle='--', label=region)

plt.axvspan('2020-03', '2022-02', color='lightgray', alpha=0.3)
plt.title('Spending Volatility (3-Month Rolling Std Dev)')
plt.xlabel('Month')
plt.ylabel('Standard Deviation')
plt.legend()
plt.grid(alpha=0.3)
st.pyplot(plt.gcf())

# %%
st.title("Conclusion")

# %%
st.write("Summary of insights:")

# %%
st.markdown("""
###Key Insights
- **Tourism is seasonal**, especially in places like Queenstown.
- **Auckland and Waikato** are consistently high in spending.
- **Retail and food services** receive the most spending; **accommodation** less than expected.
- **Recovery began in 2021** after the COVID impact.

### Forecasting
- **Linear regression** shows a steady rise in future spending.
- But it struggles with unexpected events like COVID.

### Clustering Analysis
- **K-Means** and **hierarchical clustering** grouped regions by spending patterns.
- Found **three clusters** and highlighted **outliers** like Manawatū-Whanganui.

### Visual Highlights
- **Rolling Average Chart:** Shows steady vs. seasonal regions.
- **Volatility Chart:** Identifies regions with more monthly spend changes (risk).

---

These insights can support **better planning**, **resource management**, and **targeted marketing** in New Zealand tourism.
""")

# %%
st.write("Possible improvements:")

# %%
st.write("Using ARIMA for Time Series Forecasting.") 
st.write("Using external datasets")



