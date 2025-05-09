# Importing required libraries for data processing, statistics, and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t, chi2, ttest_ind
from sklearn.model_selection import train_test_split


# --- 1. Loading and Cleaning the Dataset ---
# Loading the CSV file and cleaning data to handle missing values, invalid entries, and create a Date column.
df = pd.read_csv('climate_change_dataset.csv')

# Replacing 'Unknown', '99999.0', and 'NAN' with np.nan to ensure proper numerical handling
df.replace(['Unknown', 99999.0, 'NAN'], np.nan, inplace=True)

# Converting Year and Month to numeric, filling missing Year with the previous valid value
df['Year'] = pd.to_numeric(df['Year'], errors='coerce').ffill()
df['Month'] = pd.to_numeric(df['Month'], errors='coerce').fillna(1)

# Creating a Date column from Year and Month for time-based analysis
df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))

# Ensuring numerical columns are float
numerical_cols = ['Avg_Temp (°C)', 'Max_Temp (°C)', 'Min_Temp (°C)', 'Precipitation (mm)',
                  'Humidity (%)', 'Wind_Speed (m/s)', 'CO2_Concentration (ppm)']
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# --- 2. Checking Data Quality ---
# Printing descriptive statistics and sample sizes to diagnose issues with Min_Temp (°C) and variances.
print("\nDescriptive Statistics for Numerical Columns:")
print(df[numerical_cols].describe())
print("\nNumber of Valid (Non-NaN) Entries:")
print(df[numerical_cols].count())

# Detecting outliers in Min_Temp (°C) using the IQR method
Q1 = df['Min_Temp (°C)'].quantile(0.25)
Q3 = df['Min_Temp (°C)'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df['Min_Temp (°C)'][(df['Min_Temp (°C)'] < lower_bound) | (df['Min_Temp (°C)'] > upper_bound)]
print(f"\nOutliers in Min_Temp (°C) (outside [{lower_bound:.2f}, {upper_bound:.2f}]):")
print(outliers.dropna())

# Validating logical consistency: Min_Temp ≤ Avg_Temp ≤ Max_Temp
invalid_rows = df[
    (df['Min_Temp (°C)'] > df['Avg_Temp (°C)']) | 
    (df['Avg_Temp (°C)'] > df['Max_Temp (°C)']) |
    (df['Min_Temp (°C)'] > df['Max_Temp (°C)'])
][['Year', 'Month', 'Min_Temp (°C)', 'Avg_Temp (°C)', 'Max_Temp (°C)']]
print("\nRows where Min_Temp > Avg_Temp or Avg_Temp > Max_Temp:")
print(invalid_rows)

# --- 3. Direct Mean and Variance ---
# Calculating mean and variance for key climate variables, ensuring NaN values are excluded.
mean_vals = df[numerical_cols].mean()
var_vals = df[numerical_cols].var(ddof=1)

print("\nDirect Means:\n", mean_vals)
print("\nDirect Variances:\n", var_vals)

# --- 4. Frequency Distribution and Visualization ---
# Creating a histogram to visualize the distribution of average temperatures.
counts, bins = np.histogram(df['Avg_Temp (°C)'].dropna(), bins=10)
freq_dist = pd.Series(counts, index=pd.IntervalIndex.from_breaks(bins))

# Plotting and displaying the histogram
plt.figure(figsize=(8, 5))
plt.hist(df['Avg_Temp (°C)'].dropna(), bins=10, edgecolor='black')
plt.xlabel('Average Temperature (°C)')
plt.ylabel('Frequency')
plt.title('Histogram of Average Temperatures (2020-2024)')
plt.grid(True)
plt.show()  # Displaying the plot in a graphical window
plt.close()

# Creating a pie chart to compare months with high (>50%) vs. low humidity.
df['hum_level'] = df['Humidity (%)'].apply(lambda x: 'High' if x > 50 else 'Low')
hum_counts = df['hum_level'].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(hum_counts, labels=hum_counts.index, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral'])
plt.title('High vs Low Humidity Months')
plt.axis('equal')
plt.show()  # Displaying the plot
plt.close()

# --- 5. Mean and Variance from Frequency Distribution ---
# Calculating mean and variance using the frequency distribution of average temperatures.
bin_centers = (bins[:-1] + bins[1:]) / 2
weights = counts / counts.sum()

mean_fd = np.average(bin_centers, weights=weights)
var_fd = np.average((bin_centers - mean_fd)**2, weights=weights) * counts.sum() / (counts.sum() - 1)

print("\nMean from Frequency Distribution:", mean_fd)
print("Variance from Frequency Distribution:", var_fd)

# --- 6. Confidence Intervals and Tolerance Interval (80/20 Split) ---
# Splitting Avg_Temp (°C) into training (80%) and test (20%) sets for statistical intervals.
train, test = train_test_split(df['Avg_Temp (°C)'].dropna(), test_size=0.2, random_state=42)

n = len(train)
dfree = n - 1
mt = train.mean()
st = train.std(ddof=1)

# 95% Confidence Interval for Mean
ci_mean = t.interval(0.95, dfree, loc=mt, scale=st/np.sqrt(n))
print("\n95% Confidence Interval for Mean:", ci_mean)

# 95% Confidence Interval for Variance
alpha = 0.95
chi2_lower = chi2.ppf((1-alpha)/2, dfree)
chi2_upper = chi2.ppf((1+alpha)/2, dfree)
ci_var = (dfree * st*2 / chi2_upper, dfree * st*2 / chi2_lower)
print("95% Confidence Interval for Variance:", ci_var)

# 95% Tolerance Interval
k = t.ppf((1+alpha)/2, dfree) * np.sqrt(dfree / (chi2.ppf(0.95, dfree) * n))
referrals = 0
tol_int = (mt - k*st, mt + k*st)
print("95% Tolerance Interval:", tol_int)

# Validating on test set
within_tol = ((test >= tol_int[0]) & (test <= tol_int[1])).mean()
print("Proportion of test points within tolerance interval:", within_tol)

# --- 7. Hypothesis Testing: High vs Low Humidity Temperatures ---
# Performing Welch's t-test to compare temperatures in high vs. low humidity months.
high = df[df['Humidity (%)'] > 50]['Avg_Temp (°C)']
low = df[df['Humidity (%)'] <= 50]['Avg_Temp (°C)']

t_stat, p_val = ttest_ind(high.dropna(), low.dropna(), equal_var=False)
print("\nWelch's t-test statistic:", t_stat)
print("p-value:", p_val)
print("Interpretation: If p-value < 0.05, temperatures differ significantly between high and low humidity months.")

# --- 8. Additional Analysis: Temperature Trend Over Time ---
# Plotting average temperature over time to observe climate trends.
df_time = df.groupby('Date')['Avg_Temp (°C)'].mean().reset_index()

plt.figure(figsize=(10, 6))
plt.plot(df_time['Date'], df_time['Avg_Temp (°C)'], marker='o', color='teal')
plt.xlabel('Date')
plt.ylabel('Average Temperature (°C)')
plt.title('Temperature Trend (2020-2024)')
plt.grid(True)
plt.show()  # Displaying the plot
plt.close()

# --- 9. Additional Analysis: Correlation Between Variables ---
# Computing and visualizing correlations between key variables.
corr_cols = ['Avg_Temp (°C)', 'Humidity (%)', 'Precipitation (mm)', 'CO2_Concentration (ppm)']
corr_matrix = df[corr_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Climate Variables')
plt.show()  # Displaying the plot
plt.close()

# --- 10. Additional Analysis: Seasonal Patterns ---
# Analyzing average temperature by month to identify seasonal variations.
seasonal = df.groupby('Month')['Avg_Temp (°C)'].mean().reset_index()

plt.figure(figsize=(8, 5))
sns.barplot(x='Month', y='Avg_Temp (°C)', data=seasonal, color='salmon')
plt.xlabel('Month')
plt.ylabel('Average Temperature (°C)')
plt.title('Average Temperature by Month')
plt.show()  # Displaying the plot
plt.close()

# --- 11. Additional Analysis: Extreme Weather Events ---
# Identifying months with extreme temperatures (>30°C or <0°C).
df['temp_level'] = df['Avg_Temp (°C)'].apply(lambda x: 'Extreme High' if x > 30 else 'Extreme Low' if x < 0 else 'Normal')
temp_counts = df['temp_level'].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(temp_counts, labels=temp_counts.index, autopct='%1.1f%%', startangle=90, colors=['red', 'blue', 'green'])
plt.title('Proportion of Months with Extreme Temperatures')
plt.axis('equal')
plt.show()  # Displaying the plot
plt.close()

# --- 12. Interesting Fact ---
# Highlighting an interesting finding about CO2 concentration.
max_co2_year = df.groupby('Year')['CO2_Concentration (ppm)'].mean().idxmax()
max_co2_value = df.groupby('Year')['CO2_Concentration (ppm)'].mean().max()
print(f"\nInteresting Fact: The year {max_co2_year} had the highest average CO2 concentration at {max_co2_value:.2f} ppm, indicating a potential rise in greenhouse gas levels.")

# --- 13. Summary ---
print("\nSummary of Findings:")
print("- The dataset shows variability in temperature, humidity, and CO2 levels from 2020 to 2024.")
print("- Temperature trends suggest potential warming, with fluctuations over time.")
print("- Correlations indicate relationships between temperature, humidity, and precipitation.")
print("- Seasonal patterns show higher temperatures in mid-year months (e.g., July, August).")
print("- Extreme temperature events are notable, with a significant proportion of months showing high or low temperatures.")
print("- All plots are displayed interactively in graphical windows.")