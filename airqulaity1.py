import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\91620\Downloads\Airquality.csv")

print(df.head())

print("\nMissing values in each column:")
print(df.isnull().sum())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Fill missing numeric values
df["pollutant_min"] = df["pollutant_min"].fillna(df["pollutant_min"].mean())
df["pollutant_max"] = df["pollutant_max"].fillna(df["pollutant_max"].mean())
df["pollutant_avg"] = df["pollutant_avg"].fillna(df["pollutant_avg"].mean())

df["pollutant_range"] = df["pollutant_max"] - df["pollutant_min"]

# 1. Bar plot - Average pollution per city
city_avg = df.groupby("city")["pollutant_avg"].mean().reset_index()
# TOP 20 cities
top_cities = city_avg.sort_values(by="pollutant_avg", ascending=False).head(20)

plt.figure(figsize=(14, 6))
sns.barplot(data=top_cities, x="city", y="pollutant_avg", hue="city", palette="viridis", legend=False)
plt.title("Top 20 Cities by Average Pollution Level")
plt.xlabel("City")
plt.ylabel("Average Pollutant Level")
plt.xticks(rotation=75)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 2. Histogram 
plt.figure(figsize=(10, 6))
plt.hist(df["pollutant_avg"], bins=15, color="orange", edgecolor="black")
plt.title("Distribution of Average Pollutants")
plt.xlabel("Pollutant Average")
plt.ylabel("Frequency")
plt.grid()
plt.show()

# 3. Boxplot 
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[["pollutant_min", "pollutant_max", "pollutant_avg"]], palette="Set2")
plt.title("Boxplot of Pollutant Levels")
plt.grid()
plt.show()

# 4. Scatter plot 
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x="latitude", y="pollutant_avg", hue="country", palette="tab10")
plt.title("Pollutant Average by Latitude")
plt.xlabel("Latitude")
plt.ylabel("Pollutant Avg")
plt.grid()
plt.show()

# Correlation matrix
plt.figure(figsize=(9, 7))
sns.heatmap(df[["pollutant_min", "pollutant_max", "pollutant_avg", "pollutant_range"]].corr(), annot=True, cmap="magma", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Outlier 
plt.figure(figsize=(10, 5))
sns.boxplot(x=df["pollutant_avg"], color="lightgreen")
plt.title("Outlier Detection in Pollutant Avg")
plt.grid()
plt.show()
