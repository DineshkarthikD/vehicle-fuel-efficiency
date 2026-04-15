import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("C:/Users/Lenovo/Downloads/vehicles.csv")

df = df[['make','model','year','fuelType','displ','cylinders','city08','highway08','comb08']]

df.columns = ['Make','Model','Year','Fuel','EngineSize','Cylinders','CityMPG','HighwayMPG','CombinedMPG']

df.dropna(inplace=True)

df.drop_duplicates(inplace=True)

df['EfficiencyScore'] = (df['CityMPG'] + df['HighwayMPG']) / 2

def mpg_category(mpg):
    if mpg < 20:
        return "Low"
    elif mpg < 35:
        return "Medium"
    else:
        return "High"

df['MPG_Category'] = df['CombinedMPG'].apply(mpg_category)


df['EngineCategory'] = pd.cut(df['EngineSize'],
                             bins=[0,2,4,10],
                             labels=['Small','Medium','Large'])

print(df.head())

print("\nSummary Statistics:\n", df.describe())

print("\nFuel Type Distribution:\n", df['Fuel'].value_counts())

print("\nMPG Category Distribution:\n", df['MPG_Category'].value_counts())

# 1. Engine Size vs MPG
plt.figure(figsize=(8,5))
sns.scatterplot(x='EngineSize', y='CityMPG', hue='Fuel', data=df)
plt.title("Engine Size vs City MPG")
plt.show()

# 2. Fuel Type vs MPG
plt.figure(figsize=(10,5))
sns.boxplot(x='Fuel', y='CityMPG', data=df)
plt.xticks(rotation=45)
plt.title("Fuel Type vs City MPG")
plt.show()

# 3. Year vs MPG Trend
plt.figure(figsize=(10,5))
sns.lineplot(x='Year', y='CombinedMPG', data=df)
plt.title("Fuel Efficiency Over Years")
plt.show()

# 4. Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 5. MPG Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['CombinedMPG'], bins=30, kde=True)
plt.title("MPG Distribution")
plt.show()

# 6. Engine Category vs MPG
plt.figure(figsize=(6,4))
sns.barplot(x='EngineCategory', y='CombinedMPG', data=df)
plt.title("Engine Category vs MPG")
plt.show()

# 7. Cylinders vs MPG
plt.figure(figsize=(8,5))
sns.boxplot(x='Cylinders', y='CombinedMPG', data=df)
plt.title("Cylinders vs MPG")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(df['CombinedMPG'])
plt.title("Outlier Detection (MPG)")
plt.show()

# ================================
# 8. STATISTICAL INSIGHT
# ================================
correlation = df[['EngineSize','CombinedMPG']].corr()
print("\nCorrelation between Engine Size & MPG:\n", correlation)

# ================================
# 9. MACHINE LEARNING (LINEAR REGRESSION)
# ================================
X = df[['EngineSize','Cylinders']]
y = df['CombinedMPG']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nModel Performance:")
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# ================================
# 10. TOP 10 MOST EFFICIENT CARS
# ================================
top_cars = df.sort_values(by='CombinedMPG', ascending=False).head(10)
print("\nTop 10 Most Efficient Cars:\n", top_cars[['Make','Model','CombinedMPG']])

# ================================
# 11. SAVE CLEAN DATA
# ================================
df.to_csv("cleaned_vehicle_data.csv", index=False)