
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)


n = 1000

data = pd.DataFrame({
    "distance_km": np.random.uniform(1, 20, n),
    "package_weight_kg": np.random.uniform(0.5, 5, n),
    "wind_speed_mps": np.random.uniform(0, 15, n),
    "temperature_c": np.random.uniform(15, 40, n),
    "rain_intensity_mm": np.random.uniform(0, 10, n),
})

base = 5
data["battery_used_percent"] = (
    base
    + data["distance_km"] * 3
    + data["package_weight_kg"] * 4
    + data["wind_speed_mps"] * 1.5
    + data["rain_intensity_mm"] * 0.8
    + np.random.uniform(-5, 5, n)
)

data["battery_used_percent"] = np.clip(data["battery_used_percent"], 0, 100)

print("Initial Shape:", data.shape)


data.loc[10:20, "wind_speed_mps"] = np.nan
data = pd.concat([data, data.iloc[[5]]], ignore_index=True)
data.loc[30, "battery_used_percent"] = -10

print("\nMissing Before Cleaning:")
print(data.isnull().sum())


data["wind_speed_mps"] = data["wind_speed_mps"].fillna(
    data["wind_speed_mps"].mean()
)

data = data.drop_duplicates()

data = data[data["battery_used_percent"] > 0]


data["energy_load"] = data["distance_km"] * data["package_weight_kg"]
data["wind_impact"] = data["wind_speed_mps"] * data["distance_km"]
data["battery_per_km"] = data["battery_used_percent"] / data["distance_km"]

print("\nMissing After Cleaning:")
print(data.isnull().sum())

print("\nFinal Shape:", data.shape)
print(data.loc[500:505])


plt.figure(figsize=(8,5))
sns.scatterplot(x="distance_km", y="battery_used_percent", data=data)
plt.title("Battery Usage vs Distance")
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(x="wind_speed_mps", y="battery_used_percent", data=data)
plt.title("Wind Impact on Battery Usage")
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(x="battery_per_km", y="battery_used_percent", data=data)
plt.title("Battery Efficiency Analysis")
plt.show()


