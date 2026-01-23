from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)

# Features + target as a single DataFrame
df = housing.frame

# Quick check
print(df.head())
print(df.shape)

# Boxplot of median house value in California
df.boxplot('MedHouseVal')
plt.ylabel('Median House Value ($)')
plt.title('Boxplot of Median House Value in California')

# Save our image
plt.savefig('figs/boxplot.png')
plt.show()

