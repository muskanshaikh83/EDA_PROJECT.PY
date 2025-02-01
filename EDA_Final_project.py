import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Step 1: Read the dataset
df = pd.read_csv('Customer Churn.csv')
print(df)

# Checking for missing and duplicated values
print(df.isnull().sum().sum())  # Number of missing values
print("Duplicated values:", df['customer_ID'].duplicated().sum())

# Removing duplicates based on 'customer_ID'
df = df.drop_duplicates(subset='customer_ID', keep=False)

# Handling missing or incorrect values in 'Total_Charges'
df["Total_Charges"] = df["Total_Charges"].replace(" ", "0")
df["Total_Charges"] = df["Total_Charges"].astype("float")
print(df)

# Converting 'Senior_Citizen' column to 'yes' or 'no'
def conv(value):
        return "yes" if value == 1 else "no"

# Convert 'Total Charges' to numeric (if needed)
df["Total_Charges"] = pd.to_numeric(df["Total_Charges"], errors='coerce')


# Select only numerical columns
numeric_cols = df.select_dtypes(include=['int64', 'float64'])

# Compute correlation matrix
correlation_matrix = numeric_cols.corr()


df['Senior_Citizen'] = df["Senior_Citizen"].apply(conv)
# Scatter plot 1: Monthly Charges vs Total Charges with Churn status
sns.scatterplot(x=df['Monthly_Charges'], y=df['Total_Charges'], hue=df['Churn'])
plt.title('Monthly Charges vs Total Charges', fontsize=14)
plt.xlabel('Monthly Charges', fontsize=12)
plt.ylabel('Total Charges', fontsize=12)
plt.legend(title='Churn', loc='upper left')
plt.grid(True)
plt.show()

#2 Bar plot: Average Monthly Charges by Payment Method
#plt.figure(figsize=(3, 4))
sns.barplot(x=df['Payment_Method'], y=df['Monthly_Charges'], palette='Paired',)
plt.title('Avg Monthly Charges by Payment Method', fontsize=12)
plt.xlabel('Payment Method', fontsize=12)
plt.ylabel('Avg Monthly Charges', fontsize=12)
plt.xticks(rotation=5)
plt.show()

#3 Line plot: Avg Monthly Charges by Payment Method
sns.lineplot(x=df['Payment_Method'], y=df['Monthly_Charges'], marker='o', palette='Dark2')
plt.title('Avg Monthly Charges by Payment Method', fontsize=14)
plt.xlabel('Payment Method', fontsize=12)
plt.ylabel('Avg Monthly Charges', fontsize=12)
plt.xticks(rotation=45)
plt.show()

#4 Pie chart: Percentage of Churned Customers
plt.figure(figsize=(5,6))
gb = df.groupby("Churn").agg({'Churn': "count"})
plt.pie(gb['Churn'], labels=gb.index, autopct="%1.2f%%")
plt.title("Percentage of Churned Customers", fontsize=20)
plt.show()

#5 Box plot: Churn vs Tenure (Customer tenure by churn status)

sns.boxplot(x=df['Churn'], y=df['tenure'], palette='Set2')
plt.title('Churn vs Tenure', fontsize=14)
plt.xlabel('Churn Status', fontsize=12)
plt.ylabel('Tenure (Months)', fontsize=12)
plt.show()

#6 Bar plot: Monthly Charges by Internet Service
sns.barplot(x=df['Internet_Service'], y=df['Monthly_Charges'], palette='pastel')
plt.title('Monthly Charges by Internet Service', fontsize=14)
plt.xlabel('Internet Service', fontsize=12)
plt.ylabel('Monthly Charges', fontsize=12)
plt.show()

# 7 Stacked Bar plot: Internet Service vs Contract Type
pd.crosstab(df['Internet_Service'], df['Contract']).plot(kind='bar', stacked=True, colormap='Accent')
plt.title('Internet Service vs Contract Type', fontsize=14)
plt.xlabel('Internet Service', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()

# 8  Histogram: Monthly Charges Distribution (Churned vs Not Churned)
sns.histplot(df[df['Churn'] == 'Yes']['Monthly_Charges'], kde=True, color='red', label='Churned', linewidth=2)
sns.histplot(df[df['Churn'] == 'No']['Monthly_Charges'], kde=True, color='green', label='Not Churned', linewidth=2)
plt.title('Monthly Charges Distribution (Churn vs No Churn)', fontsize=14)
plt.xlabel('Monthly Charges', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.show()

# 9 Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Numerical Features", fontsize=20)
plt.show()


#10 the  line chart 
# Churn vs. Contract Type (Bar Chart)**: Examines churn by contract duration.
#- **Tenure vs. Monthly Charges (Scatter Plot)**: Analyzes spending patterns.
df = df.groupby("Contract")["Churn"].value_counts().unstack()
sns.lineplot(x=df.index,y=df['Yes'],color='purple')#palette='pastel
plt.title('Churn vs. Contract',fontsize=12)
plt.xlabel('Contract types',fontsize=12)
plt.ylabel('Churn',fontsize=12)
plt.show()
