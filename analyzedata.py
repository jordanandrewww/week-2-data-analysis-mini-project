import kagglehub
import pandas as pd
import os
import re

# Importing the data:


# Download the dataset from Kaggle
path = kagglehub.dataset_download("gayu14/taylor-concert-tours-impact-on-attendance-and")
print("Dataset downloaded to:", path)

# List files inside the dataset
files = os.listdir(path)
print("Available files:", files)

# We know that there is only one file, so we can directly access it
csv_file = files[0] 

# Getting the full path to the csv file
csv_path = os.path.join(path, csv_file)

# Load CSV data into a pandas DataFrame
df = pd.read_csv(csv_path, encoding="ISO-8859-1")

# Inspecting/cleaning the data:


print(df.head())
print(df.info())

# Count missing values per column by checking for missing/null values
print(df.isna().sum())

# Count duplicate rows
print("Number of duplicate rows:", df.duplicated().sum())

# Check duplicates
num_dupes = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {num_dupes}")

# Remove duplicates
df = df.drop_duplicates()
print("After removing duplicates, dataset shape:", df.shape)

# Check to ensure duplicates have been removed
num_dupes = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {num_dupes}")

# Clean Revenue column by removing everything that isn't a digit or a decimal point
# and converting it to a float
df['Revenue_clean'] = (
    df['Revenue']
    .astype(str)  # make sure it's string
    .str.replace(r'[^\d.]', '', regex=True)  # keep only numbers and decimal points
    .replace('', '0')  # replace empty strings with 0
    .astype(float)
)

# Split Attendance column safely
attendance_split = df['Attendance (tickets sold / available)'].astype(str).fillna('0/0').str.split('/', expand=True)

def clean_number(s):
    s = str(s)  # make sure it’s a string
    s = re.sub(r'[^\d]', '', s)  # remove non-digit characters
    return int(s) if s else 0

df['Tickets_Sold'] = attendance_split[0].apply(clean_number)
df['Tickets_Available'] = attendance_split[1].apply(clean_number)
df['Attendance_Rate'] = df['Tickets_Sold'] / df['Tickets_Available'].replace(0, 1)  # avoid divide by 0


# Filtering/Grouping the data:


# Filtering data for concerts in the USA
usa_concerts = df[df['Country'] == 'United States']
print("Number of concerts in the USA:", len(usa_concerts))

# Average revenue per tour
avg_revenue_tour = df.groupby('Tour')['Revenue_clean'].mean()
print("\nAverage Revenue by Tour:\n", avg_revenue_tour)

# Number of concerts per country
concerts_per_country = df.groupby('Country')['City'].count()
print("\nConcert count by Country:\n", concerts_per_country)

# Total revenue per country
total_revenue_country = df.groupby('Country')['Revenue_clean'].sum()
print("\nTotal Revenue by Country:\n", total_revenue_country)

# Average revenue per country
avg_revenue_by_country = df.groupby('Country')['Revenue_clean'].mean().reset_index()
print(avg_revenue_by_country)

# Exploring a Machine Learning Algorithm:

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Features for clustering
X_cluster = df[['Revenue_clean', 'Tickets_Sold', 'Attendance_Rate']]

# Scale features so large values (like revenue) don't dominate
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Use KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Inspect cluster centers
print("Cluster Centers (scaled):\n", kmeans.cluster_centers_)

# Quick look at concerts with their assigned cluster
print(df[['Tour', 'City', 'Country', 'Revenue_clean', 'Tickets_Sold', 
          'Attendance_Rate', 'Cluster']].head())

# --- Visualization ---
# Plot clusters based on Tickets Sold vs Revenue
plt.figure(figsize=(8,6))
plt.scatter(df['Tickets_Sold'], df['Revenue_clean'], c=df['Cluster'], cmap='viridis', alpha=0.6)
plt.xlabel("Tickets Sold")
plt.ylabel("Revenue (clean)")
plt.title("Concert Clusters (KMeans)")
plt.colorbar(label="Cluster")
plt.show()








































