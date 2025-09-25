import kagglehub
import pandas as pd
import os
import re
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# downloads and loads the dataset from Kaggle and returns the CSV path
def download_data():
    path = kagglehub.dataset_download("gayu14/taylor-concert-tours-impact-on-attendance-and")
    files = os.listdir(path)
    csv_file = files[0]
    return os.path.join(path, csv_file)

# reads in the concert dataset
def load_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, encoding="ISO-8859-1")


def clean_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    # removes duplicates from the dataframe
    dataframe = dataframe.drop_duplicates().copy()

    # cleans the Revenue column and creates a new numeric column
    dataframe.loc[:, 'Revenue_clean'] = (
        dataframe['Revenue']
        .astype(str)
        .str.replace(r'[^\d.]', '', regex=True)
        .replace('', '0')
        .astype(float)
    )

    attendance_split = split_attendance_col(dataframe)

    # Helper function to clean numbers and convert the strings to integers
    def clean_number(s):
        s = re.sub(r'[^\d]', '', str(s))
        return int(s) if s else 0

    dataframe.loc[:, 'Tickets_Sold'] = attendance_split[0].apply(clean_number)
    dataframe.loc[:, 'Tickets_Available'] = attendance_split[1].apply(clean_number)
    dataframe.loc[:, 'Attendance_Rate'] = (
        dataframe['Tickets_Sold'] / dataframe['Tickets_Available'].replace(0, 1)
    )

    return dataframe

# Splits the attendance column into two separate columns (tickets sold and tickets available)
def split_attendance_col(dataframe):
    attendance_split = (
        dataframe['Attendance (tickets sold / available)']
        .astype(str)
        .fillna('0/0')
        .str.split('/', expand=True)
    )
    
    return attendance_split

# Summarizes key statistics from the cleaned dataframe
def summarize_data(dataframe: pd.DataFrame):
    return {
        "avg_revenue_tour": dataframe.groupby('Tour')['Revenue_clean'].mean(),
        "concerts_per_country": dataframe.groupby('Country')['City'].count(),
        "total_revenue_country": dataframe.groupby('Country')['Revenue_clean'].sum(),
        "avg_revenue_by_country": dataframe.groupby('Country')['Revenue_clean'].mean().reset_index(),
    }

# Applies KMeans clustering to the cleaned dataframe
def run_kmeans(dataframe: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    X_cluster = dataframe[['Revenue_clean', 'Tickets_Sold', 'Attendance_Rate']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    dataframe.loc[:, 'Cluster'] = kmeans.fit_predict(X_scaled)

    return dataframe

# Plots the clusters and saves the plot locally as a PNG
def plot_clusters(dataframe: pd.DataFrame, save_path: str = "clusters.png"):
    plt.figure(figsize=(8, 6))
    plt.scatter(
        dataframe['Tickets_Sold'],
        dataframe['Revenue_clean'],
        c=dataframe['Cluster'],
        cmap='viridis',
        alpha=0.6
    )
    plt.xlabel("Tickets Sold")
    plt.ylabel("Revenue (clean)")
    plt.title("Concert Clusters (KMeans)")
    plt.colorbar(label="Cluster")

    # Always save (works in Docker/DevContainer)
    plt.savefig(save_path)
    print(f"Cluster plot saved as {save_path}")

    # Show only if a DISPLAY is available (local machine)
    if os.environ.get("DISPLAY"):
        plt.show()

if __name__ == "__main__":
    # Main workflow with printouts
    csv_path = download_data()
    print("Dataset downloaded to:", os.path.dirname(csv_path))

    dataframe = load_data(csv_path)
    print("\nRaw Data Preview:")
    print(dataframe.head())
    print(dataframe.info())
    print("\nMissing values per column:\n", dataframe.isna().sum())
    print("\nNumber of duplicate rows before cleaning:", dataframe.duplicated().sum())

    dataframe = clean_data(dataframe)
    print("\nShape after removing duplicates:", dataframe.shape)
    print("Number of duplicate rows after cleaning:", dataframe.duplicated().sum())

    summaries = summarize_data(dataframe)
    print("\nAverage Revenue by Tour:\n", summaries["avg_revenue_tour"])
    print("\nConcert count by Country:\n", summaries["concerts_per_country"])
    print("\nTotal Revenue by Country:\n", summaries["total_revenue_country"])
    print("\nAverage Revenue by Country:\n", summaries["avg_revenue_by_country"])

    dataframe = run_kmeans(dataframe)
    print("\nCluster Centers added. Sample with cluster labels:")
    print(dataframe[['Tour', 'City', 'Country', 'Revenue_clean',
              'Tickets_Sold', 'Attendance_Rate', 'Cluster']].head())

    plot_clusters(dataframe)
