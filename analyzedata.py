import kagglehub
import pandas as pd
import os
import re
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def download_data():
    """Download dataset from Kaggle and return CSV path."""
    path = kagglehub.dataset_download("gayu14/taylor-concert-tours-impact-on-attendance-and")
    files = os.listdir(path)
    csv_file = files[0]
    return os.path.join(path, csv_file)


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the concert dataset."""
    return pd.read_csv(csv_path, encoding="ISO-8859-1")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean revenue, attendance, and duplicates."""
    df = df.drop_duplicates().copy()

    df.loc[:, 'Revenue_clean'] = (
        df['Revenue']
        .astype(str)
        .str.replace(r'[^\d.]', '', regex=True)
        .replace('', '0')
        .astype(float)
    )

    attendance_split = (
        df['Attendance (tickets sold / available)']
        .astype(str)
        .fillna('0/0')
        .str.split('/', expand=True)
    )

    def clean_number(s):
        s = re.sub(r'[^\d]', '', str(s))
        return int(s) if s else 0

    df.loc[:, 'Tickets_Sold'] = attendance_split[0].apply(clean_number)
    df.loc[:, 'Tickets_Available'] = attendance_split[1].apply(clean_number)
    df.loc[:, 'Attendance_Rate'] = (
        df['Tickets_Sold'] / df['Tickets_Available'].replace(0, 1)
    )

    return df


def summarize_data(df: pd.DataFrame):
    """Return key summaries."""
    return {
        "avg_revenue_tour": df.groupby('Tour')['Revenue_clean'].mean(),
        "concerts_per_country": df.groupby('Country')['City'].count(),
        "total_revenue_country": df.groupby('Country')['Revenue_clean'].sum(),
        "avg_revenue_by_country": df.groupby('Country')['Revenue_clean'].mean().reset_index(),
    }


def run_kmeans(df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """Apply KMeans clustering and return dataframe with cluster labels."""
    X_cluster = df[['Revenue_clean', 'Tickets_Sold', 'Attendance_Rate']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df.loc[:, 'Cluster'] = kmeans.fit_predict(X_scaled)

    return df


def plot_clusters(df: pd.DataFrame, save_path: str = "clusters.png"):
    """Plot clusters based on Tickets Sold vs Revenue.
    Saves the plot as PNG (for containers) and shows it if running locally.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(
        df['Tickets_Sold'],
        df['Revenue_clean'],
        c=df['Cluster'],
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

    df = load_data(csv_path)
    print("\nRaw Data Preview:")
    print(df.head())
    print(df.info())
    print("\nMissing values per column:\n", df.isna().sum())
    print("\nNumber of duplicate rows before cleaning:", df.duplicated().sum())

    df = clean_data(df)
    print("\nShape after removing duplicates:", df.shape)
    print("Number of duplicate rows after cleaning:", df.duplicated().sum())

    summaries = summarize_data(df)
    print("\nAverage Revenue by Tour:\n", summaries["avg_revenue_tour"])
    print("\nConcert count by Country:\n", summaries["concerts_per_country"])
    print("\nTotal Revenue by Country:\n", summaries["total_revenue_country"])
    print("\nAverage Revenue by Country:\n", summaries["avg_revenue_by_country"])

    df = run_kmeans(df)
    print("\nCluster Centers added. Sample with cluster labels:")
    print(df[['Tour', 'City', 'Country', 'Revenue_clean',
              'Tickets_Sold', 'Attendance_Rate', 'Cluster']].head())

    plot_clusters(df)
