# week-2-data-analysis-mini-project

# Project Description

This project analyzes Taylor Swift’s concert tours dataset to explore attendance, revenue, and tour patterns. After cleaning and preparing the dataset, a KMeans clustering model is applied to identify groups of concerts (e.g., high-revenue stadium shows vs. smaller under-attended concerts). The project demonstrates skills in data cleaning, exploratory analysis, and machine learning clustering.

As a disclaimer, this dataset was last updated 2 years ago, which means it does not include data from Taylor Swift's most recent (and biggest) tour (The Era's Tour).

# Data source

This data set was taken from kaggle.com at the following link: https://www.kaggle.com/datasets/gayu14/taylor-concert-tours-impact-on-attendance-and

# Setup Instructions

1. Clone the repository: 

git clone your-repo-link

cd week-2-data-analysis-mini-project

2. Install dependencies by running make install

3. Run the script using make run

# Usage examples

1. Data cleaning and preparation

- Removes duplicates and missing values
- Cleans Revenue into numeric values
- Splits Attendance into Tickets_Sold, Tickets_Available, and Attendance_Rate

2. Exploratory Analysis

- Concert count by country
- Average revenue per tour
- Total and average revenue per country

3. Machine Learning – Clustering

- Features used: Revenue_clean, Tickets_Sold, Attendance_Rate
- KMeans groups concerts into 3 clusters:
    - Cluster 0: High-revenue, sold-out shows
    - Cluster 1: Medium-scale concerts
    - Cluster 2: Lower-revenue or under-attended shows

4. Visualization

- Scatterplot of Tickets_Sold vs. Revenue_clean colored by cluster
