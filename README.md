# week-2-data-analysis-mini-project

# CI Status Badge

[![CI](https://github.com/jordanandrewww/week-2-data-analysis-mini-project/actions/workflows/main.yml/badge.svg)](https://github.com/jordanandrewww/week-2-data-analysis-mini-project/actions/workflows/main.yml)

# Project Description

This mini project for week 3 is a continuation of week 2, but this week we focused on making the code cleaner and more readable, using refactoring, code
commenting, and code quality tools like black and flake8. The key take away for this week
and for this overall project is that it is important that we are able to take data from
a dataset, clean it, and then use it to create various visualizations that can allow us to
make clear, meaningul insights about said data.

This project analyzes Taylor Swift’s concert tours dataset to explore attendance, revenue, and tour patterns. After cleaning and preparing the dataset, a KMeans clustering model is applied to identify groups of concerts (e.g., high-revenue stadium shows vs. smaller under-attended concerts). The project demonstrates skills in data cleaning, exploratory analysis, and machine learning clustering.

For week 3, I also added another data visualization: a bar chart that shows the total revenue for each tour. I thought that this data is more impactful, and it is really interesting (in my opinion) to see how much bigger the Reputation tour was compared to the previous tours.

As a disclaimer, this dataset was last updated 2 years ago, which means it does not include data from Taylor Swift's most recent (and biggest) tour (The Era's Tour).

# Data source

This data set was taken from kaggle.com at the following link: https://www.kaggle.com/datasets/gayu14/taylor-concert-tours-impact-on-attendance-and

The data set contains the following columns:

- City
- Country
- Venue
- Opening act(s)
- Attendance (tickets sold / available)
- Revenue
- Tour

# Development Environment

A Dev Container is provided for a reproducible environment. When you open the repository in VS Code with the Dev Containers extension, all dependencies will be installed automatically based on:

- .devcontainer/devcontainer.json
- requirements.txt

# Setup Instructions

For running locally:

1. Clone the repository: 

git clone your-repo-link

cd week-2-data-analysis-mini-project

2. Install dependencies by running make install

3. Run the script using make run

4. Run tests using make tests

For running in the dev container:

0. Make sure you have Docker Desktop and Visual Studio Code with the Dev Containers extension installed.

1. Clone the repository: 

git clone your-repo-link

cd week-2-data-analysis-mini-project

2. Open the repository in VS code. You should see a prompt: "Reopen in Container"
Click it. VS Code will build the container using .devcontainer/devcontainer.json. Wait for the container to finish building (first time may take a few minutes). Once inside the container, dependencies are already installed from requirements.txt.

3. Once inside the container:

- Run analysis: make run

- Run tests: make test

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

- Scatterplot of Tickets_Sold vs. Revenue_clean colored by cluster (when run inside the dev
container, you can view the scatterplot locally, as the file called clusters.png will be created
automatically and you can click on it to view)

- Bar chart of Tour vs Revenue, where each bar's color corresponds with that tour album's theme color.
- when run when run inside the dev container, you can view the bar chart locally, as the file called avg_revenue_by_tour.png will be created
automatically and you can click on it to view