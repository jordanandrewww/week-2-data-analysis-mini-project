import pytest
import pandas as pd
import numpy as np
import analyzedata as ad


# -------------------------
# Unit Tests
# -------------------------


def test_clean_data_removes_duplicates():
    dataframe = pd.DataFrame(
        {
            "Tour": ["Tour A", "Tour A"],
            "City": ["NYC", "NYC"],
            "Country": ["United States", "United States"],
            "Revenue": ["$1000", "$1000"],
            "Attendance (tickets sold / available)": ["100/200", "100/200"],
        }
    )
    cleaned = ad.clean_data(dataframe)
    assert cleaned.shape[0] == 1  # duplicates removed


def test_clean_data_revenue_and_attendance():
    dataframe = pd.DataFrame(
        {
            "Tour": ["Tour A"],
            "City": ["NYC"],
            "Country": ["United States"],
            "Revenue": ["$1,500.75"],
            "Attendance (tickets sold / available)": ["123/456"],
        }
    )
    cleaned = ad.clean_data(dataframe)

    assert np.isclose(cleaned["Revenue_clean"].iloc[0], 1500.75)
    assert cleaned["Tickets_Sold"].iloc[0] == 123
    assert cleaned["Tickets_Available"].iloc[0] == 456
    assert np.isclose(cleaned["Attendance_Rate"].iloc[0], 123 / 456)


def test_clean_data_divide_by_zero():
    dataframe = pd.DataFrame(
        {
            "Tour": ["Tour A"],
            "City": ["NYC"],
            "Country": ["United States"],
            "Revenue": ["$500"],
            "Attendance (tickets sold / available)": ["100/0"],
        }
    )
    cleaned = ad.clean_data(dataframe)
    assert cleaned["Attendance_Rate"].iloc[0] == 100  # handled safely


def test_summarize_data_returns_expected_keys(sample_df):
    cleaned = ad.clean_data(sample_df)
    summaries = ad.summarize_data(cleaned)

    assert "avg_revenue_tour" in summaries
    assert "concerts_per_country" in summaries
    assert "total_revenue_country" in summaries
    assert "avg_revenue_by_country" in summaries


# -------------------------
# Integration / System Tests
# -------------------------


@pytest.fixture
def sample_df():
    """Provide a small sample dataframe to simulate concerts."""
    return pd.DataFrame(
        {
            "Tour": ["Tour A", "Tour A", "Tour B"],
            "City": ["New York", "LA", "London"],
            "Country": ["United States", "United States", "UK"],
            "Revenue": ["$1000", "$2000", "$1500"],
            "Attendance (tickets sold / available)": ["100/200", "300/400", "250/250"],
        }
    )


def test_run_kmeans_adds_cluster_labels(sample_df):
    cleaned = ad.clean_data(sample_df)
    clustered = ad.run_kmeans(cleaned, n_clusters=2)

    assert "Cluster" in clustered.columns
    assert clustered["Cluster"].nunique() <= 2
    assert len(clustered["Cluster"]) == len(sample_df)
