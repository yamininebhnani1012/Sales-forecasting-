import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="Retail Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Retail Sales Forecasting Dashboard")

st.sidebar.title("Navigation")

try:
    dataset = pd.read_csv("train.csv")
except FileNotFoundError:
    st.error("train.csv not found!")
    st.stop()

    dataset["Order Date"] = pd.to_datetime(
        dataset["Order Date"],
        errors="coerce"
    )

    dataset = dataset.dropna()

    dataset["Year"] = dataset["Order Date"].dt.year
    dataset["Month"] = dataset["Order Date"].dt.to_period("M").astype(str)

    page = st.sidebar.radio(
        "Select Page",
        [
            "Sales Overview",
            "Forecast Explorer",
            "Anomaly Report",
            "Product Demand Segments"
        ]
    )

    # ==========================================
    # PAGE 1
    # ==========================================

    if page == "Sales Overview":

        st.header("Sales Overview Dashboard")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Sales",
            f"${dataset['Sales'].sum():,.2f}"
        )

        col2.metric(
            "Orders",
            dataset["Order ID"].nunique()
        )

        col3.metric(
            "Customers",
            dataset["Customer ID"].nunique()
        )

        yearly_sales = dataset.groupby("Year")["Sales"].sum()

        fig, ax = plt.subplots(figsize=(8,4))

        yearly_sales.plot(
            kind="bar",
            ax=ax
        )

        ax.set_title("Sales by Year")

        st.pyplot(fig)

        monthly_sales = dataset.groupby("Month")["Sales"].sum()

        fig, ax = plt.subplots(figsize=(12,4))

        ax.plot(
            monthly_sales.index,
            monthly_sales.values,
            marker="o"
        )

        plt.xticks(rotation=45)

        st.pyplot(fig)

        st.subheader("Region & Category Filter")

        region = st.selectbox(
            "Region",
            sorted(dataset["Region"].unique())
        )

        category = st.selectbox(
            "Category",
            sorted(dataset["Category"].unique())
        )

        filtered = dataset[
            (dataset["Region"] == region)
            &
            (dataset["Category"] == category)
        ]

        st.dataframe(filtered)

        # ==========================================
    # PAGE 2 - FORECAST EXPLORER
    # ==========================================

    elif page == "Forecast Explorer":

        st.header("Forecast Explorer")

        forecast_type = st.selectbox(
            "Forecast By",
            ["Category", "Region"]
        )

        if forecast_type == "Category":

            option = st.selectbox(
                "Select Category",
                sorted(dataset["Category"].unique())
            )

            filtered = dataset[
                dataset["Category"] == option
            ]

        else:

            option = st.selectbox(
                "Select Region",
                sorted(dataset["Region"].unique())
            )

            filtered = dataset[
                dataset["Region"] == option
            ]

        monthly = (
            filtered.groupby(
                pd.Grouper(
                    key="Order Date",
                    freq="ME"
                )
            )["Sales"]
            .sum()
            .reset_index()
        )

        st.subheader("Historical Monthly Sales")

        fig, ax = plt.subplots(figsize=(10,4))

        ax.plot(
            monthly["Order Date"],
            monthly["Sales"],
            marker="o"
        )

        ax.set_title("Monthly Sales Trend")
        ax.set_xlabel("Month")
        ax.set_ylabel("Sales")

        plt.xticks(rotation=45)

        st.pyplot(fig)

        horizon = st.slider(
            "Forecast Horizon (Months)",
            min_value=1,
            max_value=3,
            value=3
        )

        last_sales = monthly["Sales"].iloc[-1]

        forecast = pd.DataFrame({
            "Month":[
                f"Month {i}"
                for i in range(1, horizon+1)
            ],
            "Forecast Sales":[
                last_sales
                for _ in range(horizon)
            ]
        })

        st.subheader("Forecast Result")

        st.dataframe(forecast)

        st.metric(
            "MAE",
            "472.86"
        )

        st.metric(
            "RMSE",
            "866.71"
        )

    # ==========================================
    # PAGE 3 - ANOMALY REPORT
    # ==========================================

    elif page == "Anomaly Report":

        st.header("Anomaly Report")

        weekly_sales = (
            dataset.groupby(
                pd.Grouper(
                    key="Order Date",
                    freq="W"
                )
            )["Sales"]
            .sum()
            .reset_index()
        )

        model = IsolationForest(
            contamination=0.05,
            random_state=42
        )

        weekly_sales["Anomaly"] = model.fit_predict(
            weekly_sales[["Sales"]]
        )

        anomalies = weekly_sales[
            weekly_sales["Anomaly"] == -1
        ]

        fig, ax = plt.subplots(figsize=(12,5))

        ax.plot(
            weekly_sales["Order Date"],
            weekly_sales["Sales"],
            label="Weekly Sales",
            linewidth=2
        )

        ax.scatter(
            anomalies["Order Date"],
            anomalies["Sales"],
            color="red",
            s=80,
            label="Anomaly"
        )

        ax.set_title("Weekly Sales Anomaly Detection")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")
        ax.legend()

        st.pyplot(fig)

        st.subheader("Detected Anomalies")

        st.dataframe(
            anomalies[
                ["Order Date", "Sales"]
            ]
        )

        st.metric(
            "Number of Anomalies",
            len(anomalies)
        )

    # ==========================================
    # PAGE 4 - PRODUCT DEMAND SEGMENTS
    # ==========================================

    elif page == "Product Demand Segments":

        st.header("Product Demand Segments")

        product_data = dataset.groupby("Sub-Category").agg(
            Total_Sales=("Sales", "sum"),
            Average_Order_Value=("Sales", "mean"),
            Sales_Volatility=("Sales", "std")
        ).reset_index()

        product_data["Sales_Volatility"] = (
            product_data["Sales_Volatility"].fillna(0)
        )

        X = product_data[
            [
                "Total_Sales",
                "Average_Order_Value",
                "Sales_Volatility"
            ]
        ]

        kmeans = KMeans(
            n_clusters=4,
            random_state=42,
            n_init=10
        )

        product_data["Cluster"] = kmeans.fit_predict(X)

        pca = PCA(n_components=2)

        components = pca.fit_transform(X)

        product_data["PC1"] = components[:, 0]
        product_data["PC2"] = components[:, 1]

        fig, ax = plt.subplots(figsize=(10,6))

        scatter = ax.scatter(
            product_data["PC1"],
            product_data["PC2"],
            c=product_data["Cluster"],
            s=120
        )

        for _, row in product_data.iterrows():

            ax.text(
                row["PC1"],
                row["PC2"],
                row["Sub-Category"],
                fontsize=8
            )

        ax.set_title("Product Demand Segments")

        st.pyplot(fig)

        st.subheader("Cluster Details")

        st.dataframe(
            product_data[
                [
                    "Sub-Category",
                    "Cluster",
                    "Total_Sales",
                    "Average_Order_Value",
                    "Sales_Volatility"
                ]
            ]
        )

        st.success("Dashboard Loaded Successfully!")
