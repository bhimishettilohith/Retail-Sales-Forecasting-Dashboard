import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------------------------------------
# Page Configuration
# ---------------------------------------

st.set_page_config(
    page_title="Retail Sales Forecast Dashboard",
    page_icon="📈",
    layout="wide"
)

# ---------------------------------------
# Load Datasets
# ---------------------------------------

@st.cache_data
def load_data():

    train = pd.read_csv("train.csv")

    processed = pd.read_csv("processed_superstore.csv")

    monthly = pd.read_csv("monthly_sales_processed.csv")

    comparison = pd.read_csv("model_comparison.csv")

    anomaly = pd.read_csv("anomaly_report.csv")

    return train, processed, monthly, comparison, anomaly


train, processed, monthly, comparison, anomaly = load_data()

# ---------------------------------------
# Dashboard Title
# ---------------------------------------

st.title("📈 Retail Sales Forecast Dashboard")

st.subheader("Machine Learning based Sales Forecasting")

st.success("All datasets loaded successfully!")

# ---------------------------------------
# Dataset Information
# ---------------------------------------

st.header("Dataset Summary")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Training Records",
    len(train)
)

col2.metric(
    "Processed Records",
    len(processed)
)

col3.metric(
    "Detected Anomalies",
    len(anomaly)
)

st.divider()

st.subheader("Dataset Preview")

st.dataframe(train.head())

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Choose a Page",
    [
        "Dashboard",
        "Forecast Explorer",
        "Anomaly Detection",
        "Product Segmentation",
        "About"
    ]
)



# =====================================================
# DASHBOARD
# =====================================================

if page == "Dashboard":

    st.header("📊 Business Overview")

    total_sales = train["Sales"].sum()
    total_orders = train["Order ID"].nunique()
    total_customers = train["Customer ID"].nunique()
    avg_sales = train["Sales"].mean()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("💰 Total Sales", f"${total_sales:,.2f}")
    c2.metric("📦 Orders", total_orders)
    c3.metric("👥 Customers", total_customers)
    c4.metric("📈 Average Sale", f"${avg_sales:,.2f}")

    st.divider()

    # --------------------------
    # Monthly Sales Trend
    # --------------------------

    st.subheader("📈 Monthly Sales Trend")

    import plotly.express as px

    fig = px.line(
        monthly,
        x="Order Date",
        y="Sales",
        markers=True,
        title="Monthly Sales"
    )

    st.plotly_chart(fig, use_container_width=True)

    # --------------------------
    # Category & Region Charts
    # --------------------------

    left, right = st.columns(2)

    with left:

        category = (
            train.groupby("Category")["Sales"]
            .sum()
            .reset_index()
        )

        fig = px.bar(
            category,
            x="Category",
            y="Sales",
            color="Category",
            title="Sales by Category"
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:

        region = (
            train.groupby("Region")["Sales"]
            .sum()
            .reset_index()
        )

        fig = px.pie(
            region,
            values="Sales",
            names="Region",
            title="Sales by Region"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --------------------------
    # Interactive Filter
    # --------------------------

    st.subheader("🔍 Explore Sales")

    col1, col2 = st.columns(2)

    with col1:
        selected_region = st.selectbox(
            "Select Region",
            sorted(train["Region"].unique())
        )

    with col2:
        selected_category = st.selectbox(
            "Select Category",
            sorted(train["Category"].unique())
        )

    filtered = train[
        (train["Region"] == selected_region) &
        (train["Category"] == selected_category)
    ]

    st.dataframe(filtered, use_container_width=True)

    st.info(f"Showing **{len(filtered)}** matching records.")

    # =====================================================
# FORECAST EXPLORER
# =====================================================

elif page == "Forecast Explorer":

    st.header("📈 Forecast Explorer")

    st.success("🏆 Best Model: XGBoost")

    st.write(
        "The XGBoost model achieved the lowest forecasting error among all models."
    )

    st.divider()

    # --------------------------
    # Model Performance Metrics
    # --------------------------

    best_model = comparison.sort_values("RMSE").iloc[0]

    c1, c2, c3 = st.columns(3)

    c1.metric("MAE", f"{best_model['MAE']:.2f}")
    c2.metric("RMSE", f"{best_model['RMSE']:.2f}")
    c3.metric("MAPE", f"{best_model['MAPE']*100:.2f}%")

    st.divider()

    # --------------------------
    # Historical Monthly Sales
    # --------------------------

    st.subheader("Monthly Sales Trend")

    fig = px.line(
        monthly,
        x="Order Date",
        y="Sales",
        markers=True,
        title="Historical Monthly Sales"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --------------------------
    # Model Comparison
    # --------------------------

    st.subheader("Forecast Model Comparison")

    fig = px.bar(
        comparison,
        x="Model",
        y="RMSE",
        color="Model",
        text="RMSE",
        title="RMSE Comparison"
    )

    fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        comparison,
        use_container_width=True
    )

    st.download_button(
        "📥 Download Model Comparison",
        comparison.to_csv(index=False),
        file_name="model_comparison.csv",
        mime="text/csv"
    )

    st.divider()

    st.subheader("Business Recommendations")

    st.info("""
✅ XGBoost is recommended for future forecasting.

✅ Increase inventory before high-demand periods.

✅ Monitor seasonal sales trends regularly.

✅ Update the forecasting model with new sales data every month.
""")
    
    # =====================================================
# ANOMALY DETECTION
# =====================================================

elif page == "Anomaly Detection":

    st.header("🚨 Sales Anomaly Detection")

    st.write(
        "The following chart shows anomalies detected using the Isolation Forest model."
    )

    st.divider()

    # ----------------------------
    # Weekly Sales with Anomalies
    # ----------------------------

    fig = px.line(
        anomaly,
        x="Order Date",
        y="Sales",
        title="Weekly Sales with Detected Anomalies",
        markers=True
    )

    anomaly_points = anomaly[anomaly["Anomaly"] == -1]

    fig.add_scatter(
        x=anomaly_points["Order Date"],
        y=anomaly_points["Sales"],
        mode="markers",
        marker=dict(
            color="red",
            size=12,
            symbol="x"
        ),
        name="Anomaly"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # ----------------------------
    # Summary Cards
    # ----------------------------

    c1, c2 = st.columns(2)

    c1.metric(
        "Total Weeks",
        len(anomaly)
    )

    c2.metric(
        "Detected Anomalies",
        len(anomaly_points)
    )

    st.divider()

    # ----------------------------
    # Anomaly Table
    # ----------------------------

    st.subheader("Detected Anomalies")

    st.dataframe(
        anomaly_points,
        use_container_width=True
    )

    st.download_button(
        "📥 Download Anomaly Report",
        anomaly_points.to_csv(index=False),
        file_name="anomaly_report.csv",
        mime="text/csv"
    )

    st.divider()

    st.success("""
### Business Insights

✔ Sudden spikes generally indicate promotional campaigns.

✔ Extremely low sales may indicate stock shortages.

✔ Monitoring anomalies improves forecasting accuracy.

✔ Investigate abnormal weeks before making inventory decisions.
""")
    
    # =====================================================
# PRODUCT SEGMENTATION
# =====================================================

elif page == "Product Segmentation":

    st.header("📦 Product Demand Segmentation")

    st.write(
        "Products are grouped into clusters using K-Means based on sales performance."
    )

    st.divider()

    # ----------------------------
    # Prepare Data
    # ----------------------------

    summary = (
        processed.groupby("Sub-Category")
        .agg(
            TotalSales=("Sales", "sum"),
            AverageSales=("Sales", "mean"),
            ShippingDays=("Shipping Days", "mean")
        )
        .reset_index()
    )

    # ----------------------------
    # Scaling
    # ----------------------------

    scaler = StandardScaler()

    X = scaler.fit_transform(
        summary[["TotalSales", "AverageSales", "ShippingDays"]]
    )

    # ----------------------------
    # KMeans
    # ----------------------------

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    summary["Cluster"] = kmeans.fit_predict(X)

    # ----------------------------
    # PCA
    # ----------------------------

    pca = PCA(n_components=2)

    components = pca.fit_transform(X)

    summary["PC1"] = components[:, 0]
    summary["PC2"] = components[:, 1]

    # ----------------------------
    # Scatter Plot
    # ----------------------------

    st.subheader("Cluster Visualization")

    fig = px.scatter(
        summary,
        x="PC1",
        y="PC2",
        color=summary["Cluster"].astype(str),
        text="Sub-Category",
        title="K-Means Product Clusters"
    )

    fig.update_traces(
        textposition="top center"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # ----------------------------
    # Cluster Summary
    # ----------------------------

    cluster_summary = (
        summary.groupby("Cluster")
        .agg(
            Products=("Sub-Category", "count"),
            AverageSales=("AverageSales", "mean"),
            TotalSales=("TotalSales", "sum")
        )
        .reset_index()
    )

    st.subheader("Cluster Summary")

    st.dataframe(
        cluster_summary,
        use_container_width=True
    )

    st.divider()

    # ----------------------------
    # Product Details
    # ----------------------------

    st.subheader("Product Segments")

    st.dataframe(
        summary,
        use_container_width=True
    )

    st.download_button(
        "📥 Download Product Segments",
        summary.to_csv(index=False),
        file_name="product_segmentation.csv",
        mime="text/csv"
    )

    st.divider()

    st.success("""
### Inventory Recommendations

🟢 **Cluster 0**  
High revenue products. Maintain higher inventory.

🟡 **Cluster 1**  
Stable demand products. Keep balanced stock.

🔵 **Cluster 2**  
Growing demand products. Monitor and increase stock gradually.
""")

# =====================================================
# ABOUT
# =====================================================

elif page == "About":

    st.header("ℹ About This Project")

    st.markdown("""
# 📈 Retail Sales Forecasting using Machine Learning

This dashboard was developed to analyze retail sales data and provide
business insights using Machine Learning, Time Series Forecasting,
Anomaly Detection, and Product Segmentation.

The project demonstrates how data analytics can help businesses
forecast demand, optimize inventory, and identify unusual sales patterns.
""")

    st.divider()

    # ----------------------------
    # Dataset Information
    # ----------------------------

    st.subheader("📊 Dataset Summary")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Rows",
        len(train)
    )

    c2.metric(
        "Columns",
        len(train.columns)
    )

    c3.metric(
        "Categories",
        train["Category"].nunique()
    )

    st.divider()

    # ----------------------------
    # Models Used
    # ----------------------------

    st.subheader("🤖 Machine Learning Models")

    models = pd.DataFrame({

        "Model":[
            "SARIMA",
            "Prophet",
            "XGBoost",
            "Isolation Forest",
            "K-Means",
            "PCA"
        ],

        "Purpose":[
            "Time Series Forecasting",
            "Forecasting",
            "Demand Forecasting",
            "Anomaly Detection",
            "Product Segmentation",
            "Dimensionality Reduction"
        ]

    })

    st.dataframe(
        models,
        use_container_width=True
    )

    st.divider()

    # ----------------------------
    # Libraries Used
    # ----------------------------

    st.subheader("🛠 Libraries Used")

    libraries = [
        "Pandas",
        "NumPy",
        "Plotly",
        "Scikit-Learn",
        "XGBoost",
        "Streamlit"
    ]

    for lib in libraries:
        st.write(f"✅ {lib}")

    st.divider()

    # ----------------------------
    # Project Deliverables
    # ----------------------------

    st.subheader("📁 Project Deliverables")

    deliverables = pd.DataFrame({

        "Deliverable":[
            "Jupyter Notebook",
            "Streamlit Dashboard",
            "requirements.txt",
            "README.md",
            "summary.docx"
        ],

        "Status":[
            "Completed",
            "Completed",
            "Completed",
            "Completed",
            "Completed"
        ]

    })

    st.dataframe(
        deliverables,
        use_container_width=True
    )

    st.divider()

    st.success("""
🎉 Project Successfully Completed!

This dashboard demonstrates:

✔ Data Cleaning

✔ Exploratory Data Analysis

✔ Time Series Forecasting

✔ Forecast Model Comparison

✔ Anomaly Detection

✔ Product Segmentation

✔ Interactive Dashboard

✔ Business Intelligence
""")
    
# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown(
"""
<div style='text-align:center; color:gray;'>

### 📈 Retail Sales Forecast Dashboard

Developed using **Python • Streamlit • Plotly • Scikit-Learn • XGBoost**

Bhimishetti Lohith | Machine Learning | Business Analytics | Time Series Forecasting

</div>
""",
unsafe_allow_html=True
)    