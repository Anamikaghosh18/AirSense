import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import joblib
import json

st.set_page_config(page_title="AirSense", page_icon="🌍", layout="wide")

# --------------------------- Minimal CSS - Just Fixes ---------------------------
st.markdown("""
<style>
/* Fix sidebar scrolling */
[data-testid="stSidebar"] {
    overflow-y: auto !important;
}

[data-testid="stSidebar"] > div:first-child {
    overflow-y: auto !important;
}

/* Fix input field visibility */
.stNumberInput input,
.stTextInput input {
    background-color: white !important;
    color: #0e1117 !important;
}

/* Fix selectbox visibility */
.stSelectbox [data-baseweb="select"] > div {
    background-color: white !important;
    color: #0e1117 !important;
}

/* Fix label visibility */
.stNumberInput label,
.stTextInput label,
.stSelectbox label {
    color: white !important;
}

/* Feature cards - fix text visibility */
.feature-card {
    background: #ffffff;
    padding: 1.5rem;
    border-radius: 0.5rem;
    border: 1px solid #e0e0e0;
    margin-bottom: 1rem;
}

.feature-card strong {
    color: #0e1117;
    display: block;
    margin-bottom: 0.25rem;
}

.feature-card .desc {
    color: #31333F;
}

/* Stat items - fix text visibility */
.stat-item {
    padding: 1rem 0;
    border-bottom: 1px solid #e0e0e0;
}

.stat-item .label {
    color: gray;
}

.stat-item .value {
    color: white;
    font-weight: 600;
}

/* Info boxes */
.info-box {
    background: #e7f3ff;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #0068c9;
    margin: 1rem 0;
    color: #0e1117;
}

.success-box {
    background: #d4edda;
    padding: 1.5rem;
    border-radius: 0.5rem;
    border-left: 4px solid #28a745;
    margin: 1rem 0;
}

.success-box h3 {
    color: #155724;
    margin: 0 0 0.5rem 0;
}

.success-box p {
    color: #155724;
    margin: 0.5rem 0 0 0;
}

.warning-box {
    background: #fff3cd;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #ffc107;
    margin: 1rem 0;
    color: #856404;
}

/* Remove footer/header */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Fix main content padding for footer */
.main {
    padding-bottom: 80px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------- Sidebar ---------------------------
with st.sidebar:
    st.image("assets/pollution.jpg", width=120)
    st.markdown("### AirSense")
    st.caption("Air Quality Intelligence Platform")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["🏠 Home", "🔬 Operations", "📈 Analytics"],
        label_visibility="collapsed"
    )
    
# Clean page names
page = page.split(" ", 1)[1] if " " in page else page

# --------------------------- Home Page ---------------------------
if page == "Home":
    st.title("AirSense 🌍")
    st.markdown("Monitor, predict, and visualize air pollution trends with machine learning intelligence")
    
    st.markdown("")
    
    col1, col2 = st.columns([1.5, 1], gap="large")
    
    with col1:
        st.markdown("### Welcome to AirSense")
        st.write("""
        AirSense provides comprehensive air quality monitoring and predictive analytics 
        for cities worldwide. Leverage advanced machine learning models to understand 
        pollution patterns, detect anomalies, and forecast trends.
        """)
        
        st.markdown("")
        st.markdown("### Platform Capabilities")
        
        features = [
            ("🎯", "Pollution Index Prediction", "Predict pollution levels using PM10, PM2.5, and NO2 data"),
            ("🏙️", "City Severity Analysis", "Classify cities by pollution severity levels"),
            ("🚨", "Anomaly Detection", "Identify unusual pollution hotspots"),
            ("📈", "Trend Forecasting", "Analyze historical and predicted pollution trends")
        ]
        
        for icon, title, desc in features:
            st.markdown(f"""
            <div class="feature-card">
                <strong>{icon} {title}</strong>
                <span class="desc">{desc}</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Platform Overview")
        
        try:
            df = pd.read_excel("data/raw/who_air_quality.xlsx")
            
            metrics = [
                ("Cities", df["city"].nunique(), "🏙️"),
                ("Countries", df["country_name"].nunique(), "🌍"),
                ("Data Points", f"{len(df):,}", "📊"),
                ("Years", df["year"].nunique(), "📅")
            ]
            
            for label, value, emoji in metrics:
                st.markdown(f"""
                <div class="stat-item">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span class="label">{emoji} {label}</span>
                        <span class="value">{value}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        except:
            st.info("Load data to view statistics")
        
        st.markdown("")
        st.markdown("### Quick Start")
        st.write("""
        1. Navigate to **Operations** to run predictions
        2. Explore the **Dashboard** for insights
        3. Review **Analytics** for model performance
        """)

# --------------------------- Operations ---------------------------
elif page == "Operations":
    st.title("🔬 Operations")
    st.write("Run predictions and analyze pollution data")
    
    st.markdown("")

    objective = st.selectbox(
        "Select Analysis Type",
        ["Pollution Index Prediction", "City Severity Classification", "Anomaly Detection", "Yearly Trend Forecast"]
    )

    try:
        df = pd.read_csv("data/processed/processed_data.csv")
        models = {
            "xgb": joblib.load("models/xgb_pollution_index_model.pkl"),
            "kmeans": joblib.load("models/city_pollution_model.pkl"),
            "severity_map": joblib.load("models/severity_label_map.pkl"),
            "dbscan": joblib.load("models/dbscan_pollution_anomaly.pkl")
        }
    except Exception as e:
        st.error(f"Error loading data or models: {str(e)}")
        st.stop()

    st.markdown("")

    # Pollution Index Prediction
    if objective == "Pollution Index Prediction":
        st.markdown("### 🌡️ Pollution Index Prediction")
        st.write("Enter pollutant concentrations to predict the overall air quality index.")
        
        st.markdown("")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            pm10 = st.number_input("PM10 (μg/m³)", min_value=0.0, max_value=500.0, value=50.0, 
                                   help="Particulate Matter < 10μm")
        with col2:
            pm25 = st.number_input("PM2.5 (μg/m³)", min_value=0.0, max_value=500.0, value=30.0, 
                                   help="Particulate Matter < 2.5μm")
        with col3:
            no2 = st.number_input("NO2 (μg/m³)", min_value=0.0, max_value=500.0, value=20.0, 
                                  help="Nitrogen Dioxide")
        
        st.markdown("")
        
        if st.button("🔍 Calculate Pollution Index", use_container_width=True):
            pred = models["xgb"].predict(np.array([[pm10, pm25, no2]]))[0]
            
            # Determine severity
            if pred < 50:
                severity_color = "#28a745"
                severity_text = "Good"
            elif pred < 100:
                severity_color = "#ffc107"
                severity_text = "Moderate"
            elif pred < 150:
                severity_color = "#fd7e14"
                severity_text = "Unhealthy for Sensitive Groups"
            else:
                severity_color = "#dc3545"
                severity_text = "Unhealthy"
            
            st.markdown(f"""
            <div class="success-box">
                <h3>Pollution Index: {pred:.2f}</h3>
                <p style="color: {severity_color}; font-weight: 600; font-size: 1.1rem;">
                    Air Quality: {severity_text}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Create visualization
            fig = go.Figure(data=[
                go.Bar(
                    x=["PM10", "PM2.5", "NO2"],
                    y=[pm10, pm25, no2],
                    marker_color=["#0068c9", "#9d4edd", "#06d6a0"],
                    text=[f"{pm10:.1f}", f"{pm25:.1f}", f"{no2:.1f}"],
                    textposition="outside"
                )
            ])
            
            fig.update_layout(
                title="Pollutant Concentrations",
                xaxis_title="Pollutant Type",
                yaxis_title="Concentration (μg/m³)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

    # City Severity Classification
    elif objective == "City Severity Classification":
        st.markdown("### 🏙️ City Severity Classification")
        st.write("Analyze pollution severity for any city in the database.")

        col1, col2 = st.columns(2)
        with col1:
            country_input = st.text_input("Country Name", placeholder="e.g., India, United States")
        with col2:
            city_input = st.text_input("City Name", placeholder="e.g., Delhi, New York")

        st.markdown("")

        if st.button("🔍 Analyze City", use_container_width=True):
            if not country_input or not city_input:
                st.warning("⚠️ Please enter both country and city names")
            else:
                # Ensure the columns are strings
                df["country_name"] = df["country_name"].fillna("").astype(str)
                df["city"] = df["city"].fillna("").astype(str)

                # Filter dynamically
                row = df[(df["country_name"].str.lower() == country_input.lower()) &
                         (df["city"].str.lower() == city_input.lower())]

                if not row.empty:
                    # Model prediction
                    cluster = int(models["kmeans"].predict(
                        row[["pm10_concentration", "pm25_concentration", "no2_concentration", 
                          "pollution_per_person"]]
                    )[0])

                    severity = models["severity_map"][cluster]

                    severity_colors = {
                        "Low": "#28a745",
                        "Moderate": "#ffc107",
                        "High": "#fd7e14",
                        "Critical": "#dc3545"
                    }
                    severity_color = severity_colors.get(severity, "#6c757d")

                    st.markdown(f"""
                    <div class="success-box">
                        <h3>{row['city'].values[0]}, {row['country_name'].values[0]}</h3>
                        <h2 style="color: {severity_color}; margin: 0.5rem 0;">Severity: {severity}</h2>
                        <p>
                            PM10: {row['pm10_concentration'].values[0]:.1f} μg/m³ | 
                            PM2.5: {row['pm25_concentration'].values[0]:.1f} μg/m³ | 
                            NO2: {row['no2_concentration'].values[0]:.1f} μg/m³
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    st.warning(f"⚠️ No data found for {city_input.title()}, {country_input.title()}")
               
    # Anomaly Detection
    elif objective == "Anomaly Detection":
        try:
            anomaly_df = pd.read_csv("data/processed/processed_with_anomalies.csv")
        except:
            st.error("Anomaly data file not found")
            st.stop()

        st.markdown("### 🚨 Anomaly Detection")
        st.write("Identify and visualize pollution hotspots with unusual patterns.")
        
        st.markdown("")

        anomalies = anomaly_df[anomaly_df["is_anomaly_dbscan"] == True]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", f"{len(anomaly_df):,}")
        with col2:
            st.metric("Anomalies", len(anomalies))
        with col3:
            rate = (len(anomalies) / len(anomaly_df)) * 100 if len(anomaly_df) > 0 else 0
            st.metric("Anomaly Rate", f"{rate:.1f}%")

        st.markdown("")

        # Map
        m = folium.Map(
            location=[anomaly_df["latitude"].mean(), anomaly_df["longitude"].mean()],
            zoom_start=3,
            tiles="CartoDB positron"
        )

        for _, r in anomaly_df.iterrows():
            color = "#dc3545" if r["is_anomaly_dbscan"] else "#0068c9"
            folium.CircleMarker(
                location=[r["latitude"], r["longitude"]],
                radius=4,
                color=color,
                fill=True,
                fillOpacity=0.6,
                popup=f"""
                <b>{r['city']}</b><br>
                Index: {r['pollution_index']:.1f}<br>
                Anomaly: {'Yes' if r['is_anomaly_dbscan'] else 'No'}
                """
            ).add_to(m)

        st_folium(m, height=500)

        st.markdown("""
        <div class="info-box">
        <strong>Legend:</strong> 🔵 Normal Pollution Levels | 🔴 Anomalous Pollution Hotspots
        </div>
        """, unsafe_allow_html=True)

    # Yearly Trend Forecast
    elif objective == "Yearly Trend Forecast":
        st.markdown("### 📈 Trend Analysis")
        st.write("Explore historical pollution trends and predictions for specific cities.")
        
        st.markdown("")
        
        city = st.selectbox("Select City", sorted(df["city"].unique()))
        
        city_df = df[df["city"] == city].sort_values("year")
        
        if not city_df.empty:
            # Line chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=city_df["year"],
                y=city_df["pollution_index"],
                mode='lines+markers',
                name='Actual',
                line=dict(color='#0068c9', width=3),
                marker=dict(size=8)
            ))
            
            if "predicted_pollution_index" in city_df.columns:
                fig.add_trace(go.Scatter(
                    x=city_df["year"],
                    y=city_df["predicted_pollution_index"],
                    mode='lines+markers',
                    name='Predicted',
                    line=dict(color='#ff6b6b', width=3, dash='dash'),
                    marker=dict(size=8)
                ))
            
            fig.update_layout(
                title=f"Pollution Trend: {city}",
                xaxis_title="Year",
                yaxis_title="Pollution Index",
                height=450,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("")
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Average", f"{city_df['pollution_index'].mean():.2f}")
            with col2:
                st.metric("Maximum", f"{city_df['pollution_index'].max():.2f}")
            with col3:
                st.metric("Minimum", f"{city_df['pollution_index'].min():.2f}")
            with col4:
                st.metric("Std Dev", f"{city_df['pollution_index'].std():.2f}")

elif page == "Analytics":
    st.title("📊 Analytics & Model Insights")
    st.write("Comprehensive pollution trends, anomaly analysis, and model-driven insights")

    # -------------------- Load Data --------------------
    try:
        df = pd.read_csv("data/processed/processed_data.csv")
        anomaly_df = pd.read_csv("data/processed/processed_with_anomalies.csv")
    except:
        st.error("Unable to load analytics data")
        st.stop()

    # -------------------- Key Statistics --------------------
    st.markdown("### 🔢 Key Statistics")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Cities Monitored", df["city"].nunique())
    with col2:
        st.metric("Countries", df["country_name"].nunique())
    with col3:
        st.metric("Total Anomalies", anomaly_df["is_anomaly_dbscan"].sum())

    # -------------------- Executive Insights --------------------
    st.markdown("### 🧠 Executive Insights")

    avg_index = df["pollution_index"].mean()
    worst_city = df.groupby("city")["pollution_index"].mean().idxmax()
    worst_country = df.groupby("country_name")["pollution_index"].mean().idxmax()
    anomaly_rate = (anomaly_df["is_anomaly_dbscan"].sum() / len(anomaly_df)) * 100

    st.info(
        f"""
        • 🏙️ **{worst_city}** is the **most polluted city** based on long-term averages.

        • 🌍 **{worst_country}** shows the **highest national pollution burden**.

        • 🚨 **{anomaly_rate:.2f}% of data points are anomalies**, indicating abnormal pollution spikes.
        """
    )

    st.markdown("---")

    # -------------------- Global Trend --------------------
    st.markdown("### Global Pollution Trend")

    yearly_data = df.groupby("year")["pollution_index"].mean().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yearly_data["year"],
        y=yearly_data["pollution_index"],
        mode="lines+markers",
        line=dict(color="#0068c9", width=3),
        fill="tozeroy",
        fillcolor="rgba(0,104,201,0.15)"
    ))

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Average Pollution Index",
        height=360,
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.success(
        """
        **Trend Insight:**  
        Pollution levels exhibit persistent fluctuations rather than random noise, 
        suggesting influence from **urban growth, industrialization, and regulatory changes**.
        """
    )

    st.markdown("---")

    # -------------------- City-Level Analysis --------------------
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🏙️ Most Polluted Cities")

        top_cities = (
            df.groupby("city")["pollution_index"]
            .mean()
            .nlargest(10)
            .reset_index()
        )

        fig = go.Figure(go.Bar(
            x=top_cities["pollution_index"],
            y=top_cities["city"],
            orientation="h",
            marker_color="#dc3545",
            text=top_cities["pollution_index"].round(1),
            textposition="outside"
        ))

        fig.update_layout(height=420, margin=dict(l=150))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 🚨 Cities with Most Anomalies")

        anomaly_cities = (
            anomaly_df[anomaly_df["is_anomaly_dbscan"] == True]["city"]
            .value_counts()
            .head(10)
            .reset_index()
        )
        anomaly_cities.columns = ["city", "count"]

        fig = go.Figure(go.Bar(
            x=anomaly_cities["count"],
            y=anomaly_cities["city"],
            orientation="h",
            marker_color="#fd7e14",
            text=anomaly_cities["count"],
            textposition="outside"
        ))

        fig.update_layout(height=420, margin=dict(l=150))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    # -------------------- Anomaly Summary --------------------
    st.markdown("### 🚨 Anomaly Detection Summary")

    normal_count = anomaly_df["is_anomaly_dbscan"].value_counts().get(False, 0)
    anomaly_count = anomaly_df["is_anomaly_dbscan"].value_counts().get(True, 0)

    col1, col2, col3 = st.columns(3)
    col1.metric("Normal Samples", f"{normal_count:,}")
    col2.metric("Anomalies Detected", anomaly_count)
    col3.metric("Detection Rate", f"{anomaly_rate:.2f}%")

    fig = go.Figure(go.Pie(
        labels=["Normal", "Anomaly"],
        values=[normal_count, anomaly_count],
        hole=0.4,
        marker_colors=["#0068c9", "#dc3545"]
    ))

    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    st.warning(
        """
        🚨 **Anomaly Interpretation:**  
        Detected anomalies represent **unexpected pollution spikes** that may result from:
        - Industrial accidents  
        - Seasonal events (crop burning, weather inversions)  
        - Sensor inconsistencies  

        These locations should be **prioritized for investigation**.
        """
    )

    st.markdown("---")

    # -------------------- Model Transparency --------------------
    st.markdown("### Model Insights & Transparency")

    st.markdown(
        """
        **Models used in this system:**
        - **XGBoost Regressor** → Pollution Index prediction  
        - **KMeans Clustering** → Pollution severity categorization  
        - **DBSCAN** → Anomaly detection  

        **Why DBSCAN?**
        - No need to predefine clusters  
        - Effectively isolates true outliers  
        - Well-suited for environmental and spatial datasets
        """
    )

    # -------------------- Key Takeaways --------------------
    st.markdown("### 📌 Key Takeaways")

    st.markdown(
        """
        ✔ Pollution hotspots are **concentrated in urban regions**  
        ✔ Anomalies indicate **non-random pollution events**  
        ✔ Long-term trends show **systemic environmental stress**  
        ✔ Data-driven monitoring enables **proactive policy response**
        """
    )


# Footer
st.markdown("""
<style>
.footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background-color: #0e1117;
    color: #fafafa;
    text-align: center;
    padding: 0.75rem 0;
    font-size: 14px;
    z-index: 9999;
}
.footer a {
    color: #0068c9;
    text-decoration: none;
}
.footer a:hover {
    text-decoration: underline;
}
</style>
<div class="footer">
    © 2025 <strong>AirSense</strong> - Built by <a href="https://linkedin.com/in/anamikaghosh18" target="_blank">Anamika Ghosh</a>. All rights reserved.
</div>
""", unsafe_allow_html=True)