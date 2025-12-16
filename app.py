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

# --------------------------- CSS ---------------------------
st.markdown("""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Global Styles */
* {
    font-family: 'Inter', sans-serif;
}

/* Main container */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* Remove extra padding */
.css-1d391kg {
    padding: 1rem;
}

/* Headers */
h1 {
    color: #111827;
    font-weight: 700;
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
}

h2 {
    color: #1f2937;
    font-weight: 600;
    font-size: 1.5rem;
    margin-top: 2rem;
    margin-bottom: 1rem;
}

h3 {
    color: white;
    font-weight: 600;
    font-size: 1.125rem;
    margin-top: 1.5rem;
    margin-bottom: 0.75rem;
}

/* Subtitle */
.subtitle {
    color: #6b7280;
    font-size: 1.125rem;
    font-weight: 400;
    margin-bottom: 2.5rem;
    line-height: 1.6;
}

/* Feature card */
.feature-card {
    background: transparent;
    padding: 1rem 0;
    border-bottom: 1px solid #f3f4f6;
    margin-bottom: 0;
}

.feature-card:last-child {
    border-bottom: none;
}

/* Feature text */
.feature-item {
    display: flex;
    align-items: center;
    padding: 0.75rem 0;
    color: #374151;
    font-size: 1rem;
}

.feature-icon {
    font-size: 1.5rem;
    margin-right: 0.75rem;
    min-width: 2rem;
}

/* Info boxes */
.info-box {
    background: #f0f9ff;
    padding: 1.25rem;
    border-radius: 8px;
    border-left: 3px solid #3b82f6;
    margin: 1.5rem 0;
    color: #1e40af;
}

.success-box {
    background: #f0fdf4;
    padding: 1.5rem;
    border-radius: 8px;
    border-left: 3px solid #10b981;
    margin: 1.5rem 0;
}

.success-box h3 {
    color: #065f46;
    margin: 0 0 0.5rem 0;
}

.success-box h2 {
    color: #047857;
    margin: 0;
    font-size: 1.5rem;
}

.warning-box {
    background: #fffbeb;
    padding: 1.25rem;
    border-radius: 8px;
    border-left: 3px solid #f59e0b;
    margin: 1.5rem 0;
    color: #92400e;
}

/* Button styling */
.stButton>button {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    color: white;
    border-radius: 8px;
    padding: 0.65rem 1.75rem;
    font-weight: 500;
    border: none;
    font-size: 1rem;
    letter-spacing: 0.01em;
    transition: all 0.2s ease;
}

.stButton>button:hover {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
    transform: translateY(-1px);
}

/* Input fields */
.stNumberInput>div>div>input,
.stTextInput>div>div>input,
.stSelectbox>div>div>div {
    border-radius: 8px;
    border: 1px solid #d1d5db;
    padding: 0.65rem;
    font-size: 0.95rem;
}

.stNumberInput>div>div>input:focus,
.stTextInput>div>div>input:focus {
    border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

/* Metric styling */
[data-testid="stMetricValue"] {
    font-size: 1.75rem;
    font-weight: 700;
    color: #111827;
}

[data-testid="stMetricDelta"] {
    font-size: 0.875rem;
}

.stMetric {
    background: white;
    padding: 1.25rem;
    border-radius: 10px;
    border: 1px solid #e5e7eb;
}


/* Divider */
hr {
    margin: 1.5rem 0;
    border: none;
    border-top: 1px solid #e5e7eb;
}

/* Section header */
.section-header {
    font-size: 0.875rem;
    font-weight: 600;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 1rem;
}

/* Stat item */
.stat-item {
    padding: 0.75rem 0;
    border-bottom: 1px solid #f3f4f6;
}

.stat-item:last-child {
    border-bottom: none;
}

/* Remove streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --------------------------- Sidebar ---------------------------
with st.sidebar:
    st.image("assets/pollution.jpg", width=120)
    st.markdown("### AirSense")
    st.caption("Air Quality Intelligence Platform")
    st.markdown("<hr>", unsafe_allow_html=True)
    
    page = st.radio(
        "Navigation",
        ["🏠 Home", "🔬 Operations", "📊 Dashboard", "📈 Analytics"],
        label_visibility="collapsed"
    )
    
    
# Clean page names
page = page.split(" ", 1)[1] if " " in page else page

# --------------------------- Home Page ---------------------------
if page == "Home":
    st.title("AirSense 🌍")
    st.markdown('<p class="subtitle">Monitor, predict, and visualize air pollution trends with machine learning intelligence</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.5, 1], gap="large")
    
    with col1:
        # Welcome section
        st.markdown("### Welcome to AirSense")
        st.markdown("""
        AirSense provides comprehensive air quality monitoring and predictive analytics 
        for cities worldwide. Leverage advanced machine learning models to understand 
        pollution patterns, detect anomalies, and forecast trends.
        """)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Key capabilities
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
                <div class="feature-item">
                    <span class="feature-icon">{icon}</span>
                    <div>
                        <strong>{title}</strong><br>
                        <span style="color: #6b7280; font-size: 0.9rem;">{desc}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Stats overview
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
                        <span style="color: #6b7280; font-size: 0.95rem;">{emoji} {label}</span>
                        <span style="color: white; font-weight: 600; font-size: 1.25rem;">{value}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        except:
            st.info("Load data to view statistics")
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Quick start
        st.markdown("### Quick Start")
        st.markdown("""
        1. **Navigate** to Operations to run predictions
        2. **Explore** the Dashboard for insights
        3. **Review** Analytics for model performance
        """)

# --------------------------- Operations ---------------------------
elif page == "Operations":
    st.title("🔬 Operations")
    st.markdown('<p class="subtitle">Run predictions and analyze pollution data</p>', unsafe_allow_html=True)

    objective = st.selectbox(
        "Select Analysis Type",
        ["Pollution Index Prediction", "City Severity Classification", "Anomaly Detection", "Yearly Trend Forecast"],
        label_visibility="visible"
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

    st.markdown("<br>", unsafe_allow_html=True)

    # Pollution Index Prediction
    if objective == "Pollution Index Prediction":
        st.markdown("### 🌡️ Pollution Index Prediction")
        st.markdown("Enter pollutant concentrations to predict the overall air quality index.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
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
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🔍 Calculate Pollution Index", use_container_width=True):
            pred = models["xgb"].predict(np.array([[pm10, pm25, no2]]))[0]
            
            # Determine severity
            if pred < 50:
                severity_color = "#10b981"
                severity_text = "Good"
            elif pred < 100:
                severity_color = "#f59e0b"
                severity_text = "Moderate"
            elif pred < 150:
                severity_color = "#f97316"
                severity_text = "Unhealthy for Sensitive Groups"
            else:
                severity_color = "#ef4444"
                severity_text = "Unhealthy"
            
            st.markdown(f"""
            <div class="success-box">
                <h3>Pollution Index: {pred:.2f}</h3>
                <p style="margin: 0; color: {severity_color}; font-weight: 600; font-size: 1.1rem;">
                    Air Quality: {severity_text}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Create visualization
            fig = go.Figure(data=[
                go.Bar(
                    x=["PM10", "PM2.5", "NO2"],
                    y=[pm10, pm25, no2],
                    marker_color=["#1555ba", "#8b5cf6", "#156027"],
                    text=[f"{pm10:.1f}", f"{pm25:.1f}", f"{no2:.1f}"],
                    textposition="outside"
                )
            ])
            
            fig.update_layout(
                title="Pollutant Concentrations",
                xaxis_title="Pollutant Type",
                yaxis_title="Concentration (μg/m³)",
                height=400,
                showlegend=False,
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(family="Inter, sans-serif", size=12),
                margin=dict(t=60, b=60, l=60, r=20)
            )
            
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=True, gridcolor="#f3f4f6")
            
            st.plotly_chart(fig, use_container_width=True)

    # City Severity Classification
    # ---------------- City Severity Classification ----------------
    elif objective == "City Severity Classification":
        st.markdown("### 🏙️ City Severity Classification")
        st.markdown("Analyze pollution severity for any city in the database.")

        col1, col2 = st.columns(2)
        with col1:
            country_input = st.text_input("Country Name", placeholder="e.g., India, United States")
        with col2:
            city_input = st.text_input("City Name", placeholder="e.g., Delhi, New York")

        st.markdown("<br>", unsafe_allow_html=True)

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
                # Model prediction (using numeric columns)
                    cluster = int(models["kmeans"].predict(
                        row[["pm10_concentration", "pm25_concentration", "no2_concentration", 
                          "pollution_per_person"]]
                    )[0])

                    severity = models["severity_map"][cluster]

                    severity_colors = {
                        "Low": "#10b981",
                        "Moderate": "#f59e0b",
                        "High": "#ef4444",
                        "Critical": "#b91c1c"
                    }
                    severity_color = severity_colors.get(severity, "#6b7280")

                    st.markdown(f"""
                    <div class="success-box">
                        <h3>{row['city'].values[0]}, {row['country_name'].values[0]}</h3>
                        <h2 style="color: {severity_color};">Severity: {severity}</h2>
                        <p style="margin: 0.5rem 0 0 0; color: #6b7280;">
                            PM10: {row['pm10_concentration'].values[0]:.1f} | 
                            PM2.5: {row['pm25_concentration'].values[0]:.1f} | 
                            NO2: {row['no2_concentration'].values[0]:.1f}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    st.warning(f"⚠️ No data found for {city_input.title()}, {country_input.title()}")
               

    # Anomaly Detection
        anomaly_df = pd.read_csv("data/processed/processed_with_anomalies.csv")
        st.markdown("### 🚨 Anomaly Detection")
        st.markdown("Identify and visualize pollution hotspots with unusual patterns.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        anomalies = anomaly_df[anomaly_df["is_anomaly"] == True]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Anomalies", len(anomalies))
        with col3:
            st.metric("Anomaly Rate", f"{(len(anomalies)/len(df)*100):.1f}%")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Interactive map
        m = folium.Map(
            location=[df["latitude"].mean(), df["longitude"].mean()], 
            zoom_start=3,
            tiles="CartoDB positron"
        )
        
        for _, r in df.iterrows():
            color = '#ef4444' if r["is_anomaly"] else '#3b82f6'
            folium.CircleMarker(
                location=[r["latitude"], r["longitude"]], 
                radius=4,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.6,
                weight=1,
                popup=folium.Popup(
                    f"<b>{r['city']}</b><br>Index: {r['pollution_index']:.1f}<br>Anomaly: {'Yes' if r['is_anomaly'] else 'No'}",
                    max_width=200
                )
            ).add_to(m)
        
        st_folium(m, width=None, height=500)
        
        st.markdown("""
        <div class="info-box">
            <strong>Legend:</strong> 🔵 Normal Pollution Levels | 🔴 Anomalous Pollution Hotspots
        </div>
        """, unsafe_allow_html=True)

    # Yearly Trend Forecast
    elif objective == "Yearly Trend Forecast":
        st.markdown("### 📈 Trend Analysis")
        st.markdown("Explore historical pollution trends and predictions for specific cities.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        city = st.selectbox("Select City", sorted(df["city"].unique()))
        
        city_df = df[df["city"] == city].sort_values("year")
        
        if not city_df.empty:
            # Line chart elif objective == "Anomaly Detection":
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=city_df["year"],
                y=city_df["pollution_index"],
                mode='lines+markers',
                name='Actual',
                line=dict(color='#3b82f6', width=3),
                marker=dict(size=8)
            ))
            
            if "predicted_pollution_index" in city_df.columns:
                fig.add_trace(go.Scatter(
                    x=city_df["year"],
                    y=city_df["predicted_pollution_index"],
                    mode='lines+markers',
                    name='Predicted',
                    line=dict(color='#f59e0b', width=3, dash='dash'),
                    marker=dict(size=8)
                ))
            
            fig.update_layout(
                title=f"Pollution Trend: {city}",
                xaxis_title="Year",
                yaxis_title="Pollution Index",
                height=450,
                hovermode='x unified',
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(family="Inter, sans-serif"),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            fig.update_xaxes(showgrid=True, gridcolor="#f3f4f6")
            fig.update_yaxes(showgrid=True, gridcolor="#f3f4f6")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Average", f"{city_df['pollution_index'].mean():.2f}")
            with col2:
                st.metric("Maximum", f"{city_df['pollution_index'].max():.2f}")
            with col3:
                st.metric("Minimum", f"{city_df['pollution_index'].min():.2f}")
            with col4:
                st.metric("Std Dev", f"{city_df['pollution_index'].std():.2f}")

# --------------------------- Dashboard ---------------------------
elif page == "Dashboard":
    st.title("📊 Dashboard")
    st.markdown('<p class="subtitle">Comprehensive overview of pollution trends and statistics</p>', unsafe_allow_html=True)
    
    try:
        df = pd.read_csv("data/processed/processed_data.csv")
    except:
        st.error("Unable to load data")
        st.stop()
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Cities Monitored", df["city"].nunique())
    with col2:
        st.metric("Countries", df["country_name"].nunique())
    with col3:
        st.metric("Avg Pollution", f"{df['pollution_index'].mean():.1f}")
    with col4:
        st.metric("Total Anomalies", df["is_anomaly"].sum())
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Yearly Trend
    st.markdown("### 📈 Global Pollution Trend")
    
    yearly_data = df.groupby("year")["pollution_index"].mean().reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yearly_data["year"],
        y=yearly_data["pollution_index"],
        mode='lines+markers',
        line=dict(color='#3b82f6', width=3),
        marker=dict(size=10, color='#3b82f6'),
        fill='tozeroy',
        fillcolor='rgba(59, 130, 246, 0.1)'
    ))
    
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Average Pollution Index",
        height=350,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif"),
        showlegend=False,
        hovermode='x'
    )
    
    fig.update_xaxes(showgrid=True, gridcolor="#f3f4f6")
    fig.update_yaxes(showgrid=True, gridcolor="#f3f4f6")
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Two column layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Top polluted cities
        st.markdown("### 🏙️ Most Polluted Cities")
        
        top_cities = df.groupby("city")["pollution_index"].mean().nlargest(10).reset_index()
        
        fig = go.Figure(go.Bar(
            x=top_cities["pollution_index"],
            y=top_cities["city"],
            orientation='h',
            marker_color='#ef4444',
            text=top_cities["pollution_index"].round(1),
            textposition='outside'
        ))
        
        fig.update_layout(
            xaxis_title="Average Pollution Index",
            yaxis_title="",
            height=400,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Inter, sans-serif"),
            showlegend=False,
            margin=dict(l=150)
        )
        
        fig.update_xaxes(showgrid=True, gridcolor="#f3f4f6")
        fig.update_yaxes(showgrid=False)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Anomaly distribution
        st.markdown("### 🚨 Cities with Most Anomalies")
        
        anomaly_cities = df[df["is_anomaly"] == True]["city"].value_counts().head(10).reset_index()
        anomaly_cities.columns = ["city", "count"]
        
        fig = go.Figure(go.Bar(
            x=anomaly_cities["count"],
            y=anomaly_cities["city"],
            orientation='h',
            marker_color='#f59e0b',
            text=anomaly_cities["count"],
            textposition='outside'
        ))
        
        fig.update_layout(
            xaxis_title="Number of Anomalies",
            yaxis_title="",
            height=400,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Inter, sans-serif"),
            showlegend=False,
            margin=dict(l=150)
        )
        
        fig.update_xaxes(showgrid=True, gridcolor="#f3f4f6")
        fig.update_yaxes(showgrid=False)
        
        st.plotly_chart(fig, use_container_width=True)

# --------------------------- Analytics ---------------------------
elif page == "Analytics":
    st.title("📈 Model Analytics")
    st.markdown('<p class="subtitle">Model performance metrics and insights</p>', unsafe_allow_html=True)
    
    try:
        df = pd.read_csv("data/processed/processed_data.csv")
    except:
        st.error("Unable to load data")
        st.stop()
    
    # Model Performance
    st.markdown("### 🎯 XGBoost Performance")
    st.markdown("Comparison of predicted vs actual pollution index values")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    try:
        st.image("assets/xgb_pred_vs_actual.png", use_container_width=True)
    except:
        st.info("Performance chart not available")
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Feature Importance
    st.markdown("### 🔍 Feature Importance")
    st.markdown("Key factors influencing pollution predictions")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    try:
        st.image("assets/xgb_feature_importance.png", use_container_width=True)
    except:
        st.info("Feature importance chart not available")
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Anomaly Statistics
    st.markdown("### 🚨 Anomaly Detection Statistics")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        normal_count = df["is_anomaly"].value_counts().get(False, 0)
        st.metric("Normal Samples", f"{normal_count:,}")
    with col2:
        anomaly_count = df["is_anomaly"].value_counts().get(True, 0)
        st.metric("Anomalies Detected", anomaly_count)
    with col3:
        anomaly_rate = (anomaly_count / len(df)) * 100 if len(df) > 0 else 0
        st.metric("Detection Rate", f"{anomaly_rate:.2f}%")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Pie chart
    fig = go.Figure(data=[go.Pie(
        labels=["Normal", "Anomaly"],
        values=[normal_count, anomaly_count],
        marker_colors=["#3b82f6", "#ef4444"],
        hole=0.4
    )])
    
    fig.update_layout(
        title="Distribution of Samples",
        height=350,
        font=dict(family="Inter, sans-serif"),
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<style>
.footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background-color: #111827;
    color: white;
    text-align: center;
    padding: 0.75rem 0;
    font-size: 14px;
    z-index: 9999;
}
.footer a {
    color: #2563eb;
    text-decoration: none;
    font-weight: 500;
}
.footer a:hover {
    text-decoration: underline;
}
</style>
<div class="footer">
    © 2025 <strong>AirSense</strong> - Built by <a href="https://linkedin.com/in/anamikaghosh18" target="_blank">Anamika Ghosh</a>. All rights reserved.
</div>
""", unsafe_allow_html=True)
