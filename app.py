import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import folium
from streamlit_folium import st_folium
import joblib
import json
from streamlit_lottie import st_lottie
import requests

st.set_page_config(page_title="AirSense", page_icon="🌍", layout="wide")

# --------------------------- CSS ---------------------------
st.markdown("""
<style>
/* Main container padding */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Card styling */
.card {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    margin-bottom: 1.5rem;
    border: 1px solid #e5e7eb;
}

/* Headers */
h1 {
    color: #1f2937;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

h2, h3 {
    color: #374151;
    font-weight: 600;
    margin-top: 1rem;
}

/* Subheader styling */
.subtitle {
    color: #6b7280;
    font-size: 1.1rem;
    margin-bottom: 1.5rem;
}

/* Button styling */
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 6px;
    padding: 0.6rem 1.5rem;
    font-weight: 500;
    border: none;
    transition: all 0.3s ease;
}

.stButton>button:hover {
    background-color: #1d4ed8;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
}

/* Info boxes */
.info-box {
    background: #eff6ff;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #2563eb;
    margin: 1rem 0;
}

.success-box {
    background: #f0fdf4;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #16a34a;
    margin: 1rem 0;
}

/* Sidebar styling */
.css-1d391kg {
    background-color: #f9fafb;
}

/* Input fields */
.stNumberInput>div>div>input, .stTextInput>div>div>input {
    border-radius: 6px;
}

/* Metric styling */
.stMetric {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# --------------------------- Sidebar ---------------------------
with st.sidebar:
    st.image("assets/pollution.jpg", width=140)
    st.title("AirSense")
    st.markdown("---")
    page = st.radio("Navigate", ["🏠 Home", "🔬 Operations", "📊 Dashboard", "📈 Model Analysis"], 
                    label_visibility="collapsed")

# Clean page names
page = page.split(" ", 1)[1] if " " in page else page

# --------------------------- Home Page ---------------------------
if page == "Home":
    st.markdown("""
    <style>
    /* Remove default Streamlit container background */
    .css-1d391kg, .css-1d391kg .css-1offfwp { 
        background-color: transparent !important;
        padding: 0px !important;
    }

    /* Subtitle styling */
    .subtitle {
        font-size: 20px;
        color: #4b5563;
        font-weight: 500;
        margin-bottom: 1rem;
    }

    /* Card Styling */
    .card {
        background: linear-gradient(145deg, #ffffff, #f0f0f0);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        transition: transform 0.3s;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(0,0,0,0.15);
    }

    /* Feature styling */
    .feature-icon {
        font-size: 28px;
        margin-right: 10px;
    }
    .feature-text {
        font-size: 16px;
        font-weight: 500;
        color: #1f2937;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("AirSense 🌍")
    st.markdown('<p class="subtitle">Intelligent Air Pollution Monitoring Platform</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1], gap="large")

    # ------------------ Left Column ------------------
    with col1:
        # Welcome Card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        ### Welcome to AirSense
        
        Monitor, predict, and visualize air pollution trends with **advanced machine learning models**. 
        Get **real-time insights** into air quality across different cities and countries.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        # Key Features Card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### ✨ Key Features")
        
        feature_data = [
            ("🎯", "Multi-objective Predictions"),
            ("📊", "Dynamic Dashboards"),
            ("🚨", "Anomaly Detection"),
            ("🗺️", "Interactive Maps")
        ]
        
        col_a, col_b = st.columns(2)
        for i, col in enumerate([col_a, col_b]):
            for icon, text in feature_data[i*2:i*2+2]:
                col.markdown(f'<p><span class="feature-icon">{icon}</span><span class="feature-text">{text}</span></p>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Hero Image
        st.image("assets/hero.jpg", use_column_width=True, caption="Visualize pollution trends dynamically")

    # ------------------ Right Column ------------------
    with col2:
        import requests
        from streamlit_lottie import st_lottie

        # Load Lottie animation
        def load_lottieurl(url):
            try:
                r = requests.get(url, timeout=5)
                if r.status_code == 200:
                    return r.json()
            except:
                pass
            return None
        
        lottie_air = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_rpC4ik.json")
        if lottie_air:
            st_lottie(lottie_air, height=350, key="home_animation")

# --------------------------- Operations ---------------------------
elif page == "Operations":
    st.title("🔬 Analyze & Predict Pollution")
    st.markdown('<p class="subtitle">Select an analysis objective below</p>', unsafe_allow_html=True)

    objective = st.selectbox("Select Objective", 
                             ["Pollution Index Prediction",
                              "City Severity Classification",
                              "Anomaly Detection",
                              "Yearly Trend Forecast"],
                             label_visibility="collapsed")

    df = pd.read_csv("data/processed/processed_data.csv")
    models = {
        "xgb": joblib.load("models/xgb_pollution_index_model.pkl"),
        "kmeans": joblib.load("models/city_pollution_model.pkl"),
        "severity_map": joblib.load("models/severity_label_map.pkl"),
        "dbscan": joblib.load("models/dbscan_pollution_anomaly.pkl")
    }

    st.markdown('<div class="card">', unsafe_allow_html=True)

    # Pollution Index Prediction
    if objective == "Pollution Index Prediction":
        st.subheader("🌡️ Predict Pollution Index")
        st.markdown("Enter pollutant concentrations to predict the overall pollution index")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            pm10 = st.number_input("PM10 (μg/m³)", 0.0, 500.0, 50.0, help="Particulate Matter 10")
        with col2:
            pm25 = st.number_input("PM2.5 (μg/m³)", 0.0, 500.0, 30.0, help="Particulate Matter 2.5")
        with col3:
            no2 = st.number_input("NO2 (μg/m³)", 0.0, 500.0, 20.0, help="Nitrogen Dioxide")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🔍 Predict Index", use_container_width=True):
            pred = models["xgb"].predict(np.array([[pm10, pm25, no2]]))[0]
            
            st.markdown(f'<div class="success-box"><h3>Predicted Pollution Index: {pred:.2f}</h3></div>', 
                       unsafe_allow_html=True)

            fig = px.bar(x=["PM10", "PM2.5", "NO2"], y=[pm10, pm25, no2],
                         color=["PM10", "PM2.5", "NO2"],
                         color_discrete_sequence=["#2563eb", "#f97316", "#16a34a"],
                         labels={'x': 'Pollutant', 'y': 'Concentration (μg/m³)'},
                         title="Pollutant Concentrations")
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)

    # City Severity Classification
    elif objective == "City Severity Classification":
        st.subheader("🏙️ City Severity Analysis")
        st.markdown("Check the pollution severity level for any city")
        
        col1, col2 = st.columns(2)
        with col1:
            country = st.text_input("Country", placeholder="e.g., India")
        with col2:
            city = st.text_input("City", placeholder="e.g., Delhi")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🔍 Check Severity", use_container_width=True):
            row = df[(df["country_name"].str.lower() == country.lower()) & 
                    (df["city"].str.lower() == city.lower())]
            
            if not row.empty:
                cluster = models["kmeans"].predict(
                    row[["pm10_concentration", "pm25_concentration", "no2_concentration", 
                         "pollution_index", "pollution_per_person"]]
                )[0]
                severity = models["severity_map"][cluster]
                
                st.markdown(f'<div class="success-box"><h3>{city.title()}, {country.title()}</h3><h2>Severity Level: {severity}</h2></div>', 
                           unsafe_allow_html=True)

                fig = px.scatter(df, x="pm10_concentration", y="pm25_concentration", 
                               color="pollution_severity",
                               labels={"pm10_concentration": "PM10 Concentration", 
                                      "pm25_concentration": "PM2.5 Concentration"},
                               title="City Pollution Severity Map",
                               color_discrete_sequence=px.colors.qualitative.Set2)
                
                fig.add_scatter(x=row["pm10_concentration"], y=row["pm25_concentration"], 
                              mode='markers',
                              marker=dict(size=20, color='red', symbol='star', 
                                        line=dict(width=2, color='white')), 
                              name="Selected City")
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"⚠️ No data found for {city}, {country}")

    # Anomaly Detection
    elif objective == "Anomaly Detection":
        st.subheader("🚨 Pollution Hotspots Detection")
        st.markdown("Visualize anomalous pollution patterns across locations")
        
        anomalies = df[df["is_anomaly"] == True]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Anomalies Detected", len(anomalies), 
                     delta=f"{(len(anomalies)/len(df)*100):.1f}%")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        m = folium.Map(location=[df["latitude"].mean(), df["longitude"].mean()], 
                      zoom_start=4, tiles="OpenStreetMap")
        
        for _, r in df.iterrows():
            color = 'red' if r["is_anomaly"] else 'blue'
            folium.CircleMarker(
                [r["latitude"], r["longitude"]], 
                radius=5, 
                color=color, 
                fill=True,
                fillOpacity=0.6,
                popup=f"{r['city']}: {r['pollution_index']:.1f}"
            ).add_to(m)
        
        st_folium(m, width=None, height=500)
        
        st.markdown('<div class="info-box">🔵 Normal Pollution Levels | 🔴 Anomalous Pollution Levels</div>', 
                   unsafe_allow_html=True)

    # Yearly Trend Forecast
    elif objective == "Yearly Trend Forecast":
        st.subheader("📈 Pollution Trend Analysis")
        st.markdown("Explore historical and predicted pollution trends")
        
        city = st.selectbox("Select City", sorted(df["city"].unique()))
        
        city_df = df[df["city"] == city].sort_values("year")
        
        if not city_df.empty:
            fig = px.line(city_df, x="year", 
                         y=["pollution_index", "predicted_pollution_index"],
                         labels={"value": "Pollution Index", "variable": "Type", "year": "Year"},
                         title=f"Pollution Trend for {city}",
                         color_discrete_sequence=["#2563eb", "#f97316"])
            
            fig.update_layout(height=450, hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Pollution Index", f"{city_df['pollution_index'].mean():.2f}")
            with col2:
                st.metric("Max Recorded", f"{city_df['pollution_index'].max():.2f}")
            with col3:
                st.metric("Min Recorded", f"{city_df['pollution_index'].min():.2f}")

    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------- Dashboard ---------------------------
elif page == "Dashboard":
    st.title("📊 Pollution Dashboard")
    st.markdown('<p class="subtitle">Overview of pollution trends and statistics</p>', unsafe_allow_html=True)
    
    df = pd.read_csv("data/processed/processed_data.csv")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Cities Monitored", df["city"].nunique())
    with col2:
        st.metric("Countries", df["country_name"].nunique())
    with col3:
        st.metric("Avg Pollution Index", f"{df['pollution_index'].mean():.2f}")
    with col4:
        st.metric("Anomalies", df["is_anomaly"].sum())
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📈 Yearly Pollution Trend")
    yearly_data = df.groupby("year")["pollution_index"].mean().reset_index()
    fig = px.line(yearly_data, x="year", y="pollution_index", 
                  labels={"pollution_index": "Average Pollution Index", "year": "Year"},
                  markers=True)
    fig.update_traces(line_color='#2563eb', line_width=3)
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🚨 Top Cities with Anomalies")
    anomaly_cities = df[df["is_anomaly"] == True]["city"].value_counts().head(10)
    fig = px.bar(x=anomaly_cities.index, y=anomaly_cities.values,
                 labels={"x": "City", "y": "Anomaly Count"},
                 color=anomaly_cities.values,
                 color_continuous_scale="Reds")
    fig.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------- Model Analysis ---------------------------
elif page == "Model Analysis":
    st.title("📈 Model Performance Analysis")
    st.markdown('<p class="subtitle">Evaluate model accuracy and insights</p>', unsafe_allow_html=True)
    
    df = pd.read_csv("data/processed/processed_data.csv")
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🎯 XGBoost: Predicted vs Actual")
    st.image("assets/xgb_pred_vs_actual.png", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔍 Feature Importance Analysis")
    st.image("assets/xgb_feature_importance.png", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🚨 DBSCAN Anomaly Detection Stats")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Normal Samples", df["is_anomaly"].value_counts().get(False, 0))
    with col2:
        st.metric("Anomalies", df["is_anomaly"].value_counts().get(True, 0))
    
    # Anomaly distribution
    fig = px.pie(values=df["is_anomaly"].value_counts().values,
                 names=["Normal", "Anomaly"],
                 title="Anomaly Distribution",
                 color_discrete_sequence=["#2563eb", "#ef4444"])
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)