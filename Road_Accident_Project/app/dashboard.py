import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

from preprocessing import load_data
from analysis import accident_stats


# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="Road Accident Analytics Platform",
    page_icon="🚗",
    layout="wide"
)


# -----------------------------
# CSS STYLE
# -----------------------------

st.markdown("""
<style>

.stApp{
background-color:#f5f7fb;
}

section[data-testid="stSidebar"]{
background-color:#e8eef7;
}

.kpi-card{
background:#ffffff;
padding:20px;
border-radius:12px;
box-shadow:0px 4px 10px rgba(0,0,0,0.1);
text-align:center;
transition:0.3s;
}

.kpi-card:hover{
transform:scale(1.05);
}

.kpi-title{
font-size:18px;
color:#555;
}

.kpi-value{
font-size:32px;
font-weight:bold;
color:#1f4e79;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# TITLE
# -----------------------------

st.title("🚗 Road Accident Analytics Platform")
st.write("Interactive dashboard for analyzing accident patterns and risk factors.")


# -----------------------------
# LOAD DATA
# -----------------------------

df = load_data()


# -----------------------------
# FORECAST MODEL
# -----------------------------

year_counts = df.groupby("Year")["Accidents"].sum().reset_index()

X = year_counts["Year"].values.reshape(-1,1)
y = year_counts["Accidents"]

model = LinearRegression()
model.fit(X,y)

max_year = year_counts["Year"].max()

if max_year < 2026:

    future_years = np.arange(max_year+1,2027)

    pred = model.predict(future_years.reshape(-1,1))

    forecast = pd.DataFrame({
        "Year":future_years,
        "Accidents":pred
    })

    trend_df = pd.concat([year_counts,forecast])

else:
    trend_df = year_counts


# -----------------------------
# SIDEBAR FILTERS
# -----------------------------

st.sidebar.header("Dashboard Filters")

year = st.sidebar.selectbox(
    "Select Year",
    sorted(df["Year"].unique())
)

state_filter = st.sidebar.multiselect(
    "State",
    sorted(df["State"].unique())
)

weather_filter = st.sidebar.multiselect(
    "Weather",
    sorted(df["Weather_Condition"].unique())
)

vehicle_filter = st.sidebar.multiselect(
    "Vehicle",
    sorted(df["Vehicle_Type"].unique())
)


# -----------------------------
# APPLY FILTERS
# -----------------------------

filtered = df[df["Year"] == year]

if state_filter:
    filtered = filtered[filtered["State"].isin(state_filter)]

if weather_filter:
    filtered = filtered[filtered["Weather_Condition"].isin(weather_filter)]

if vehicle_filter:
    filtered = filtered[filtered["Vehicle_Type"].isin(vehicle_filter)]


# -----------------------------
# KPI METRICS
# -----------------------------

total_accidents, total_casualties, avg_casualties = accident_stats(filtered)

col1,col2,col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
    <div class="kpi-title">Total Accidents</div>
    <div class="kpi-value">{total_accidents}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
    <div class="kpi-title">Total Casualties</div>
    <div class="kpi-value">{total_casualties}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
    <div class="kpi-title">Average Casualties</div>
    <div class="kpi-value">{round(avg_casualties,2)}</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()


# -----------------------------
# WEATHER IMPACT
# -----------------------------

col1,col2 = st.columns(2)

with col1:

    weather = filtered.groupby("Weather_Condition")["Accidents"].sum().reset_index()

    fig_weather = px.bar(
        weather,
        x="Weather_Condition",
        y="Accidents",
        color="Accidents",
        text="Accidents",
        title=f"Weather Impact ({year})"
    )

    st.plotly_chart(fig_weather,use_container_width=True)


# -----------------------------
# VEHICLE RISK
# -----------------------------

with col2:

    vehicle = filtered.groupby("Vehicle_Type")["Accidents"].sum().reset_index()

    fig_vehicle = px.bar(
        vehicle,
        x="Vehicle_Type",
        y="Accidents",
        color="Accidents",
        text="Accidents",
        title=f"Vehicle Risk ({year})"
    )

    st.plotly_chart(fig_vehicle,use_container_width=True)

st.divider()


# -----------------------------
# DRIVER AGE RISK
# -----------------------------

col1,col2 = st.columns(2)

with col1:

    age = filtered.groupby("Driver_Age_Group")["Accidents"].sum().reset_index()

    fig_age = px.bar(
        age,
        x="Driver_Age_Group",
        y="Accidents",
        color="Accidents",
        text="Accidents",
        title="Driver Age Risk"
    )

    st.plotly_chart(fig_age,use_container_width=True)


# -----------------------------
# ROAD TYPE DISTRIBUTION
# -----------------------------

with col2:

    road = filtered.groupby("Road_Type")["Accidents"].sum().reset_index()

    fig_road = px.pie(
        road,
        names="Road_Type",
        values="Accidents",
        title="Road Type Distribution"
    )

    st.plotly_chart(fig_road,use_container_width=True)

st.divider()


# -----------------------------
# TOP 10 DANGEROUS STATES
# -----------------------------

st.subheader("Top 10 Dangerous States")

danger_states = df.groupby("State")["Accidents"].sum().sort_values(ascending=False).head(10).reset_index()

fig_states = px.bar(
    danger_states,
    x="State",
    y="Accidents",
    color="Accidents",
    text="Accidents",
    title="Top Accident Prone States"
)

st.plotly_chart(fig_states,use_container_width=True)

st.divider()


# -----------------------------
# WEATHER + VEHICLE COMBINATION
# -----------------------------

st.subheader("Dangerous Weather + Vehicle Combination")

combo = df.groupby(["Weather_Condition","Vehicle_Type"])["Accidents"].sum().reset_index()

fig_combo = px.sunburst(
    combo,
    path=["Weather_Condition","Vehicle_Type"],
    values="Accidents",
    title="Accident Risk Combination"
)

st.plotly_chart(fig_combo,use_container_width=True)

st.divider()


# -----------------------------
# ACCIDENT TREND FORECAST
# -----------------------------

st.subheader("Accident Trend Forecast (2005–2026)")

fig_trend = px.line(
    trend_df,
    x="Year",
    y="Accidents",
    markers=True,
    title="Accident Trend Forecast"
)

st.plotly_chart(fig_trend,use_container_width=True)

st.divider()


# -----------------------------
# ACCIDENT RISK PREDICTOR
# -----------------------------

st.subheader("🚨 Accident Risk Predictor")

weather_input = st.selectbox("Weather Condition",df["Weather_Condition"].unique())
vehicle_input = st.selectbox("Vehicle Type",df["Vehicle_Type"].unique())
road_input = st.selectbox("Road Type",df["Road_Type"].unique())

risk_data = df[
(df["Weather_Condition"]==weather_input) &
(df["Vehicle_Type"]==vehicle_input) &
(df["Road_Type"]==road_input)
]

if risk_data.empty:
    st.info("No historical accident data available for these conditions.")
else:

    risk_score = risk_data["Accidents"].sum()

    avg_accidents = df["Accidents"].mean()
    high_threshold = avg_accidents * 2
    moderate_threshold = avg_accidents

    st.write("Accident Risk Score:", int(risk_score))

    if risk_score > high_threshold:
        st.error("High Accident Risk ⚠️")
    elif risk_score > moderate_threshold:
        st.warning("Moderate Accident Risk")
    else:
        st.success("Low Accident Risk")

# -----------------------------
# KEY INSIGHTS
# -----------------------------

st.subheader("Key Insights")

top_weather = df.groupby("Weather_Condition")["Accidents"].sum().idxmax()
top_vehicle = df.groupby("Vehicle_Type")["Accidents"].sum().idxmax()
top_state = df.groupby("State")["Accidents"].sum().idxmax()

st.write(f"• Most accidents occur during **{top_weather}** weather.")
st.write(f"• **{top_vehicle}** vehicles are involved in the highest accidents.")
st.write(f"• **{top_state}** is the most accident-prone state.")


st.divider()


# -----------------------------
# DATA PREVIEW
# -----------------------------

st.subheader("Dataset Preview")

st.dataframe(filtered.head(100))


# -----------------------------
# DOWNLOAD DATA
# -----------------------------

st.download_button(
    label="Download Filtered Dataset",
    data=filtered.to_csv(index=False),
    file_name="filtered_accidents.csv",
    mime="text/csv"
)


st.divider()


# -----------------------------
# ABOUT DEVELOPER
# -----------------------------

st.subheader("👨‍💻 About the Developer")

st.markdown("""
**ABHIJIT PAWAR**

📧 Email: abhipawar.9399@gmail.com  

🔗 GitHub  
https://github.com/Apabhi96k  

🔗 LinkedIn  
https://www.linkedin.com/in/abhijit-pawar-3438a925a  

This project demonstrates:

• Data preprocessing  
• Exploratory data analysis  
• Interactive dashboards  
• Machine learning forecasting  
• Accident risk prediction

Built using **Python, Streamlit, Pandas, Plotly and Scikit-learn**
""")