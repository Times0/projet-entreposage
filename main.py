import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

@st.cache_data
def load_data(dataset_path):
    return pd.read_csv(dataset_path, delimiter="\t", encoding="utf-16")

dataset_path = "dataset/Physical dataset/phy_att_2.csv"
df = load_data(dataset_path)

st.title("Physical Dataset")

st.write(df)

st.write("## Scatter Plot") 
"""
Time	Tank_1	Tank_2	Tank_3	Tank_4	Tank_5	Tank_6	Tank_7	Tank_8	Pump_1	Pump_2	Pump_3	Pump_4	Pump_5	Pump_6	Flow_sensor_1	Flow_sensor_2	Flow_sensor_3	Flow_sensor_4	Valv_1	Valv_2	Valv_3	Valv_4	Valv_5	Valv_6	Valv_7	Valv_8	Valv_9	Valv_10	Valv_11	Valv_12	Valv_13	Valv_14	Valv_15	Valv_16	Valv_17	Valv_18	Valv_19	Valv_20	Valv_21	Valv_22	Lable_n	Label
19/04/2021 15:37:12	0	0	0	0	0	0	0	0	false	false	false	false	false	false	0	0	0	0	false	false	false	false	false	false	false	false	false	false	false	false	false	false	false	false	false	false	false	false	false	false	0	normal
19/04/2021 15:37:13	0	0	0	0	0	0	0	0	false	false	false	false	false	false	0	0	0	0	false	false	false	false	false	false	false	false	false	false	false	false	false	false	false	false	false	false	false	false	false	false	0	normal
"""

def create_tank_plot(df):
    fig = go.Figure()

    for tank in ["Tank_1", "Tank_2", "Tank_3", "Tank_4", "Tank_5", "Tank_6", "Tank_7", "Tank_8"]:
        fig.add_trace(go.Scatter(x=df["Time"], y=df[tank], mode="lines", name=tank))
    
    fig.update_layout(
        title="Tank Capacity Over Time with Attack Highlights",
        xaxis_title="Time",
        yaxis_title="Tank Capacity",
        legend_title="Tanks"
    )

    return fig

# exploratory data analysis
fig = create_tank_plot(df)
st.plotly_chart(fig)
