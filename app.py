"""
California housing dashboard (Streamlit).

Run from the project folder:
    streamlit run app.py
"""

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BLUE, CYAN, ORANGE = "#3b82f6", "#06b6d4", "#f97316"
FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
            "Population", "AveOccup", "Latitude", "Longitude"]

st.set_page_config(page_title="California housing", page_icon="🏠", layout="wide")
st.markdown(
    """<style>
    .card {background: linear-gradient(145deg,#131c2e,#0f1829);
           border:1px solid #22304a; border-radius:16px; padding:16px 18px;}
    .t {color:#8fa3c4; font-size:0.85rem;}
    .v {color:#e6edf7; font-size:1.8rem; font-weight:700;}
    .s {color:#06b6d4; font-size:0.85rem;}
    </style>""", unsafe_allow_html=True)
LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
              font=dict(color="#e6edf7"), margin=dict(l=10, r=10, t=30, b=10))


def card(col, title, value, sub=""):
    """Render one KPI card."""
    col.markdown(f'<div class="card"><div class="t">{title}</div>'
                 f'<div class="v">{value}</div><div class="s">{sub}</div></div>',
                 unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load("models/model.joblib")


@st.cache_data
def load_predictions():
    return pd.read_csv("models/test_predictions.csv")


model, df = load_model(), load_predictions()
df["error"] = (df["predicted"] - df["actual"]) * 100_000

st.title("California Housing Prices")
st.caption("Median house value of a district (1990 census), not an individual house. "
           "Gradient boosting, test set of 4,128 districts.")

tab1, tab2, tab3 = st.tabs(["Predict", "Model quality", "Error map"])

# ---------- Predict ----------
with tab1:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("District parameters")
        inc = st.slider("Median income (x $10,000)", 0.5, 15.0, 4.0, 0.1)
        age = st.slider("House age, years", 1, 52, 28)
        rooms = st.slider("Average rooms per house", 2.0, 10.0, 5.4, 0.1)
        beds = st.slider("Average bedrooms per house", 0.8, 3.0, 1.1, 0.05)
        pop = st.slider("Population", 3, 10000, 1400, 50)
        occ = st.slider("Average occupancy", 1.0, 6.0, 3.0, 0.1)
        lat = st.slider("Latitude", 32.5, 42.0, 34.2, 0.05)
        lon = st.slider("Longitude", -124.3, -114.3, -118.5, 0.05)
    row = pd.DataFrame([[inc, age, rooms, beds, pop, occ, lat, lon]], columns=FEATURES)
    price = float(model.predict(row)[0]) * 100_000
    with right:
        st.subheader("Prediction")
        card(st, "Predicted median house value", f"${price:,.0f}",
             "typical error of the model: about $30,000")
        st.caption("Values above $500,000 are capped in the data, so the model "
                   "cannot predict above that level.")
        pts = go.Figure(go.Scatter(x=df["Longitude"], y=df["Latitude"], mode="markers",
                                   marker=dict(size=3, color=df["actual"], colorscale="Viridis",
                                               opacity=0.5), name="Districts"))
        pts.add_trace(go.Scatter(x=[lon], y=[lat], mode="markers", name="Your district",
                                 marker=dict(size=14, color=ORANGE, symbol="x")))
        pts.update_layout(height=380, xaxis_title="Longitude", yaxis_title="Latitude", **LAYOUT)
        st.plotly_chart(pts, width="stretch")

# ---------- Model quality ----------
with tab2:
    mae = df["error"].abs().mean()
    r2 = 1 - ((df["actual"] - df["predicted"]) ** 2).sum() / ((df["actual"] - df["actual"].mean()) ** 2).sum()
    within = (df["error"].abs() <= 50_000).mean()
    cols = st.columns(3)
    card(cols[0], "MAE", f"${mae:,.0f}")
    card(cols[1], "R²", f"{r2:.3f}")
    card(cols[2], "Within $50,000", f"{within:.1%}")
    fig = go.Figure(go.Scatter(x=df["actual"] * 100_000, y=df["predicted"] * 100_000,
                               mode="markers", marker=dict(size=3, color=BLUE, opacity=0.3)))
    fig.add_trace(go.Scatter(x=[0, 520_000], y=[0, 520_000], mode="lines",
                             line=dict(color=ORANGE), name="Perfect"))
    fig.update_layout(height=420, xaxis_title="Actual, $", yaxis_title="Predicted, $",
                      showlegend=False, **LAYOUT)
    st.plotly_chart(fig, width="stretch")
    st.caption("The horizontal band at the right is the capped value ($500,001).")

# ---------- Error map ----------
with tab3:
    limit = st.slider("Show districts with absolute error above, $", 0, 150_000, 50_000, 5_000)
    big = df[df["error"].abs() > limit]
    st.write(f"{len(big)} districts ({len(big) / len(df):.1%})")
    fig = go.Figure(go.Scatter(x=big["Longitude"], y=big["Latitude"], mode="markers",
                               marker=dict(size=5, color=big["error"], colorscale="RdBu",
                                           cmid=0, colorbar=dict(title="Error, $"))))
    fig.update_layout(height=480, xaxis_title="Longitude", yaxis_title="Latitude", **LAYOUT)
    st.plotly_chart(fig, width="stretch")
    st.caption("Red: overpredicted, blue: underpredicted (by the colour scale sign).")