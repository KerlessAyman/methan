import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ---------- Page Config ----------
st.set_page_config(
    page_title="Methane Prediction",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Custom CSS ----------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1b4332;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #52796f;
        margin-top: 0;
    }
    .metric-card {
        background-color: #f0f7f4;
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #d8e8e0;
    }
    .stButton>button {
        background-color: #2d6a4f;
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1b4332;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Load Model ----------
@st.cache_resource
def load_model():
    return joblib.load("methane_model.pkl")

model = load_model()

# ---------- Header ----------
st.markdown('<p class="main-title">🌍 Methane Emissions Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict methane levels from population and GDP indicators</p>', unsafe_allow_html=True)
st.divider()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("ℹ️ About the Model")
    st.write(
        "This model was trained using **Linear Regression** on "
        "`population` and `gdp` to predict `methane` emissions."
    )
    st.warning(
        "⚠️ The training data covers only **18-19 countries** "
        "(country-level values). Predictions for values far outside "
        "that range may not be reliable."
    )
    st.divider()
    st.caption("Model: Linear Regression")
    st.caption("Features: population, gdp")

# ---------- Tabs ----------
tab1, tab2 = st.tabs(["🔮 Prediction", "📊 Visualization"])

# ===================== TAB 1: PREDICTION =====================
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Values")
        population = st.number_input(
            "Population", min_value=0.0, value=1_000_000.0, step=1000.0, format="%.0f"
        )
        gdp = st.number_input(
            "GDP", min_value=0.0, value=100_000_000.0, step=1_000_000.0, format="%.0f"
        )
        predict_btn = st.button("Predict Methane", use_container_width=True)

    with col2:
        st.subheader("Result")
        if predict_btn:
            input_data = np.array([[population, gdp]])
            prediction = model.predict(input_data)[0]

            st.markdown(
                f"""
                <div class="metric-card">
                    <h4>Predicted Methane</h4>
                    <h2 style="color:#1b4332;">{prediction:,.4f}</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )

            m1, m2 = st.columns(2)
            m1.metric("Population", f"{population:,.0f}")
            m2.metric("GDP", f"{gdp:,.0f}")
        else:
            st.info("👈 Enter values and click **Predict Methane** to see the result.")

# ===================== TAB 2: VISUALIZATION =====================
with tab2:
    st.subheader("Data Insights & Model Behavior")

    st.write(
        "Upload your full dataset (the same CSV used for the project) to explore "
        "emissions patterns. If no file is uploaded, the charts below show how "
        "predictions change as population/gdp vary."
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df_viz = pd.read_csv(uploaded_file)
        df_viz.columns = [c.strip() for c in df_viz.columns]

        # ---- Filters ----
        with st.expander("🔎 Filters", expanded=False):
            fcol1, fcol2, fcol3 = st.columns(3)
            countries = sorted(df_viz["iso3"].dropna().unique()) if "iso3" in df_viz.columns else []
            subsectors = sorted(df_viz["subsector"].dropna().unique()) if "subsector" in df_viz.columns else []
            source_types = sorted(df_viz["source_type"].dropna().unique()) if "source_type" in df_viz.columns else []

            sel_countries = fcol1.multiselect("Country (iso3)", countries, default=countries)
            sel_subsectors = fcol2.multiselect("Subsector", subsectors, default=subsectors)
            sel_source = fcol3.multiselect("Source type", source_types, default=source_types)

        f_df = df_viz.copy()
        if sel_countries:
            f_df = f_df[f_df["iso3"].isin(sel_countries)]
        if sel_subsectors:
            f_df = f_df[f_df["subsector"].isin(sel_subsectors)]
        if sel_source:
            f_df = f_df[f_df["source_type"].isin(sel_source)]

        # ---- KPI row ----
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Records", f"{len(f_df):,}")
        if "emission_quantity" in f_df.columns:
            k2.metric("Total Emissions", f"{f_df['emission_quantity'].sum():,.0f}")
            k3.metric("Avg Emissions", f"{f_df['emission_quantity'].mean():,.2f}")
        if "known_super_emitter_flag" in f_df.columns:
            super_count = f_df["known_super_emitter_flag"].astype(str).str.upper().eq("TRUE").sum()
            k4.metric("Super Emitters", f"{super_count:,}")

        st.divider()

        # ---- Emissions by subsector ----
        if {"subsector", "emission_quantity"}.issubset(f_df.columns):
            sub_agg = f_df.groupby("subsector", as_index=False)["emission_quantity"].sum().sort_values(
                "emission_quantity", ascending=False
            )
            fig_sub = px.bar(
                sub_agg, x="subsector", y="emission_quantity",
                title="Total Emissions by Subsector",
                color="emission_quantity", color_continuous_scale="Greens",
            )
            st.plotly_chart(fig_sub, use_container_width=True)

        # ---- Emissions by country ----
        if {"iso3", "emission_quantity"}.issubset(f_df.columns):
            country_agg = f_df.groupby("iso3", as_index=False)["emission_quantity"].sum().sort_values(
                "emission_quantity", ascending=False
            )
            fig_country = px.bar(
                country_agg, x="iso3", y="emission_quantity",
                title="Total Emissions by Country",
                color="emission_quantity", color_continuous_scale="Teal",
            )
            st.plotly_chart(fig_country, use_container_width=True)

        # ---- Trend over time ----
        if {"year", "month", "emission_quantity"}.issubset(f_df.columns):
            f_df["period"] = pd.to_datetime(
                f_df["year"].astype(str) + "-" + f_df["month"].astype(str) + "-01", errors="coerce"
            )
            trend = f_df.groupby("period", as_index=False)["emission_quantity"].sum().sort_values("period")
            fig_trend = px.line(
                trend, x="period", y="emission_quantity",
                title="Emissions Trend Over Time", markers=True,
            )
            fig_trend.update_traces(line_color="#2d6a4f")
            st.plotly_chart(fig_trend, use_container_width=True)

        # ---- Source type breakdown ----
        if {"source_type", "emission_quantity"}.issubset(f_df.columns):
            fig_source = px.pie(
                f_df, names="source_type", values="emission_quantity",
                title="Emissions Share by Source Type",
                color_discrete_sequence=px.colors.sequential.Greens_r,
            )
            st.plotly_chart(fig_source, use_container_width=True)

        # ---- Confidence level breakdown ----
        if "emissions_quantity_confidence" in f_df.columns:
            conf_counts = f_df["emissions_quantity_confidence"].value_counts().reset_index()
            conf_counts.columns = ["confidence", "count"]
            fig_conf = px.bar(
                conf_counts, x="confidence", y="count",
                title="Records by Confidence Level",
                color="confidence", color_discrete_sequence=px.colors.sequential.Emrld,
            )
            st.plotly_chart(fig_conf, use_container_width=True)

        # ---- Map ----
        if {"lat", "lon", "emission_quantity"}.issubset(f_df.columns):
            map_df = f_df.dropna(subset=["lat", "lon", "emission_quantity"]).copy()
            map_df = map_df[map_df["emission_quantity"] >= 0]

            if map_df.empty:
                st.info("No valid lat/lon/emission_quantity rows to display on the map.")
            else:
                max_points = st.slider(
                    "Max points on map (lower = faster)",
                    min_value=100, max_value=5000, value=500, step=100,
                )
                if len(map_df) > max_points:
                    map_df = map_df.sample(max_points, random_state=42)
                    st.caption(f"Showing a random sample of {max_points:,} out of {len(f_df):,} rows for performance.")

                with st.spinner("Rendering map..."):
                    fig_map = px.scatter_geo(
                        map_df, lat="lat", lon="lon", size="emission_quantity",
                        color="subsector" if "subsector" in map_df.columns else None,
                        hover_name="facility_name" if "facility_name" in map_df.columns else None,
                        title="Facility Locations (sized by emissions)",
                        projection="natural earth",
                        opacity=0.7,
                    )
                    fig_map.update_layout(height=500, margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig_map, use_container_width=True)

        # ---- Actual vs Predicted methane (if model features present) ----
        if {"population", "gdp", "methane"}.issubset(f_df.columns):
            st.divider()
            st.markdown("##### Methane Model: Actual vs Predicted")
            m_df = f_df.dropna(subset=["population", "gdp", "methane"]).drop_duplicates(
                subset=["population", "gdp", "methane"]
            )
            m_df["predicted_methane"] = model.predict(m_df[["population", "gdp"]])
            fig1 = px.scatter(
                m_df, x="methane", y="predicted_methane",
                title="Actual vs Predicted Methane (unique country values)",
                color_discrete_sequence=["#2d6a4f"],
            )
            fig1.add_shape(
                type="line",
                x0=m_df["methane"].min(), y0=m_df["methane"].min(),
                x1=m_df["methane"].max(), y1=m_df["methane"].max(),
                line=dict(color="red", dash="dash"),
            )
            st.plotly_chart(fig1, use_container_width=True)

    else:
        st.markdown("##### Predicted Methane vs Population (GDP fixed at median)")
        pop_range = np.linspace(0, 5_000_000, 50)
        fixed_gdp = 100_000_000.0
        preds_pop = model.predict(np.column_stack([pop_range, np.full_like(pop_range, fixed_gdp)]))

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=pop_range, y=preds_pop, mode="lines", line=dict(color="#2d6a4f", width=3)))
        fig3.update_layout(xaxis_title="Population", yaxis_title="Predicted Methane")
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("##### Predicted Methane vs GDP (Population fixed at median)")
        gdp_range = np.linspace(0, 500_000_000, 50)
        fixed_pop = 1_000_000.0
        preds_gdp = model.predict(np.column_stack([np.full_like(gdp_range, fixed_pop), gdp_range]))

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=gdp_range, y=preds_gdp, mode="lines", line=dict(color="#40916c", width=3)))
        fig4.update_layout(xaxis_title="GDP", yaxis_title="Predicted Methane")
        st.plotly_chart(fig4, use_container_width=True)

st.divider()
st.caption("Built with Streamlit • Linear Regression Model")