# tabs/forecast.py
import streamlit as st
import pandas as pd
from functions.forecast_client import call_lambda, extract_summary_and_series, DEFAULT_DAYS

def forecast(st, tab_fore):
    with tab_fore:
        st.subheader("EOM / EOQ Forecast")

        c1, c2 = st.columns([1, 3])
        with c1:
            days = st.number_input("Horizon (days)", min_value=7, max_value=365,
                                   value=DEFAULT_DAYS, step=1)
            fetch = st.button("📈 Fetch forecast", width="stretch")
        # Removed caption text

        if not fetch:
            st.info("Set a horizon and click **Fetch forecast**.")
            return

        try:
            raw = call_lambda(days=int(days))
        except Exception as e:
            st.error(f"Failed to invoke forecasting Lambda: {e}")
            return

        summary, df = extract_summary_and_series(raw)

        m1, m2, m3 = st.columns(3)
        # Added dollar sign + commas for better readability
        m1.metric("Daily Average (USD)",
                  f"${summary.get('daily_avg_usd'):,.2f}" if summary.get('daily_avg_usd') is not None else "—")
        m2.metric("EOM (USD)",
                  f"${summary.get('eom_usd'):,.2f}" if summary.get('eom_usd') is not None else "—")
        m3.metric("EOQ (USD)",
                  f"${summary.get('eoq_usd'):,.2f}" if summary.get('eoq_usd') is not None else "—")

        if df.empty:
            st.info("No forecast time series returned.")
            return

        # Normalize, sort, and slice to selected horizon
        df_plot = df.copy()
        df_plot["date"] = pd.to_datetime(df_plot["date"])
        df_plot = df_plot.sort_values("date").head(int(days)).set_index("date")

        # Render area chart
        st.area_chart(df_plot["cost_usd"])

# Back-compat alias for imports
forecast_tab = forecast
