
from datetime import datetime, timedelta
import pandas as pd
import random

def forecast(st,tab_fore):
    with tab_fore:
        st.subheader("EOM / EOQ Forecast")
        today = datetime.utcnow().date()
        rows = [{"date": (today - timedelta(days=60-i)).isoformat(), "cost_usd": 250 + random.uniform(-25,25)} for i in range(60)]
        df_cost = pd.DataFrame(rows)
        x = df_cost["cost_usd"].mean()
        eom = (datetime(today.year, today.month, 28) + timedelta(days=4)).date().replace(day=1) - timedelta(days=1)
        eoqm = ((today.month - 1)//3 + 1)*3
        eoq = (datetime(today.year, eoqm, 28) + timedelta(days=4)).date().replace(day=1) - timedelta(days=1)
        st.metric("Daily Average (USD)", round(float(x),2))
        st.metric(f"EOM ({eom.isoformat()})", round(float(df_cost["cost_usd"].sum() + x*max(0,(eom-today).days)),2))
        st.metric(f"EOQ ({eoq.isoformat()})", round(float(df_cost["cost_usd"].sum() + x*max(0,(eoq-today).days)),2))
        st.area_chart(df_cost.set_index("date")["cost_usd"])