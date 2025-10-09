
from datetime import datetime, timedelta
import pandas as pd
import random

def cost(st,tab_cost):
    with tab_cost:
        st.subheader("Historical Cost")
        today = datetime.utcnow().date()
        rows = [{"date": (today - timedelta(days=90-i)).isoformat(), "cost_usd": 250 + random.uniform(-25,25) + (150 if random.random()<0.05 else 0)} for i in range(90)]
        df_cost = pd.DataFrame(rows)
        st.bar_chart(df_cost.set_index("date")["cost_usd"])
        with st.expander("Show raw data"):
            st.dataframe(df_cost, use_container_width=True, height=300)
        st.download_button("Download costs (CSV)", df_cost.to_csv(index=False), "cost_history.csv")

    st.divider()