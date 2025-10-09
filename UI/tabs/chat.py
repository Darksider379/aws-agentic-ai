
import time
from functions.call_bedrock_agent import call_bedrock_agent

def chat(st,tab_chat):
    with tab_chat:
        h1, h2 = st.columns([0.8, 0.2])
        with h1:
            st.subheader("Chat with Agent")
        with h2:
            if st.button("🧹 Clear chat"):
                st.session_state.chat = []
                st.rerun()
        for m in st.session_state.chat:
            with st.chat_message(m["role"]):
                st.markdown(m["text"])
        q = st.chat_input("Ask about cross-cloud pricing, anomalies, forecasting, rightsizing…")
        if q:
            st.session_state.chat.append({"role":"user","text":q,"ts":time.time()})
            with st.chat_message("assistant"):
                ph = st.empty(); buf = ""; ans = call_bedrock_agent(q)
                for ch in ans:
                    buf += ch; time.sleep(0.002); ph.markdown(buf)
            st.session_state.chat.append({"role":"assistant","text":ans,"ts":time.time()})
            st.rerun()