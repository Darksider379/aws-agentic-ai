#Agentic FinOps Streamlit UI
# -------------------------------
    FROM python:3.11-slim

    # Prevent Python cache, set no-buffering
    ENV PYTHONDONTWRITEBYTECODE=1 \
        PYTHONUNBUFFERED=1 \
            PIP_NO_CACHE_DIR=1
    
    # ---- System deps (for pandas, numpy, cryptography, etc.) ----
    RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl ca-certificates && \
            rm -rf /var/lib/apt/lists/*
    
    # ---- Workdir ----
    WORKDIR /app
    
    # ---- Copy and install dependencies ----
    COPY requirements.txt .
    RUN pip install --upgrade pip && pip install -r requirements.txt
    
    # ---- Copy source ----
    COPY . .
    # ---- Streamlit config ----
    #RUN mkdir -p .streamlit
    #COPY .streamlit/config.toml .streamlit/config.toml
    
    # ---- Environment variables (editable defaults) ----
    # AWS / Athena / FinOps Agent configuration
    ENV AWS_REGION="us-east-1" \
        ATHENA_DB="synthetic_cur" \
        ATHENA_TABLE="raw" \
        ATHENA_WORKGROUP="primary" \
        ATHENA_OUTPUT="s3://athena-query-results-agentic-ai/athena/" \
        RESULTS_BUCKET="athena-query-results-agentic-ai" \
        RESULTS_PREFIX="cost-agent-v2" \
        AGENT_FUNCTION_NAME="mainagent" \
        BEDROCK_AGENT_ID="IO47D3HMWR" \
        BEDROCK_ALIAS_ID="TSTALIAS123" \
        STREAMLIT_SERVER_HEADLESS=true \
        STREAMLIT_SERVER_PORT=8501 \
        STREAMLIT_SERVER_ENABLECORS=false \
        STREAMLIT_SERVER_ENABLEXsSRFProtection=false
    
    # ---- Expose Streamlit port ----
    EXPOSE 8501
    
    # ---- Health check ----
    HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
      CMD curl -f http://localhost:8501/_stcore/health || exit 1
    
    # ---- Run Streamlit ----
    #CMD ["streamlit", "run", "app_ui.py", "--server.address=0.0.0.0", "--server.port=8501"]
    CMD ["streamlit", "run", "app_ui.py","--server.address=0.0.0.0","--server.port=8501","--server.headless=true","--server.enableCORS=false","--server.enableXsrfProtection=false","--server.enableWebsocketCompression=false","--browser.gatherUsageStats=false"]