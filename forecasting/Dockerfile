FROM public.ecr.aws/lambda/python:3.12
COPY requirements.txt .
RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

COPY forecasting.py ${LAMBDA_TASK_ROOT}/
COPY functions ${LAMBDA_TASK_ROOT}/functions

# Equivalent to config.ini
ENV AWS_REGION=us-east-1 \
    ATHENA_DB=synthetic_cur \
    ATHENA_TABLE=raw \
    ATHENA_WORKGROUP=primary \
    ATHENA_OUTPUT=s3://athena-query-results-agentic-ai/athena-output/ \
    RESULTS_BUCKET=athena-query-results-agentic-ai \
    RESULTS_PREFIX=cost-agent-v2 \
    ATHENA_FORECAST_TABLE=forecast_daily_v1 \
    FORECAST_TABLE_PREFIX=cost-agent-v2 \
    FORECAST_DEFAULT_DAYS=90 \
    COST_MODE=unblended \
    EXCLUDE_FEE=true \
    EXCLUDE_NON_USAGE_PRODUCTS=true \
    CLIP_OUTLIERS=true \
    OUTLIER_Q=0.98 \
    OUTLIER_MULT=1.20 \
    FORECAST_DEFAULT_DAYS=90 \
    CONFIG_INI_DISABLE=1


CMD ["forecasting.lambda_handler"]