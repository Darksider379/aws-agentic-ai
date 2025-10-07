FROM public.ecr.aws/lambda/python:3.12
COPY requirements.txt .
RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

COPY mainagent.py ${LAMBDA_TASK_ROOT}/
COPY recommendations ${LAMBDA_TASK_ROOT}/recommendations
COPY functions ${LAMBDA_TASK_ROOT}/functions

# Equivalent to config.ini
ENV ATHENA_DB=synthetic_cur \
    ATHENA_TABLE=raw \
    ATHENA_OUTPUT=s3://athena-query-results-agentic-ai/athena/ \
    RESULTS_BUCKET=athena-query-results-agentic-ai \
    RESULTS_PREFIX=cost-agent-v2 \
    USE_LLM=true \
    BEDROCK_REGION=us-east-1 \
    BEDROCK_MODEL_ID=amazon.nova-pro-v1:0 \
    S3_STANDARD_PER_GB_MONTH=0.023 \
    S3_IA_PER_GB_MONTH=0.0125 \
    EC2_M5_LARGE=0.096 \
    EC2_M7G_LARGE=0.084 \
    S3_DEFAULT_RETRIEVAL_RATE=0.10 \
    LLM_ABS_CAP=100000 \
    LLM_REL_CAP_PCT=30

CMD ["mainagent.handler"]
