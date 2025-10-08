lambda_functions=(
    "anomaly-http-proxy"
    "forecasting"
    "forecasting-proxy"
    "finops-agent-proxy"
    "finops-agent"
    "cross-cloud-pricing"
    "anomaly-handler"
)


for i in "${lambda_functions[@]}"; do
    aws lambda add-permission \
        --region us-east-1 \
        --function-name "$i" \
        --statement-id bedrock-CCVLQDODWY \
        --action lambda:InvokeFunction \
        --principal bedrock.amazonaws.com \
        --source-arn arn:aws:bedrock:us-east-1:784161806232:agent/IO47D3HMWR/alias/CCVLQDODWY
done