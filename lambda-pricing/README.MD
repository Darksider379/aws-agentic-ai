Command to fix the permission issue when bedrock tries to invoke Lambda function:

(.venv) chandramukherjee@FVFH83M2Q05N aws-agentic-ai %   aws lambda add-permission \
  --region us-east-1 \
  --function-name cross-cloud-pricing \
  --qualifier 2 \
  --statement-id bedrock-invocation \
  --action lambda:InvokeFunction \
  --principal bedrock.amazonaws.com \
  --source-arn arn:aws:bedrock:us-east-1:784161806232:agent/IO47D3HMWR

  Replace the function version in qualifier