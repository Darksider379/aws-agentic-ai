from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
sp = ScriptProcessor(
    image_uri="763104351884.dkr.ecr.<region>.amazonaws.com/sagemaker-scikit-learn:1.4-1-cpu-py3",
    role="<SageMakerExecutionRole>", instance_count=1, instance_type="ml.m5.large",
    env={  # same envs
      "ATHENA_DB":"synthetic_cur",
      "ATHENA_TABLE":"raw_cast",
      "ATHENA_WORKGROUP":"primary",
      "ATHENA_OUTPUT":"s3://<athena-out>/",
      "RESULTS_BUCKET":"<your-results-bucket>",
      "RESULTS_PREFIX":"cost-agent",
      "ATHENA_RECS_TABLE":"recommendations"
    }
)
sp.run(code="main.py")
