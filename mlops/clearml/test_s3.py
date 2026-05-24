from clearml import PipelineController
import boto3
from botocore.client import Config

def test_s3_download():
    session = boto3.session.Session()
    s3_client = session.client(
        's3',
        endpoint_url='https://storage.yandexcloud.net',
        aws_access_key_id='YOUR_ACCESS_KEY',
        aws_secret_access_key='YOUR_SECRET_KEY'
    )
    
    obj = s3_client.get_object(Bucket='r-mlops-bucket-12-1-1-22209764', Key='nil_project/test/test.csv')
    content = obj['Body'].read().decode('utf-8')
    lines = content.split('\n')[:10]
    
    print("First 10 lines of test.csv:")
    for i, line in enumerate(lines):
        print(f"{i+1}: {line}")

pipe = PipelineController(
    name="Test S3 Download",
    project="mlops",
    version="1.0.0"
)

pipe.set_default_execution_queue("default")

pipe.add_function_step(
    name="test_s3",
    function=test_s3_download,
    function_return=[],
    packages=["boto3==1.43.0"]
)

if __name__ == "__main__":
    pipe.start(queue="default")
    print("Pipeline submitted")