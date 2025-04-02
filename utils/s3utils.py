import boto3
import os
from botocore.exceptions import NoCredentialsError  

def download_file_from_s3(bucket_name, file_name, download_path):
    access_key = os.getenv('access_key')
    secret_key = os.getenv('secret_key')
    region = os.getenv('region')
    namespace = os.getenv('namespace')
    s3 = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key, region_name=region, endpoint_url=f'https://{namespace}.compat.objectstorage.{region}.oraclecloud.com')
    try:
        s3.download_file(bucket_name, file_name, download_path)
    except NoCredentialsError:
        print("Credentials not available")
        return False
    return True 