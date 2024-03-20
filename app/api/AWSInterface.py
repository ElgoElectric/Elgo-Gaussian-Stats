import json
import boto3
from dotenv import load_dotenv
import os
from time import time
import pandas as pd
from datetime import datetime

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")

class AWSInterface:
    '''
    This class is meant to provide an abstracted way to interact with the AWS API, boto3. 
    This will also help modularize the security aspect of credentials management - all the environement variables should be pulled here. 

    This will feature the following functionalities:
    1. Read data from S3, given a bucket name and a file name
    2. Read list of files from S3, given a bucket name
    '''
    def __init__(self):
        '''
        1. Initialize an S3 object
        '''
        print("Connecting to AWS API...")
        start = time()
        self.s3 =boto3.client('s3', aws_access_key_id = AWS_ACCESS_KEY_ID, aws_secret_access_key = AWS_SECRET_ACCESS_KEY, aws_session_token=AWS_SESSION_TOKEN) 
        self.status = self.s3.list_buckets().get("ResponseMetadata", {}).get("HTTPStatusCode")
        self.last_read_stream = datetime.fromisoformat('2000-01-01 00:00:00.001+00:00')
        print(f"Connection status {self.status}. Finished in {time() - start}")
    

    def get_csv_file(self, bucket_path, file_name):
        '''
        Function to load the content of a csv file. 
        For collective anomalies, this is used to read the training dataset.
        '''
        print(f"Requesting from bucket {bucket_path}/{file_name}")
        response = self.s3.get_object(Bucket=bucket_path, Key=file_name)

        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        print("Got response from S3")
        
        if status == 200:
            print(f"Successful S3 get_object response. Status - {status}")
            print("Reading CSV from loaded body...")
            return pd.read_csv(response.get("Body"))
        else:
            print(f"Unsuccessful S3 get_object response. Status - {status}")
            exit(-1)

    def get_latest_in_bucket(self, bucket_path):
        '''
        Function to read all the files in the bucket, and get the content from the latest one. 
        Additionally, the function also detects if the bucket has been read before, and filters it if it has not been returned before. 
        '''
        contents = self.s3.list_objects_v2(Bucket = bucket_path).get("Contents")

        keys = {}

        for obj in contents:
            keys[obj.get("Key")] = obj.get("LastModified")

        latest = max(keys, key=keys.get)
        print(f"Last updated stream bucket at time {str(keys[latest])}")
        if keys[latest] <= self.last_read_stream:
            return []
        response = self.s3.get_object(Bucket=bucket_path, Key=latest)
        data = response.get("Body").readlines()
        power = []
        
        for line in data:
            buffer = json.loads(line)
            power.append(buffer.get("devicePower"))
        return power

    def write_to_bucket(self, bucket_name, target_directory, body):
        print("Writing data to S3 bucket", bucket_name)
        response = self.s3.put_object(Bucket=bucket_name, Key=target_directory, Body=body)

        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

        if status == 200:
            print(f"Successful S3 put_object response. Status - {status}")
        else:
            print(f"Unsuccessful S3 put_object response. Status - {status}")

    def get_status(self):
        return self.status