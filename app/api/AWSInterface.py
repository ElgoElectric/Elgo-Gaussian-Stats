import boto3
from dotenv import load_dotenv
import os
from time import time
import pandas as pd

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

    def get_bucket_content(self, bucket_path):
        '''
        Function to read all the files in the bucket
        '''
        pass

    def get_latest_file(self, files):
        pass

    def write_to_bucket(self, bucket_name, file_name):
        pass

    def get_status(self):
        return self.status