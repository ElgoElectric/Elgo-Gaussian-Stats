import pandas as pd
import os
from time import sleep, time
import boto3
from components import CycleDetection, GaussianCalculator
from random import randrange
from statistics import mean
from dotenv import load_dotenv

load_dotenv()

# Environment variables - for local development
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")

# General constants
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
FILE_NAME = "House_1.csv"
TARGET_TRAINING_SET = f"training/refrigerator/{FILE_NAME}"

class Orchestrator:
  '''
    Function to orchestrate the whole process from start to finish.

    This relies on the following environment variables:
    1. TARGET_TRAINING_SET - this is a file path along for a csv file. ENSURE THAT TARGET_TRAINING_SET only has normal operation only.
    2. DEVICE - specific device for which we are carrying out the training.
  '''

  def __init__(self, device, device_mapping):
    print(AWS_ACCESS_KEY_ID)
    print("Reading CSV file")
    start = time()
    self.s3 = boto3.client('s3', aws_access_key_id = AWS_ACCESS_KEY_ID, aws_secret_access_key = AWS_SECRET_ACCESS_KEY, aws_session_token=AWS_SESSION_TOKEN)
    response = self.s3.list_buckets()
    print("Bucket list:", response["Buckets"])

    response = self.s3.get_object(Bucket=AWS_S3_BUCKET, Key=TARGET_TRAINING_SET)

    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

    if status == 200:
        print(f"Successful S3 get_object response. Status - {status}")
        print("Reading CSV...")
        self.df = pd.read_csv(response.get("Body"))
        
    else:
        print(f"Unsuccessful S3 get_object response. Status - {status}")
        exit(-1)



    # self.df = pd.read_csv(TARGET_TRAINING_SET)
    print("Renaming...")
    self.df = self.df.rename(index=str, columns=device_mapping)
    print(f"Time taken: {time() - start}")

    # Cycle detection and count helpers
    print("Initializing Cycle Detector")
    start = time()
    self.cycle_detector = CycleDetection(df = self.df, device = device, model = "knn", mode = "train", s3_obj = self.s3)
    print(f"Time taken: {time() - start}")

    self.current_cycle = -1
    self.previous_cycle = -1

    self.current_power_list = []

    # Gaussian Detection
    self.normal_operation = self.df[device].tolist()
    self.gauss = GaussianCalculator(data = self.normal_operation)

  def run(self):
    # First train on the data made available for training.
    print(f"Training cycle detector")
    start = time()
    self.cycle_detector.KMeansTraining()
    print(f"Time taken: {time() - start}")

    # Continuously request for datapoint from an endpoint and check to see if it is anomalous or not
    while(True):
      sleep(3)
      datapoint = self.receive() # THIS METHOD IS YET TO BE IMPLEMENTED BASED ON AWS DATA STREAMING
      print(f"\n\nReceived Datapoint {datapoint}")
      self.current_cycle = self.cycle_detector.detect_on_off([[datapoint]]) # Outputs 0 or 1 for OFF or ON respectively
      print(f"Datapoint classified as {'ON' if self.current_cycle else 'OFF'}")

      if self.previous_cycle == -1 or self.previous_cycle == self.current_cycle:
        # If the previous cycle is uninitialized or if the current and previous datapoints belong to the same cycle
        print("Same as previous cycle")
        self.previous_cycle = self.current_cycle
        self.current_power_list.append(datapoint)

      if self.current_cycle != self.previous_cycle:
        if len(self.current_power_list) < 2:
          # The minimum number of datapoints to qualify as a cycle is 2. However, in the rare scenario that the power cycle keeps oscillating
          # between ON and OFF before accumulating atleast 2 datapoints in each cycle, we can continue the loop and set the current cycle to the
          # same as the previous cycle. This effectively accumulates atleast two datapoints in current power list, even if the power cycle is erratic
          # and jumps around.
          self.current_power_list.append(datapoint)
          self.previous_cycle = self.current_cycle
          continue

        print("Different from previous cycle")
        # The cycle has changed, and it's time to check if the cycle is anomalous or not.
        # Step 1: Calculate the average power in the current cycle (Done by the Gauss class, internally)
        # Step 2: Plug this into gaussian pdf formula (Need to trigger pdf function of Gauss)
        # Step 3: Evaluate to see if you need to raise alarm (Need to carry out check in this function)

        average_power = mean(self.current_power_list)
        self.gauss.calculate_pdf(average_power)
        alarm = self.gauss.sigma_rule()
        if alarm:
          print(f"ANOMALOUS CYCLE | Datapoint: {average_power}")
        else:
          print(f"NORMAL CYCLE | Datapoint: {average_power}")
          # Call update here using self.normal_operation
          self.update_normal_operation(self.current_power_list)


        self.current_power_list = []
        self.current_power_list.append(datapoint)
        self.previous_cycle = self.current_cycle
  def receive(self):
    '''
      Function to ping endpoint and see if there is any datapoint available.
      Endpoint to ping is AWS S3, and it contains buffered data.
    '''

    return randrange(0, 150, 5)
  def update_normal_operation(self, new_data):
    '''
    Function to:
      1. update self.normal_operation
      2. update mean and stdev by calling GaussianCalculator.update
      3. Periodically dump everything into csv file in S3 bucket
    In order to update the normal_operation, we have to dump everything from the previous cycle into
    the normal operation array.
    '''
    self.normal_operation += new_data
    self.gauss.update(data = self.normal_operation)
    # You have the option to prune or dump data here.
    pass