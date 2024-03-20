import io
import os
from dotenv import load_dotenv
from api import AWSInterface

from sklearn.cluster import KMeans
import sklearn.metrics as metrics

load_dotenv()

# General constants
AWS_S3_BUCKET_TRAINING = os.getenv("AWS_S3_BUCKET_TRAINING")

class CycleDetection:

  def __init__(self, df, device, model, mode, n_clusters = 2):
    '''
    df - the data frame which we want to train on.
    n_clusters - defines the number of clusters we want.
    device - the name of the device, for which you want to detect cycles.
    model - either "knn" or "gmm". User needs to pass in this parameter
    mode - "test" or "train"
    '''
    self.n_clusters = n_clusters
    self.df = df
    self.device = device
    self.mode = mode
    self.aws_api = AWSInterface.AWSInterface()
    if model == "knn":
      self.model = KMeans(n_clusters= self.n_clusters, random_state=0) # Two states of the device: ON cycle and OFF cycle
    elif model!="knn" or model!="gmm":
      raise ValueError("The value passed in for the value paramter is incorrect")

  def KMeansTraining(self):
    '''
    Function will return a dataframe, and also dump it into a csv file.
    '''

    # Data frame creation
    self.df['Power Cycle'] = self.model.fit_predict(self.df[[self.device]]) # Add a new column called power cycle. This becomes z_k

    cluster_on = self.df.groupby('Power Cycle')[self.device].mean().idxmax()

    # Uncomment the line below, if ON or OFF output is preferred to 1s and 0s.
    # self.df['Power Cycle'] = self.df['Power Cycle'].map({cluster_on: 'ON', 1 - cluster_on: 'OFF'})

    # Data dump into csv file.
    target_directory = f"training/refrigerator/House_1_{self.device}_{self.mode}_labelled.csv"

    '''
    Data will be put into a file with the following format:
    "./drive/MyDrive/Datasets/House_1_{self.device}_labelled.csv"
    '''
    with io.StringIO() as csv_buffer:
      self.df.to_csv(csv_buffer, index=False)

      status = self.aws_api.write_to_bucket(bucket_name=AWS_S3_BUCKET_TRAINING, target_directory=target_directory, body=csv_buffer.getvalue())

      print(f"{'Successful' if status == 200 else 'Unsuccessful'}. Target Directory - {target_directory}")

    return self.df

  def detect_on_off(self, datapoint):
    '''
    Function takes in a single datapoint and tells you if it is part of
    an on cycle or an off cycle.
    '''
    # Use the KMeans model to predict the cluster for the single data point
    power_cycle = self.model.predict(datapoint)[0]

    # Determine the cluster representing ON and OFF states
    cluster_on = self.df.groupby('Power Cycle')[self.device].mean().idxmax()

    # Map the cluster label to 'ON' or 'OFF'
    # power_cycle_label = 'ON' if power_cycle == cluster_on else 'OFF'
    return power_cycle #, power_cycle_label

  def bulk_detect_on_off(self, data):
    '''
    Function to detect if a group of points "data" is on or off individually.
    '''
    return self.model.predict(data[[self.device]])

  def evaluation(self, ground_truth, test_data, metric = "acc"):
    '''
    Function to evaluate the trained dataset against the test data set, according the metric
    provided in the argument.

    ground_truth - dataframe object that represents the ground truth
    metric -  can be one of the following:
              1. "acc" - accuracy
              2. "prec" - precision
              3. "f1" - f1 score
    '''

    #Step 1: Calculate the predicted values first
    predicted_values = self.bulk_detect_on_off(test_data)
    print(type(predicted_values), type(ground_truth))

    print(f"Ground truth values: {ground_truth}\n")
    print(f"Obtained predicted values: {predicted_values}\nComputing {metric} metric now...")

    match metric:
      case "acc":
        return metrics.accuracy_score(y_true = list(ground_truth), y_pred = list(predicted_values))
      case "prec":
        return metrics.precision_score(y_true = list(ground_truth), y_pred = list(predicted_values))
      case "f1":
        return metrics.f1_score(y_true = list(ground_truth), y_pred = list(predicted_values))
      case _:
        raise ValueError('''
              Provide a valid metric:\n
              \t1. "acc" - accuracy\n
              \t2. "prec" - precision\n
              \t3. "f1" - f1 score"\n
            '''
            )