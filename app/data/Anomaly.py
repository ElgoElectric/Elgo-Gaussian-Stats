import json
from datetime import datetime

'''
Defines the packet that will be sent over HTTP to detect anomalies.
'''

class Anomaly:
    def __init__(self, device_label:str, timestamp_start:datetime, timestamp_end:datetime, valid_anomaly:bool, action_taken:bool) -> None:
        self.device_label = device_label
        self.timestamp_start = timestamp_start
        self.timestamp_end = timestamp_end
        self.valid_anomaly = valid_anomaly
        self.action_taken = action_taken
    
    def dict(self) -> dict:
        return {
            "device_label" : self.device_label,
            "timestamp_start" : self.timestamp_start.isoformat(),
            "timestamp_end" : self.timestamp_end.isoformat(),
            "valid_anomaly" : self.valid_anomaly,
            "action_taken" : self.action_taken
        }
    
    def json(self) -> str:
        return json.dumps(dict(
            device_label = self.device_label,
            timestamp_start = self.timestamp_start.isoformat(),
            timestamp_end = self.timestamp_end.isoformat(),
            valid_anomaly = self.valid_anomaly,
            action_taken = self.action_taken
        ))