import socket
import numpy as np
import requests
import threading 
import time
import base64
import matplotlib
from matplotlib import pyplot as plt

def threaded(func):
    def wrapper(*k, **kw):
        thread = threading.Thread(target=func,args=k,kwargs=kw, daemon=True)
        thread.start()
        return thread
    return wrapper

# import logging

# def configure_logger(logger_name='my_logger', level=logging.INFO):
#     """
#     Configures the logger with colors for different log levels.
    
#     :param logger_name: Name of the logger.
#     :param level: Logging level.
#     :return: A configured logger.
#     """
#     # Create the logger
#     logger = logging.getLogger(logger_name)
#     logger.setLevel(level)

#     # Create console handler
#     console_handler = logging.StreamHandler()
#     console_handler.setLevel(level)

#     # Add the handler to the logger
#     if not logger.hasHandlers():
#         logger.addHandler(console_handler)

#     return logger

class Plot():
    def __init__(self, filename, plotname) -> None:
        self.filename = filename
        self.plotname = plotname
        
    def plot(self):
        x_data = []
        
        cur_data = []
        target_data = []
        
        with open(self.filename, "r") as fp:
            for line in fp:
                timestamp, speed, target = line.split(";")
                x_data.append(float(timestamp))
                cur_data.append(float(speed))
                target_data.append(float(target))
            fp.close()
        plt.plot(x_data, cur_data)
        plt.plot(x_data, target_data)
        plt.savefig(self.plotname)
        
        
        
class Streamer():

    def __init__(self, IP):
        self.run = True
        self.verbose = False

        self.data ={
                "url" : "http://"+IP+":9824/new_frame",
                "RGB" : {
                    "frame_lock" : threading.Lock(),
                    "frame" : None,
                    "update" : False
                },
                "Depth" : {
                    "frame_lock" : threading.Lock(),
                    "frame" : None,
                    "update" : False
                },
                "BEV" : {
                    "frame_lock" : threading.Lock(),
                    "frame" : None,
                    "update" : False
                },
                "Controls" : {
                    "data_lock" : threading.Lock(),
                    "data" : None,
                    "update" : False
                }
        }

        # launch the streamer thread
        threading.Thread(target=self.sendRGBImage, name = "RGBStreamer").start()
        threading.Thread(target=self.sendDepthImage, name = "DepthStreamer").start()
        threading.Thread(target=self.sendBEVImage, name = "BEVStreamer").start()
        threading.Thread(target=self.sendControlsData, name = "ControlsDataStreamer").start()
        

    def sendRGBImage(self):
        self.__sendImage("RGB")

    def sendDepthImage(self):   
        self.__sendImage("Depth")
    
    def sendBEVImage(self):   
        self.__sendImage("BEV")
    
    def sendControlsData(self):   
        self.__sendData("Controls")
    
    def __sendData(self, datatype):
        while self.run:
            try:
                if self.data[datatype]["data"] is not None and self.data[datatype]["update"]:
                    if self.verbose:
                        print("acquire")
                    self.data[datatype]["data_lock"].acquire()
                    if self.verbose:
                        print("send_image")

                    data = {
                        "type" : datatype, 
                        "data" : self.data[datatype]["data"]
                    }
                    requests.post(self.data["url"], json=data, timeout=10)

                    if self.verbose:
                        print("post_ok")
                    self.data[datatype]["update"] = False
                    self.data[datatype]["data_lock"].release()
                    if self.verbose:
                        print("sent")
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(e)
                if self.data[datatype]["data_lock"].locked():
                    self.data[datatype]["data_lock"].release() 

    def __sendImage(self, datatype):
        ''''
        Take the objects image and send it
        '''
        while self.run:
            try:
                if self.data[datatype]["frame"] is not None and self.data[datatype]["update"]:
                    if self.verbose:
                        print("acquire")
                    self.data[datatype]["frame_lock"].acquire()
                    if self.verbose:
                        print("send_image")

                    data = {
                        "type" : datatype, 
                        "data" : {
                            "encode" : base64.b64encode(self.data[datatype]["frame"].tobytes()).decode('utf-8'),
                            "dtype" : str(self.data[datatype]["frame"].dtype),
                            "shape" : self.data[datatype]["frame"].shape
                        }
                    }
                    requests.post(self.data["url"], json=data, timeout=10)

                    if self.verbose:
                        print("post_ok")
                    self.data[datatype]["update"] = False
                    self.data[datatype]["frame_lock"].release()
                    if self.verbose:
                        print("sent")
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(e)
                if self.data[datatype]["frame_lock"].locked():
                    self.data[datatype]["frame_lock"].release()
        print("---- Stream finished ----")
    
    def send_data(self, datatype, data):
        self.data[datatype]["data_lock"].acquire()
        self.data[datatype]["data"] = data
        self.data[datatype]["update"] = True
        self.data[datatype]["data_lock"].release()

    def send_frame(self, datatype, frame):
        self.data[datatype]["frame_lock"].acquire()
        self.data[datatype]["frame"] = frame
        self.data[datatype]["update"] = True
        self.data[datatype]["frame_lock"].release()

if __name__ == "__main__":
    speed_plot = Plot("./userCode/speed.txt", "./userCode/speedplot.png")
    speed_plot.plot()
