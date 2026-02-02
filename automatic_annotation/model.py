import ultralytics
import numpy as np
import cv2
import time
from ultralytics import YOLO
model = YOLO("best.pt")
# start= time.time()
results = model("sample.jpeg")
print(results[0].show())