# Capstone
Draft of Capstone Repository for Downloadable Model

Imports/Dependencies Section:
---------------------------------------------------------------------------------------------------------------------------
from picamera2 import Picamera2
import cv2
import numpy as np
import time
import sys
import pytesseract
from tflite_runtime.interpreter import Interpreter

Prior to next steps ensure you have installed correct package and unzipped to downloads for car function

Set pathway for motor controls on vehicle:
sys.path.append("/home/"insert_your_raspberrypi_username"/Downloads/Yahboom_project/Raspbot/2.Hardware Control course/02.Drive motor")

Pull correct import from path above for motor controls:
import YB_Pcb_Car
-------------------------------------------------------------------------------------------------------------------------
