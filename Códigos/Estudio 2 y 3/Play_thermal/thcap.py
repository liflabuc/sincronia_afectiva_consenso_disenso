import cv2 as cv
import sys
import numpy as np
import threading
import time
import gzip
from datetime import datetime
from time import time

def getSize(cap):
    w, h = [int(cap.get(x)) for x in [3,4]]
    return w, h

def enumerate():
    thcams = []
    vcams = []
    index = 0
    while True:
        try:
            print(f"Querying device {index}")
            cap = cv.VideoCapture(index)
            backend = cap.getBackendName()
            w, h = getSize(cap)
            print(backend, w, h)
            if w == 80 and h == 60:
                thcams.append(index)
            else:
                vcams.append(index)
        except Exception as e:
            print(e)
            if index > 4:
                break
        index += 1
    return thcams, vcams

class VideoFileFrameWriter:
    def __init__(self, filename, fourcc, framerate, size):
        self.videoWriter = cv.VideoWriter(
            filename,
            cv.VideoWriter_fourcc(*fourcc),
            framerate,
            size)
        print(f"Initialized video writer: {filename}")
    def write_frame(self, frame):
        self.videoWriter.write(frame)
    def close(self):
        self.videoWriter.release()

class ThermalFileFrameWriter:
    def __init__(self, filename, framerate, size):
        self.gzip_file = gzip.open(filename, 'wb')
        self.size = np.array(size, dtype=np.uint16)
        self.framerate = np.array(framerate, dtype=float)
        self.gzip_file.write(self.framerate.tobytes())
        self.gzip_file.write(self.size.tobytes())
    def write_frame(self, frame, t = None):
        if t is None:
            t = time()
        t=np.array(t, dtype=float)
        self.gzip_file.write(t.tobytes())
        self.gzip_file.write(frame.tobytes())
    def close(self):
        self.gzip_file.close()

        
class FrameRecorder:
    def __init__(self, capture_string, capture_type, imgcallback = None):
        self.capture_string = capture_string
        self.frame_writer = None
        self.img_cb = imgcallback
        self.image_lock = threading.Lock()
        self.keep_going = False
        print(f"Video capture: {capture_string}")
        self.cap = cv.VideoCapture(capture_string, capture_type)
        self.framerate = self.cap.get(cv.CAP_PROP_FPS)
        self.size = getSize(self.cap)
        self.frame = None

    def setFrameWriter(self, frame_writer):
        self.frame_writer = frame_writer
        
    def recLoop(self):
        while self.keep_going:
            self.image_lock.acquire()
            self.cap.grab()
            t = time()
            ret, self.frame = self.cap.retrieve()
            if self.img_cb is not None:
                self.img_cb(self.frame)
            self.image_lock.release()
            if not ret:
                print("Can't read image frame")
                break
            if self.frame_writer is not None:
                self.frame_writer.write_frame(self.frame, t)



    def start(self):
        self.keep_going = True
        self.recThread = threading.Thread(target=self.recLoop, args=())
        self.recThread.start()
    def stop(self):
        self.keep_going = False
        self.recThread.join(3.0)

def timeString():
    return datetime.now().strftime("%Y-%m-%d %H.%M.%S")

def gststr(dev_path):
    return \
        f"gst-launch-1.0 v4l2src device={dev_path}" +\
        " ! video/x-raw,format=GRAY16_LE" +\
        " ! appsink"

def raw2celsius(frame):
    return frame / 100.0 - 273.15

def raw2image(frame):
    image = np.array(
        np.clip(255*(((raw2celsius(frame)-28.0)/8.0)**2),0,255),
        dtype=np.uint8)
    return cv.resize(image, (640, 480))

if __name__ == "__main__":
    import yaml, os
    enumerate()
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    timestamp = timeString()
    # Set up thermal recording
    threc = FrameRecorder(
        config['thermal_device'],
        raw2image)
    thw = ThermalFileFrameWriter(
        os.path.join(
            config['data_folder'],
            f"{timestamp} thermal.80x60.16bit.raw.gzip"),
        threc.framerate,
        threc.size)
    threc.setFrameWriter(thw)
    # Set up webcam recording
    vrec = FrameRecorder(3)
    vw = VideoFileFrameWriter(
        os.path.join(
            config['data_folder'],
            f"{timestamp} video.mp4"),
        'mp4v',
        vrec.framerate,
        vrec.size)
    vrec.setFrameWriter(vw)
    # start
    print("starting")
    threc.start()
    vrec.start()
    # wait
    time.sleep(10)
    # stop
    threc.stop()
    vrec.stop()
    thw.close()
    vw.close()
