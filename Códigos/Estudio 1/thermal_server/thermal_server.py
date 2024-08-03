#!/usr/bin/env python3

from flask import Flask, render_template, Response, request
import cv2
import time
import yaml
from enum import Enum
from functools import wraps
import threading
import numpy as np
from datetime import datetime
import gzip
import sys
import os

with open('config.yaml') as f:
    config = yaml.safe_load(f)

class VideoFileFrameWriter:
    def __init__(self, filename, fourcc, framerate, size):
        self.videoWriter = cv2.VideoWriter(
            filename,
            cv2.VideoWriter_fourcc(*fourcc),
            framerate,
            size)
        print(f"Initialized video writer: {filename}")
    def write_frame(self, frame):
        self.videoWriter.write(frame)
    def close(self):
        self.videoWriter.release()

class ThermalFileFrameWriter:
    def __init__(self, filename, framewrite, size):
        self.gzip_file = gzip.open(filename, 'wb')
        self.size = np.array(size, dtype=np.uint16)
        self.gzip_file.write(self.size.tobytes())
    def write_frame(self, frame):
        self.gzip_file.write(frame.tobytes())
    def close(self):
        self.gzip_file.close()

class State(Enum):
    stopped = 0
    playing  = 1
    recording = 2

def fnsafe(s):
    return "".join([c for c in s if c.isalpha() or c.isdigit() or c==' ']).rstrip()

def timeString():
    return datetime.now().strftime("%Y-%m-%d %H.%M.%S")

class Recorder:
    instance = None
    @staticmethod
    def getInstance():
        if Recorder.instance is None:
            Recorder.instance = Recorder()
        return Recorder.instance
    def __init__(self):
        self.state = State.stopped
        self.keepGoing = False
        self.video = None
        self.thermal = None
        self.captureThread = None
        self.binImageData = None
        self.imageAccessLock = threading.Lock()
        self.recordingLock = threading.Lock()
        self.videoWriter = None
        self.thermalWriter = None
        self.imageCount = 0
        self.recordingLabel = ""        
    def play(self):
        if self.state != State.stopped:
            return
        self.video = cv2.VideoCapture(config['video_device'])
        self.thermal = cv2.VideoCapture(config['thermal_device'])
        self.captureThread = threading.Thread(target=self.captureLoop, args=())
        self.captureThread.start()
        self.state = State.playing
        print("play")
    def stop(self):
        if self.state == State.stopped:
            return
        if self.state == State.recording:
            self.stopRecording()
        self.keepGoing = False
        self.captureThread.join(3.0)
        self.captureThread = None
        self.video.release()
        self.thermal.release()
        self.video = None
        self.thermal = None
        self.state = State.stopped
        self.binImageData = None
        print("stop")
    def startRecording(self, condition):
        if self.state == State.recording:
            return
        if self.state == State.stopped:
            self.play()
        videoFramerate = self.video.get(cv2.CAP_PROP_FPS)
        videoSize = Recorder.getSize(self.video)
        thermalFramerate = self.thermal.get(cv2.CAP_PROP_FPS)
        thermalSize = Recorder.getSize(self.thermal)
        condition_str = fnsafe(condition)
        timestamp = timeString()
        self.recordingLabel = f"{timestamp} {condition}"
        videoWriter = VideoFileFrameWriter(
            os.path.join(
                config['data_folder'],
                f"{timestamp} {condition_str} video.mp4"),
            'mp4v',
            videoFramerate,
            videoSize)
        thermalWriter = ThermalFileFrameWriter(
            os.path.join(
                config['data_folder'],
                f"{timestamp} {condition_str} thermal.80x60.16bit.raw.gzip"),
            thermalFramerate,
            thermalSize)
        with self.recordingLock:
            self.videoWriter = videoWriter
            self.thermalWriter = thermalWriter
            self.state = State.recording
        print("start recording")
    def stopRecording(self):
        if self.state != State.recording:
            return
        with self.recordingLock:
            self.state = State.playing
            self.videoWriter.close()
            self.thermalWriter.close()
            self.vieoWriter = None
            self.thermalWriter = None
            self.recordingLabel = None
        print("stop recording")
    def captureLoop(self):
        print(f"Starting data capture {id(self)}", flush=True)
        self.keepGoing = True
        while self.keepGoing:
            # capture frames
            video_ok, video_frame = self.video.read()
            thermal_ok, thermal_frame = self.thermal.read()
            if not video_ok and thermal_ok:
                break
            if video_frame is None or thermal_frame is None:
                break
            # format for visualization
            video_img = self.resize_h(video_frame)
            thermal_img = self.resize_h(Recorder.raw2image(thermal_frame))
            T = self.raw2celsius(thermal_frame[25:35,35:45].mean())
            thermal_img = cv2.putText(
                thermal_img,
                f"Temperatura: {T:.2f} C",
                (10, 470),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                3)
            thermal_img = cv2.putText(
                thermal_img,
                f"Temperatura: {T:.2f} C",
                (10, 470),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,255,255),
                1)
            thermal_img = cv2.rectangle(
                thermal_img,
                (35*8, 25*8),
                (45*8, 35*8),
                (0, 0, 200),
                1)
            img = cv2.hconcat([video_img, thermal_img])
            ret, buffer = cv2.imencode('.jpg', img)
            data = buffer.tobytes()
            with self.imageAccessLock:
                self.binImageData = data
                self.imageCount += 1
            with self.recordingLock:
                if self.state == State.recording:
                    self.videoWriter.write_frame(video_frame)
                    self.thermalWriter.write_frame(thermal_frame)
    @staticmethod
    def resize_h(frame, H=480):
        h,w = frame.shape[:2]
        W = int(w * 480 / h)
        return cv2.resize(frame, (W,H))
    @staticmethod
    def raw2image(frame):
        image = np.array(
            np.clip(255*(((Recorder.raw2celsius(frame)-28.0)/8.0)**2),0,255),
            dtype=np.uint8)
        return cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    @staticmethod
    def raw2celsius(frame):
        return frame / 100.0 - 273.15
    @staticmethod
    def getSize(cap):
        w, h = [int(cap.get(x)) for x in [3,4]]
        return w, h


def gen_frames():
    recorder = Recorder.getInstance()
    print(f"Recorder: {id(recorder)}")
    imageCount = -1
    while recorder.state != State.stopped:
        with recorder.imageAccessLock:
            if imageCount != recorder.imageCount:
                imageCount = recorder.imageCount
                if recorder.binImageData is not None:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + recorder.binImageData + b'\r\n')
        time.sleep(0.01)


app = Flask(__name__)
recorder = Recorder.getInstance()
#recorder.play()

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        condition = request.form['condition']
    except:
        condition = ""
    recorder = Recorder.getInstance()
    if 'start' in request.form.keys():
        recorder.play()
    elif 'stop' in request.form.keys():
        recorder.stop()
    elif 'rec' in request.form.keys():
        recorder.startRecording(condition)
    elif 'stopRec' in request.form.keys():
        recorder.stopRecording()
    return render_template(
        'index.html',
        server_name = config['server_name'],
        state = recorder.state,
        condition = condition,
        recordingLabel = recorder.recordingLabel
    )

@app.route('/video_feed')
def video_feed():
    keep_going = False
    time.sleep(0.2)
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
