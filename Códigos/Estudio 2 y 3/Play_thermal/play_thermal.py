import gzip
import numpy as np
import cv2 as cv
import time
from thcap import raw2image, raw2celsius


def play_file(filename):
    with gzip.open(filename, 'rb') as f:
        buffer = f.read(8)
        framerate = np.frombuffer(buffer, dtype=float)
        buffer = f.read(4)
        size = np.frombuffer(buffer, dtype=np.uint16)
        print(size)
        w, h = size
        framelen = w*h*2

        while True:
            buffer = f.read(8)
            if len(buffer) == 0:
                break
            time = np.frombuffer(buffer, dtype=float)
            print(float(time[0]))
            buffer = f.read(framelen)
            if len(buffer) != framelen:
                break
            frame = np.frombuffer(buffer, dtype=np.uint16)
            frame.shape = (h, w)
            frame = ((frame / 100. - 273.15) - 20.0) / 40.
            cv.imshow('Thermal image', cv.resize(frame, (400, 300)))
            if cv.waitKey(20) & 0xFF == ord('q'):
                break


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        prog='Play thermal capture file',
        description='Displays a thermal capture as video')
    parser.add_argument('filename', help="input filename")
    args = parser.parse_args()

    path_file = args.filename

    play_file(path_file)
