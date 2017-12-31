from communication import Communicator

from cv2 import VideoCapture, imwrite
from time import sleep, time
from datetime import datetime
from threading import Thread

import os
import sys
import random

# import the Queue class from Python 3
if sys.version_info >= (3, 0):
    from queue import LifoQueue
else:
    print("run this code in python 3 please")
    exit()


class VideoCaptureWorker:

    def __init__(self, path, queueSize=5, sleep_for=0):
        # initialize the camera along with the boolean
        # used to indicate if the thread should be stopped or not
        self.cam = VideoCapture(path)
        self.stopped = False
        self.sleep_time = sleep_for

        # initialize the queue used to store frames read from
        # the video file
        self.Q = LifoQueue(maxsize=queueSize)

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                return

            # read the next frame from the file
            (grabbed, frame) = self.cam.read()

            # if the `grabbed` boolean is `False`, then we have
            # failed to read a frame and we should skip
            if not grabbed:
                print("skipped a frame due to read failure")
            else:
                # add the frame to the queue, and empty the queue if necessary
                if self.Q.full():
                    self.Q.empty()

                self.Q.put(frame)

            sleep(self.sleep_time)

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def more(self):
        # return True if there are still frames in the queue
        return self.Q.qsize() > 0

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


if __name__ == '__main__':
    desired_fps = 20
    iteration_length_in_s = 1/desired_fps
    take_images_for_s = 5
    total_count = desired_fps * take_images_for_s

    cam_worker = VideoCaptureWorker(0)
    cam_worker.cam.set(6, desired_fps) # https://stackoverflow.com/questions/11420748/setting-camera-parameters-in-opencv-python
    cam_worker.start()

    serial = Communicator("/dev/cu.usbserial-A6003X31") #on mac for jose, pablo and nik
    sleep(1)

    print("Starting")
    print("Initial position: {}".format(serial.observe_state()))

    count = 0
    buffer = [None for _ in range(0, total_count)]
    start_time = time()

    base = 90
    offset = 30

    while count < total_count:
        end_iteration_time = time() + iteration_length_in_s

        serial.send_command(base + random.randint(-offset, offset + 1), base + random.randint(-offset, offset + 1))
        sleep(0.005)

        img = cam_worker.read()
        buffer[count] = img
        count += 1
        print(count, datetime.now(), serial.observe_state())

        sleep_for = end_iteration_time - time()
        if sleep_for > 0:
            sleep(sleep_for)
        else:
            print("cannot keep up, had to sleep for {}".format(sleep_for))

    end_time = time()
    print(end_time - start_time)

    cam_worker.stop()

    ct = time()
    path = "video_backup/{}".format(ct)
    os.mkdir(path)
    for i, img in enumerate(buffer):
        imwrite(os.path.join(path, "file_{}.jpg".format(i)), img)

