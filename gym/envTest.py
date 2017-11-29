from communication.communication import Communicator
from time import sleep

port = "/dev/cu.usbserial-A6003X31"
com = Communicator(usb_port=port, baudrate=9600)
change = 10

while True:
    change = -change
    sleep(0.3)
    com.send_command(90 + change, 90 + change)
