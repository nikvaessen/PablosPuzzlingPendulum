from communication.communication import Communicator

port = "/dev/cu.usbserial-A6003X31"
com = Communicator(usb_port=port)
while True:
    print(com.observe_state())