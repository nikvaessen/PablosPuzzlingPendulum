from com import Communicator, Converter
import sys
import time

port = "/dev/cu.usbserial-A6003X31"
if sys.platform == 'linux' or sys.platform == 'linux2':
    port = '/dev/ttyUSB1'
elif sys.platform == 'win32':
    port = 'COM4'

com = Communicator(usb_port='/dev/ttyUSB0')
conv = Converter()
change = 30
while True:
    '''
    commands = input('Command: ').split(' ')
    
    if commands[0] == 'read':
        print(com.observe_state())
    else:
        com.send_command(int(commands[0]), int(commands[1]))
    '''
    #com.send_command(90, 90 + change)
    #change = -change
    #com.ser.write('READ\n'.encode())
    print(com.observe_state())

    time.sleep(0.01)