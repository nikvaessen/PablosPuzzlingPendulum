from com import Communicator, Converter
import sys
import time

port = "/dev/cu.usbserial-A6003X31"
if sys.platform == 'linux' or sys.platform == 'linux2':
    port = '/dev/ttyUSB0'
elif sys.platform == 'win32':
    port = 'COM4'

com = Communicator(usb_port='/dev/ttyUSB0')
conv = Converter()
while True:

    commands = input('Command: ').split(' ')
    
    if commands[0] == 'read':
        print(conv.convert_vals(com.observe_state()))
    else:
        com.send_command(int(commands[0]), int(commands[1]))
    '''
    #com.ser.write('READ\n'.encode())
    print(com.observe_state())
    '''
    time.sleep(0.01)