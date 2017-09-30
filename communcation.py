import serial # python -m pip install pyserial OR apt-get install python3-serial


if __name__ == '__main__':
    # open serial connection
    ser = serial.Serial(
        port='COM5',
        baudrate=9600, # this needs to be set on micro-controller by doing Serial.begin(9600)
        parity=serial.PARITY_NONE,  # check parity of UC32, maybe it's even/odd
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=0) # set time-out higher if we want to wait for input

    print("connected to: " + ser.portstr)

    # Buffer for line
    line = []

    # infinite loop checking for input
    while True:
        for c in ser.read():
            line.append(c)
            if c == '\n':
                print("Line: " + str(line))
                line = []