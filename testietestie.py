import sys

from time import sleep, time
from ourgym import RobotArm

if sys.platform == 'linux' or sys.platform == 'linux2':
    port = '/dev/ttyUSB0'
elif sys.platform == 'win32':
    port = 'COM4'
else:
    port = "/dev/cu.usbserial-A6003X31"

if __name__ == '__main__':

    # Make the object which which we can manipulate the environment
    arm = RobotArm(usb_port="/dev/cu.usbserial-A6003X31")

    # gym API
    # arm.reset() # sets it back to 90, 90
    # arm.step(5) # takes an argument (integer between 0, 81) and it will perform that action
    #
    # state, reward, done, info = arm.step(10) # same, but returns the new state and the reward of that state

    # manually sending commands


    # COLLECT THE DATA
    all_states = []
    for i in range(5):
        arm.reset()
        arm.com.send_command(40, 40) # always sleep a bit after sending a command
        # print('first command')
        # sleep(0.01)
        # arm.com.send_command(140, 140) # always sleep a bit after sending a command
        # print('second command')
        # sleep(0.00001)
        # arm.com.send_command(60, 60) # always sleep a bit after sending a command
        # print('third command')
        # sleep(0.00001)

        #arm.step(31)
        print(1)
        #arm.step(41)
        print(2)
        #arm.step(51)
        print(3)
        #arm.step(61)
        print(4)

        current_time = time()
        states = []
        while time() - current_time < 0.1:
            state = arm._get_current_state()
            states.append(state)
            sleep(0.00000001)

        #print(states)
        all_states.append(states)

        sleep(3)


    # LOOP over the collected data
    for obs in all_states:
        print(all_states)


    # Store the data in a file by looping over it
    # open(f, 'w') means 'w' = writing, 'r' = reading, 'a' = appending
    with open("fancy_file_name.txt", "w") as file:
        for obs in all_states:
            file.write(str(obs))
            file.write("\n")

        file.flush()

