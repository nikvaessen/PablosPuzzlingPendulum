#!/usr/bin/python3
from cppcomm import *
from time import sleep

while True:
	result = observe_state()
	print("Result:", result)
	send_commands(120, 45)
	sleep(0.01)

