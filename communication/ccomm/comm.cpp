#include <iostream>
#include <unistd.h>
#include "SerialStream.h"
#include "comm.h"

using namespace std;
using namespace LibSerial;

void Observe_state(int* states, int nr_states) {
	SerialStream serial_port(
		"/dev/ttyACM0", 
		SerialStreamBuf::BAUD_9600, 
		SerialStreamBuf::CHAR_SIZE_8, 
		SerialStreamBuf::PARITY_NONE, 
		1, SerialStreamBuf::FLOW_CONTROL_NONE);

	if (!serial_port.good()) {
		cerr << "Problem opening serial port." << endl;
		return;
	}
	usleep(1000);

	serial_port << REQUEST_DATA_TOKEN;
	while (serial_port.rdbuf()->in_avail() == 0) {
		usleep(100) ;
	}
	char mult, mod;
	for (int i = 0; i < nr_states; i++) {
		// getting the two values
		serial_port.get(mult);
		serial_port.get(mod);

		// converting to actual number and storing 
		*(states + i) = ((int) mult) * 256 + (int) mod;
	}

	serial_port.Close();
}

void Send_commands(int motor1, int motor2) {
	SerialStream serial_port(
		"/dev/ttyACM0", 
		SerialStreamBuf::BAUD_9600, 
		SerialStreamBuf::CHAR_SIZE_8, 
		SerialStreamBuf::PARITY_NONE, 
		1, SerialStreamBuf::FLOW_CONTROL_NONE);

	if (!serial_port.good()) {
		cerr << "Problem opening serial port." << endl;
		return;
	}
	usleep(1000);

	serial_port << WRITE_MOTOR_TOKEN << (Byte*) &motor1 << (Byte*) &motor2;
	char temp;
	serial_port.get(temp);
	cout << (int) temp << " ";
	serial_port.get(temp);
	cout << (int) temp << endl;

	serial_port.Close();
}