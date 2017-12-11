#include <iostream>
#include <unistd.h>
#include "SerialStream.h"
#include "comm.h"

using namespace std;
using namespace LibSerial;

SerialStream serial_port;

void Init(const char* port) {
	if (!serial_port.IsOpen()) {
		cout << "Info: Trying to open serial port " << port << endl;

		serial_port.Open(port) ;
		serial_port.SetBaudRate(SerialStreamBuf::BAUD_9600);
		serial_port.SetCharSize(SerialStreamBuf::CHAR_SIZE_8);
		serial_port.SetParity(SerialStreamBuf::PARITY_NONE);
		serial_port.SetNumOfStopBits(1);
		serial_port.SetFlowControl(SerialStreamBuf::FLOW_CONTROL_NONE);

		usleep(100000);
		if (!serial_port.good()) {
			cerr << "[" << __FILE__ << ":" << __LINE__ << "] " << "Error: Could not open and initialise serial port " << port << "." << endl;
			exit(1);
		} else {
			cout << "Info: Successfully opened serial port " << port << endl;
		}
	} else {
		cout << "Warning: Serial port " << port << " is already open." << endl;
	}
}

void Close(void) {
	if (serial_port.IsOpen()) {
		serial_port.Close();
	} else {
		cout << "Warning: The serial port you are trying to close is not open." << endl;
	}
}

void Observe_state(int* states, int nr_states) {
	if (serial_port.IsOpen()) {
		if (!serial_port.good()) {
			cerr << "[" << __FILE__ << ":" << __LINE__ << "] " << "Error: Problem writing to serial port." << endl;
			return;
		}

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
	}
}

void Send_commands(int motor1, int motor2) {
	if (!serial_port.IsOpen()) {
		if (!serial_port.good()) {
			cerr << "[" << __FILE__ << ":" << __LINE__ << "] " << "Error: Problem writing to serial port." << endl;
			return;
		}

		serial_port << WRITE_MOTOR_TOKEN << (Byte*) &motor1 << (Byte*) &motor2;
		/*char temp;
		serial_port.get(temp);
		cout << (int) temp << " ";
		serial_port.get(temp);
		cout << (int) temp << endl;*/
	}
}