cdef extern from "comm.h":
	void Init(const char* port)
	void Observe_state(int* states, int nr_states)
	void Send_commands(int motor1, int motor2)

def init(port):
	port_encoded = port.encode('UTF-8')
	cdef char* c_port = port_encoded
	Init(c_port)

def observe_state():
	cdef int nr_states = 3
	cdef int states[3] 
	cdef int i
	Observe_state(&states[0], nr_states)
	result = []
	i = 0
	while i < nr_states:
		result.append(states[i])
		i = i + 1
	return result

def send_commands(m1, m2):
	cdef int c_m1 = m1, c_m2 = m2
	Send_commands(c_m1, c_m2)
