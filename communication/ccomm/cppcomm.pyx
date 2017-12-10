cdef extern from "comm.h":
	void Observe_state(int* states, int nr_states)
	void Send_commands(int motor1, int motor2)

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
	cdef int motor1 = m1, motor2 = m2
	Send_commands(motor1, motor2)
