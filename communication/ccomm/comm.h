typedef unsigned char Byte;

const static char REQUEST_DATA_TOKEN = '0';
const static char WRITE_MOTOR_TOKEN = '1';

void Observe_state(int* states, int nr_states);
void Send_commands(int motor1, int motor2);