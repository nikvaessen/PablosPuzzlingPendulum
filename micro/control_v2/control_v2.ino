#include <Servo.h>
const unsigned long UPDATE_INTERVAL = 1000; // ms

const int POTENTIOMETER_PENDULUM    = A0;
const int POTENTIOMETER_LOWER_JOINT = A1;
const int POTENTIOMETER_UPPER_JOINT = A2;

const int READ_POTENTIOMETERS_TOKEN = 0;
const int WRITE_MOTOR_COMMANDS_TOKEN = 1;
const String FAILURE_TOKEN = "FAILURE";
const int NEWLINE_INT = 10;

byte readings[6];
int pendulum_reading, lower_joint_reading, upper_joint_reading;

Servo s1;
Servo s2;

unsigned long prev_update_timestamp = 0;

// for dead zone adjustment
int critical_value = 15;
int max_allowed_distance = 30;
int last_value = 0, llast_value = 0;
int pot_range = 1023;
bool in_critical_zone = false;

void setup() {
    s1.attach(30);
    s2.attach(31);

    s1.write(90 - 10);
    s2.write(90 - 15);

    pinMode(POTENTIOMETER_PENDULUM, INPUT);
    pinMode(POTENTIOMETER_LOWER_JOINT, INPUT);
    pinMode(POTENTIOMETER_UPPER_JOINT, INPUT);

    last_value = analogRead(POTENTIOMETER_PENDULUM);
    llast_value = last_value;

    Serial.begin(9600);
}

void loop() {
    if (Serial.available() > 0) {
        int token = Serial.read();

        if (READ_POTENTIOMETERS_TOKEN == token) {
            pendulum_reading = adjust_for_deadzone(analogRead(POTENTIOMETER_PENDULUM));
            lower_joint_reading = analogRead(POTENTIOMETER_LOWER_JOINT);
            upper_joint_reading = analogRead(POTENTIOMETER_UPPER_JOINT);
            readings[0] = pendulum_reading / 256;
            readings[1] = pendulum_reading % 256;
            readings[2] = lower_joint_reading / 256;
            readings[3] = lower_joint_reading % 256;
            readings[4] = upper_joint_reading / 256;
            readings[5] = upper_joint_reading % 256;
            Serial.write(readings, 6);    
        } else if(WRITE_MOTOR_COMMANDS_TOKEN == token) {
            Serial.readBytes(readings, 2);
            if (readings[0] >= 0 && readings[0] <= 180 && readings[1] >= 0 && readings[1] <= 180) {
                s1.write(readings[0]);
                s2.write(readings[1]);
            }
        }
    } else {
        int temp = adjust_for_deadzone(analogRead(POTENTIOMETER_PENDULUM));
    }
}

int adjust_for_deadzone(int pot) {
    if (!in_critical_zone && (!in_range(pot, critical_value, pot_range - critical_value) || abs(last_value - pot) + abs(llast_value - last_value) > max_allowed_distance)) {
        in_critical_zone = true;
    }
    if (in_critical_zone && abs(last_value - pot) + abs(llast_value - last_value) <= max_allowed_distance && (in_range(pot, critical_value, critical_value * 2) || in_range(pot, pot_range - critical_value * 2, pot_range - critical_value))) {
        in_critical_zone = false;
    }
    llast_value = last_value;
    last_value = pot;
    if (in_critical_zone && in_range(pot, critical_value, pot_range - critical_value)) {
        pot = 0;
    }
    return pot;
}

bool in_range(int val, int lower, int upper) {
    return val >= lower && val <= upper;
}
