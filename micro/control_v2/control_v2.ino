#include <Servo.h>
const unsigned long UPDATE_INTERVAL = 1000; // ms

const int POTENTIOMETER_PENDULUM    = A0;
const int POTENTIOMETER_LOWER_JOINT = A1;
const int POTENTIOMETER_UPPER_JOINT = A2;

const String READ_POTENTIOMETERS_TOKEN = "READ";
const String WRITE_MOTOR_COMMANDS_TOKEN = "WRITE";
const String FAILURE_TOKEN = "FAILURE";
const int NEWLINE_INT = 10;

String req;

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
    /*Serial.print(temp);
    Serial.print(" (");
    Serial.print(reading);
    Serial.print(")");
    Serial.print(" ");
    Serial.println(in_critical_zone);*/
    if (Serial.available() > 0) {
        String token = Serial.readStringUntil('\n');

        if (READ_POTENTIOMETERS_TOKEN.equals(token)) {
            Serial.print(adjust_for_deadzone(analogRead(POTENTIOMETER_PENDULUM)));
            //Serial.print(analogRead(POTENTIOMETER_PENDULUM));
            Serial.print(" ");
            Serial.print(analogRead(POTENTIOMETER_LOWER_JOINT));
            Serial.print(" ");
            Serial.println(analogRead(POTENTIOMETER_UPPER_JOINT));
        } else if(WRITE_MOTOR_COMMANDS_TOKEN.equals(token)) {
            int motor1 = Serial.parseInt();
            int motor2 = Serial.parseInt();

            if (motor1 >= 0 && motor1 <= 180 && motor2 >= 0 && motor2 <= 180) {
                s1.write(motor1);
                s2.write(motor2);
                //Serial.printf("%d %d\n", motor1, motor2);
            } else {
                Serial.println(FAILURE_TOKEN);
            }
        } else {
            //send failure token because we don't know what we just read :(
            Serial.print(FAILURE_TOKEN);
            Serial.print(" ");
            Serial.println("token received: ");
            Serial.print(" ");
            Serial.println(token);
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
