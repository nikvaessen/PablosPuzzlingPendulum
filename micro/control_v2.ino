#include <Servo.h>

const unsigned long UPDATE_INTERVAL = 1000; // ms

const int POTENTIOMETER_PENDULUM    = A0;
const int POTENTIOMETER_LOWER_JOINT = A1;
const int POTENTIOMETER_UPPER_JOINT = A2;

const String READ_POTENTIOMETERS_TOKEN = "READ";
const String WRITE_MOTOR_COMMANDS_TOKEN = "WRITE";
const String FAILURE_TOKEN = "FAILURE";
const int NEWLINE_INT = 10;

String req

Servo s1;
Servo s2;

unsigned long prev_update_timestamp = 0;

void setup() {
  // put your setup code here, to run once:
  s1.attach(30);
  s2.attach(31);

  pinMode(POTENTIOMETER_PENDULUM, INPUT);
  pinMode(POTENTIOMETER_LOWER_JOINT, INPUT);
  pinMode(POTENTIOMETER_UPPER_JOINT, INPUT);

  Serial.begin(250000);
}

void loop() {
  // put your main code here, to run repeatedly:
  if(Serial.available() > 0)
  {
    String token = Serial.readStringUntil('\n');
    if(READ_POTENTIOMETERS_TOKEN.equals(token))
    {
        Serial.print(analogRead(POTENTIOMETER_PENDULUM));
        Serial.print(" ");
        Serial.print(analogRead(POTENTIOMETER_LOWER_JOINT));
        Serial.print(" ");
        Serial.println(analogRead(POTENTIOMETER_UPPER_JOINT));
      }
      else
      {
        // something went wrong
        Serial.println(FAILURE_TOKEN);
      }
    }
    else if(WRITE_MOTOR_COMMANDS_TOKEN.equals(token))
    {
       int motor1 = Serial.parseInt();
       int motor2 = Serial.parseInt();

       if(Serial.read() == NEWLINE_INT)
       {
          //s1.write(motor1);
          //s2.write(motor2);
          Serial.printf("%d %d\n", motor1, motor2);
       }
       else
       {
         Serial.println(FAILURE_TOKEN);
       }
    }
    else
    {
       //sent failure token because we don't know what we just read :(
       Serial.println(FAILURE_TOKEN);
    }
  }
}

