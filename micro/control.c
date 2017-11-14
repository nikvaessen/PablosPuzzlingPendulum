#include <Servo.h>

const unsigned long UPDATE_INTERVAL = 50; // ms

const int POTENTIOMETER_PENDULUM    = A0;
const int POTENTIOMETER_LOWER_JOINT = A1;
const int POTENTIOMETER_UPPER_JOINT = A2;

Servo s1;
Servo s2;

unsigned long PREV_UPDATE_TIME = 0;

void setup() {
  // put your setup code here, to run once:
  s1.attach(30);
  s2.attach(31);

  pinMode(POTENTIOMETER_PENDULUM, INPUT)
  pinMode(POTENTIOMETER_LOWER_JOINT, INPUT)
  pinMode(POTENTIOMETER_UPPER_JOINT, INPUT)

  Serial.begin(9600);
}

void loop() {
  // put your main code here, to run repeatedly:
  if(Serial.available() > 0){
    int x = Serial.parseInt();
    int y = Serial.parseInt();

    // we send something like "10, 10\n)
    if(Serial.read() == 10){
      Serial.printf("Received %d, %d\n", x, y);
      s1.write(x);
      s2.write(y);
    }
    else {
      // something went wrong
    }
  }

  unsigned long current = millis()
  if(current - PREV_UPDATE_TIME > UPDATE_INTERVAL){
    int p = analogRead(POTENTIOMETER_PENDULUM)
    int jl = analogRead(POTENTIOMETER_LOWER_JOINT);
    int ju = analogRead(POTENTIOMETER_UPPER_JOINT);

    Serial.printf("%d, %d, %d\n", p, jl, ju);
  }
}