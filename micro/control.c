#include <Servo.h>

Servo s1;
Servo s2;

void setup() {
  // put your setup code here, to run once:
  s1.attach(30);
  s2.attach(31);

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
}