String commandString, valueString, indexString;

void setup() {
    Serial.begin(9600);
    pinMode(A0, INPUT);
    pinMode(A1, INPUT);
    pinMode(A2, INPUT);
}

void loop() {
    readInput();
}

void readInput() {
    if (Serial.available() > 0) {
        commandString = Serial.readStringUntil('\n');
        if (commandString.startsWith("req")) {
            Serial.print(analogRead(A0));
            Serial.print(" ");
            Serial.print(analogRead(A1));
            Serial.print(" ");
            Serial.println(analogRead(A2)); 
        }
    }
}
