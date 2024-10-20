#include "HX711.h"

const int irPin = 4;
const int DT = 2;
const int CLK = 3;
HX711 scale;
float calibration_factor = -21900;

void setup() {
  Serial.begin(9600);
  pinMode(irPin, INPUT);
  scale.begin(DT, CLK);
  scale.tare();
}

void loop() {
  int irState = digitalRead(irPin);
  if (irState == HIGH) {
    float weight = scale.get_units(10);
    Serial.print("Weight: ");
    Serial.println(weight * 1000); // Send weight in grams
  }
  delay(1000);
}
