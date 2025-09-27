void setup() {
  // Initialize serial communication at 115200 baud rate
  Serial.begin(115200);
  
  // Initialize built-in LED pin as output (usually GPIO 2)
  pinMode(2, OUTPUT);
  
  // Wait for serial port to connect
  delay(1000);
  Serial.println("ESP32 Started!");
  Serial.println("Built-in LED will blink every second");
}

void loop() {
  // Turn LED on
  digitalWrite(2, HIGH);
  Serial.println("LED ON");
  delay(1000);
  
  // Turn LED off
  digitalWrite(2, LOW);
  Serial.println("LED OFF");
  delay(1000);
}
