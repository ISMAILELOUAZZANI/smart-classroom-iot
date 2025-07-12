#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <PubSubClient.h>
#include <Wire.h>
#include <DHT.h>

// === DHT11 ===
#define DHTPIN 26
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);

// === Capteurs ===
#define KY038_PIN 35
#define MQ2_PIN   34
#define PIR_PIN   14

// === LEDs ===
#define LED_ROUGE  17
#define LED_ORANGE 2
#define LED_VERTE  16

// === Wi-Fi ===
const char* ssid = "realme 11";
const char* password = "12345678";

// === HiveMQ Cloud ===
const char* mqtt_server = "383d295644b342beb205f38f7c4125fa.s1.eu.hivemq.cloud";
const int mqtt_port = 8883;
const char* mqtt_user = "Houssine";
const char* mqtt_password = "Houssine2001";

WiFiClientSecure wifiClient;
PubSubClient client(wifiClient);

void setup_wifi() {
  Serial.print("Connexion WiFi à ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nConnecté au WiFi !");
}

void reconnect() {
  while (!client.connected()) {
    Serial.print("Connexion MQTT...");
    if (client.connect("ESP32Client", mqtt_user, mqtt_password)) {
      Serial.println("Connecté à HiveMQ !");
    } else {
      Serial.print("Échec, rc=");
      Serial.print(client.state());
      delay(5000);
    }
  }
}

void setup() {
  Serial.begin(115200);
  Wire.begin();
  dht.begin();

  pinMode(PIR_PIN, INPUT);
  pinMode(LED_ROUGE, OUTPUT);
  pinMode(LED_ORANGE, OUTPUT);
  pinMode(LED_VERTE, OUTPUT);

  digitalWrite(LED_ROUGE, LOW);
  digitalWrite(LED_ORANGE, LOW);
  digitalWrite(LED_VERTE, LOW);

  setup_wifi();
  wifiClient.setInsecure();  // ⚠ Insecure TLS pour tests
  client.setServer(mqtt_server, mqtt_port);
}

void loop() {
  if (!client.connected()) reconnect();
  client.loop();

  float temp = dht.readTemperature();
  float hum = dht.readHumidity();
  int sound = analogRead(KY038_PIN);
  int gas = analogRead(MQ2_PIN);
  bool motion = digitalRead(PIR_PIN);

  String qualite = classerQualite(temp, hum, gas, sound, motion);

  Serial.println("=====");
  Serial.print("Temp: "); Serial.println(temp);
  Serial.print("Hum: "); Serial.println(hum);
  Serial.print("Gaz: "); Serial.println(gas);
  Serial.print("Son: "); Serial.println(sound);
  Serial.print("Mouvement: "); Serial.println(motion ? "Oui" : "Non");
  Serial.print("Classe: "); Serial.println(qualite);
  Serial.println("=====");

  String json = "{";
  json += "\"temperature\":" + String(temp, 1) + ",";
  json += "\"humidite\":" + String(hum, 0) + ",";
  json += "\"gaz\":" + String(gas) + ",";
  json += "\"son\":" + String(sound) + ",";
  json += "\"mouvement\":" + String(motion ? "true" : "false") + ",";
  json += "\"qualite\":\"" + qualite + "\"";
  json += "}";

  client.publish("esp32/air", json.c_str());

  delay(5000);
}

String classerQualite(float temp, float hum, int gaz, int son, bool mouvement) {
  // Éteindre toutes les LEDs
  digitalWrite(LED_ROUGE, LOW);
  digitalWrite(LED_ORANGE, LOW);
  digitalWrite(LED_VERTE, LOW);

  float score = 0.0;

  // Température (idéal : 20-28°C)
  if (temp < 18 || temp > 35) score += 2.5;
  else if (temp < 20 || temp > 30) score += 1.5;
  else if (temp < 22 || temp > 28) score += 0.5;

  // Humidité (idéal : 40-60%)
  if (hum < 20 || hum > 80) score += 2.5;
  else if (hum < 30 || hum > 70) score += 1.5;
  else if (hum < 40 || hum > 60) score += 0.5;

  // Gaz (MQ2) – seuil brut
  if (gaz > 1000) score += 3;
  else if (gaz > 700) score += 2;
  else if (gaz > 400) score += 1;

  // Son (KY-038)
  if (son > 3000) score += 2.5;
  else if (son > 2000) score += 1.5;
  else if (son > 1000) score += 0.5;

  // Mouvement
  if (mouvement) score += 1;

  // === Interprétation du score total ===
  String qualite;
  if (score >= 6.0) {
    digitalWrite(LED_ROUGE, HIGH);
    qualite = "Mauvaise 🔴";
  } else if (score >= 3.0) {
    digitalWrite(LED_ORANGE, HIGH);
    qualite = "Moyenne 🟡";
  } else {
    digitalWrite(LED_VERTE, HIGH);
    qualite = "Bonne 🟢";
  }

  return qualite;
}