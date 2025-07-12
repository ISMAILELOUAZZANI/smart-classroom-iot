# Smart Classroom IoT Monitoring System

## 🧠 Description
An ESP32-based IoT system that collects classroom environment data and sends it to a secure MQTT broker. A Python-based dashboard displays temperature, gas, sound, and motion detection in real time.

## ⚙️ Tech Stack
- ESP32 + DHT11 + MQ2 + KY-038 + PIR
- HiveMQ Cloud (MQTT Broker)
- Python + paho-mqtt + tkinter
- LCD 16x2 with I2C for display on ESP32

## 🛠 Features
- Real-time sensor data collection
- Secure communication with HiveMQ
- GUI dashboard to monitor classroom status

## ▶️ How to Run (Dashboard)
```bash
pip install paho-mqtt
python main.py
