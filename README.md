# AWAREX – Driver Drowsiness & Attention Monitoring System

**A Multi-Sensor Embedded System for Driver Fatigue and Distraction Detection**

## Overview

AWAREX is a low-cost, externally installable driver monitoring system designed to detect **drowsiness and driver distraction** in buses, trucks, auto-rickshaws, and other small vehicles.

Unlike traditional systems that rely on a single sensor, AWAREX combines camera-based facial analysis with a neck-mounted motion sensor to monitor eye closure, head direction, and posture. An ESP32 processes sensor data and triggers distinct, staged buzzer alerts to warn drivers.

The system aims to make driver safety technology affordable and accessible without requiring vehicle integration or uncomfortable face-worn devices.

## Key Features

- **Drowsiness Detection:** Identifies prolonged eye closure, half-closed eyes, and forward head nodding.
- **Distraction Detection:** Detects sustained head movements away from the road.
- **Multi-Sensor Fusion:** Combines camera data and MPU6050 motion readings for more reliable detection.
- **Personalized Calibration:** Adapts eye and posture thresholds to individual drivers.
- **Staged Buzzer Alerts:** Different alert patterns for partial drowsiness, full drowsiness, and distraction.
- **Sensor Failure Handling:** Supports fallback modes when one of the sensors becomes unavailable.
- **False Alarm Reduction:** Uses time-based scoring, filtering, and hysteresis to distinguish normal movements from risky behavior.

## Hardware & Technologies

| Component | Purpose |
|---|---|
| ESP32 | Main decision-making and control unit |
| MPU6050 | Detects neck movement and posture |
| Webcam | Captures facial landmarks and head pose |
| Active Buzzer | Provides condition-specific alerts |
| Transistor Driver | Controls the buzzer |
| Calibrate & Reset Buttons | Driver calibration and system reset |

**Software:** C++, Arduino IDE, Python, OpenCV, MediaPipe

## How It Works

1. **Vision Processing:** A webcam captures facial landmarks to calculate Eye Aspect Ratio (EAR), head yaw, head pitch, and face visibility.
2. **Motion Sensing:** The MPU6050 measures forward and sideways neck movements.
3. **Data Transmission:** Camera measurements are transmitted to the ESP32 through USB serial communication.
4. **Sensor Fusion:** The ESP32 combines both sensor inputs using weighted scoring to identify driver behavior.
5. **Alert Generation:** Based on the detected condition, the ESP32 activates the corresponding buzzer pattern.

## Detection States

- **Attentive:** Normal eye state and head position.
- **Partially Drowsy:** Sustained half-closed eyes, providing an early warning.
- **Drowsy:** Prolonged eye closure or forward head nodding.
- **Distracted:** Sustained head movement away from the road while eyes remain open.

## Calibration

AWAREX includes a driver-specific calibration process:

- **1 Beep:** Maintain a normal upright posture for 3 seconds.
- **2 Beeps:** Nod forward and hold for 4 seconds.
- **3 Beeps:** Calibration successful.
- **5 Beeps:** Calibration failed; repeat the process.

This allows the system to adapt to individual eye openness and normal posture.

## Current Development Status

The initial working prototype includes:

- TinyML Integration: Develop an ESP32-based attention classifier using Edge Impulse.
- ESP32-based decision unit.
- Laptop webcam with OpenCV and MediaPipe.
- MPU6050 neck-mounted motion sensor.
- Sensor fusion and condition-specific buzzer alerts.
- Calibration and reset functionality.
- USB serial communication and sensor failure handling.

## Future Enhancements

- **Standalone Operation:** Replace laptop-based vision processing with an onboard camera module.
- **IR Illumination:** Improve detection in low-light conditions.
- **Phone Detection:** Add direct phone-use detection.
- **Haptic Feedback:** Introduce vibration-based alerts.
- **Wireless Connectivity:** Add Bluetooth Low Energy (BLE) or Wi-Fi monitoring.
- **Fatigue Analytics:** Record driving patterns, alert frequency, and fatigue trends.

## Limitations

- Sunglasses, glare, and poor lighting can affect eye detection.
- Road vibrations may introduce noise into motion sensor readings.
- Phone usage is currently inferred from head-down posture.
- Vision processing currently runs on a laptop rather than directly on the ESP32.
- Detection thresholds require further testing and refinement.

## Project Goal

To develop an affordable, compact, and reliable driver monitoring device that combines computer vision, embedded systems, and machine learning to improve road safety.

**Note:** AWAREX is currently a prototype under development and is not a certified vehicle safety device.
