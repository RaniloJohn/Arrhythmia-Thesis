/**
 * @file main.cpp
 * @brief ESP32-C3 + MAX30102 Firmware for Wearable Arrhythmia Acquisition Node
 * 
 * Thesis: An Internet of Things-Based Framework for Cardiac Arrhythmia Detection
 *         via Photoplethysmography and Machine Learning
 * Authors: Bauzon, Condino, Delos Angeles, Rana, Villaflor (3CPE-2A, Univ. of the East)
 * Adviser: Dr. Nelson Rodelas
 * 
 * Specifications:
 *  - Microcontroller: ESP32-C3 RISC-V (160 MHz)
 *  - Sensor: MAX30102 Optical Biosensor (Red 660nm, IR 940nm) via I2C (SDA=GPIO 8, SCL=GPIO 9)
 *  - Display: SSD1306 0.96" OLED (128x64, I2C 0x3C)
 *  - Sampling Rate: Deterministic 100 Hz (10 ms period)
 *  - Protocol: Framed binary UART with CRC16-CCITT and 0xAA / 0x55 delimiters (ADR-001)
 *  - On-board Heuristic: Local peak detection for OLED fallback alert
 */

#include <Arduino.h>
#include <Wire.h>
#include "MAX30105.h"
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// Pin Definitions for ESP32-C3
#ifndef I2C_SDA_PIN
#define I2C_SDA_PIN 8
#endif
#ifndef I2C_SCL_PIN
#define I2C_SCL_PIN 9
#endif

// Display Dimensions
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
#define SCREEN_ADDRESS 0x3C

// Sampling & Framing Definitions
#define SAMPLING_RATE_HZ 100
#define SAMPLE_INTERVAL_US (1000000 / SAMPLING_RATE_HZ) // 10,000 us = 10 ms
#define FRAME_START_BYTE 0xAA
#define FRAME_END_BYTE   0x55
#define FRAME_TYPE_RAW   0x01

#pragma pack(push, 1)
struct PpgPacket {
    uint8_t  start_byte;        // 0xAA
    uint8_t  frame_type;        // 0x01
    uint32_t timestamp_ms;      // Monotonic time on MCU
    uint32_t ir_raw;            // 18-bit raw IR channel
    uint32_t red_raw;           // 18-bit raw Red channel
    uint16_t heuristic_bpm_x10; // e.g. 725 = 72.5 BPM (on-board fallback)
    uint16_t crc16;             // CRC-16-CCITT (polynomial 0x1021)
    uint8_t  end_byte;          // 0x55
};
#pragma pack(pop)

// Peripheral Instances
MAX30105 particleSensor;
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// Ring Buffer for Deterministic 100 Hz Sampling
#define RING_BUFFER_SIZE 256
struct SampleData {
    uint32_t ts;
    uint32_t ir;
    uint32_t red;
};
volatile SampleData ringBuffer[RING_BUFFER_SIZE];
volatile uint16_t rbHead = 0;
volatile uint16_t rbTail = 0;

// Local Heuristic Peak Detection & Fallback State
float dcFilterIR = 0.0f;
float acFilteredIR = 0.0f;
uint32_t lastPeakTime = 0;
float currentBpm = 0.0f;
float avgBpm = 0.0f;
#define BPM_HISTORY_SIZE 4
float bpmHistory[BPM_HISTORY_SIZE] = {0};
uint8_t bpmIndex = 0;
bool irregularRhythmAlert = false;
uint8_t consecutiveIrregularCount = 0;

// Display Throttling (5 Hz)
uint32_t lastOledUpdateMs = 0;
const uint32_t OLED_UPDATE_INTERVAL_MS = 200;

// CRC-16-CCITT (Polynomial 0x1021, Init 0xFFFF)
uint16_t computeCRC16(const uint8_t* data, size_t length) {
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < length; i++) {
        crc ^= (uint16_t)data[i] << 8;
        for (uint8_t bit = 0; bit < 8; bit++) {
            if (crc & 0x8000) {
                crc = (crc << 1) ^ 0x1021;
            } else {
                crc = crc << 1;
            }
        }
    }
    return crc;
}

// On-board Heuristic Peak Detector (Fallback for OLED Alert)
void processLocalPeak(uint32_t ir, uint32_t nowMs) {
    // Basic single-pole IIR high-pass filter to remove DC baseline
    // y[n] = x[n] - DC; DC = alpha * DC + (1 - alpha) * x[n]
    const float alpha = 0.95f;
    dcFilterIR = alpha * dcFilterIR + (1.0f - alpha) * (float)ir;
    acFilteredIR = (float)ir - dcFilterIR;

    // Threshold peak detector with refractory period (300 ms = 200 BPM max)
    static float peakThreshold = 150.0f;
    static bool aboveThreshold = false;

    if (acFilteredIR > peakThreshold && !aboveThreshold && (nowMs - lastPeakTime > 300)) {
        aboveThreshold = true;
        uint32_t ibiMs = nowMs - lastPeakTime;
        lastPeakTime = nowMs;

        if (ibiMs > 300 && ibiMs < 2000) {
            float instantBpm = 60000.0f / (float)ibiMs;
            bpmHistory[bpmIndex] = instantBpm;
            bpmIndex = (bpmIndex + 1) % BPM_HISTORY_SIZE;

            // Rolling average BPM
            float sum = 0;
            uint8_t validCount = 0;
            for (int i = 0; i < BPM_HISTORY_SIZE; i++) {
                if (bpmHistory[i] > 30.0f && bpmHistory[i] < 220.0f) {
                    sum += bpmHistory[i];
                    validCount++;
                }
            }
            if (validCount > 0) {
                avgBpm = sum / validCount;
                currentBpm = instantBpm;
            }

            // Local irregularity heuristic: large variance between consecutive IBIs (>25% jump)
            if (fabs(instantBpm - avgBpm) > 20.0f) {
                consecutiveIrregularCount++;
                if (consecutiveIrregularCount >= 3) {
                    irregularRhythmAlert = true;
                }
            } else {
                if (consecutiveIrregularCount > 0) consecutiveIrregularCount--;
                if (consecutiveIrregularCount == 0) irregularRhythmAlert = false;
            }
        }
    } else if (acFilteredIR < (peakThreshold * 0.4f)) {
        aboveThreshold = false;
    }
}

// Render SSD1306 Display
void updateOled(uint32_t nowMs) {
    display.clearDisplay();

    // Top Status Bar
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);
    display.print("UE 3CPE-2A | 100Hz");

    // Finger placement check (IR threshold ~ 50,000)
    if (ringBuffer[(rbHead - 1) & (RING_BUFFER_SIZE - 1)].ir < 50000) {
        display.setCursor(10, 24);
        display.setTextSize(1);
        display.print("PLACE FINGER/WRIST");
        display.setCursor(20, 40);
        display.print("ON MAX30102...");
        display.display();
        return;
    }

    // BPM Readout
    display.setCursor(0, 16);
    display.setTextSize(1);
    display.print("PULSE RATE:");

    display.setCursor(0, 28);
    display.setTextSize(2);
    if (avgBpm > 30.0f && avgBpm < 220.0f) {
        display.printf("%3.0f", avgBpm);
        display.setTextSize(1);
        display.print(" BPM");
    } else {
        display.print("-- BPM");
    }

    // Arrhythmia Alert Banner (Local Heuristic Fallback)
    display.setCursor(0, 50);
    display.setTextSize(1);
    if (irregularRhythmAlert) {
        display.print("[!] IRREGULAR RHYTHM");
    } else {
        display.print("STATUS: NSR (NORMAL)");
    }

    display.display();
}

void setup() {
    // Initialize Hardware Serial for high-speed binary frame transport
    Serial.begin(115200);
    while (!Serial && millis() < 2000) {
        delay(10);
    }

    // Initialize I2C Bus on ESP32-C3 pins
    Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
    Wire.setClock(400000); // 400 kHz Fast-Mode I2C

    // Initialize OLED
    if (!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
        // Continue even if OLED fails so UART acquisition is never blocked
    } else {
        display.clearDisplay();
        display.setTextColor(SSD1306_WHITE);
        display.setTextSize(1);
        display.setCursor(10, 20);
        display.println("Arrhythmia IoT Node");
        display.setCursor(10, 36);
        display.println("Init MAX30102 @ 100Hz");
        display.display();
        delay(800);
    }

    // Initialize MAX30102 Sensor
    if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
        if (display.getBuffer()) {
            display.clearDisplay();
            display.setCursor(5, 25);
            display.print("MAX30102 NOT FOUND!");
            display.display();
        }
    } else {
        // Sensor Configuration per ANTIGRAVITY.md §4.1:
        // Red + IR mode, 100 Hz sampling rate, 411 us pulse width (18-bit ADC), 6.4mA LED current
        byte ledBrightness = 60;   // 0=Off to 255=50mA (60 ~= 12mA)
        byte sampleAverage = 1;    // 1 sample per FIFO read (no hardware averaging for pure 100Hz)
        byte ledMode = 2;          // 2 = Red + IR mode
        int sampleRate = 100;      // 100 Hz sampling rate
        int pulseWidth = 411;      // 411 us (18-bit resolution)
        int adcRange = 4096;       // 4096 nA full-scale

        particleSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange);
        particleSensor.setPulseAmplitudeRed(0x1F); // ~6.4mA
        particleSensor.setPulseAmplitudeIR(0x1F);  // ~6.4mA
    }
}

void loop() {
    static uint32_t lastSampleTimeUs = 0;
    uint32_t currentUs = micros();
    uint32_t nowMs = millis();

    // 100 Hz Sampling Timer (every 10,000 us)
    if (currentUs - lastSampleTimeUs >= SAMPLE_INTERVAL_US) {
        lastSampleTimeUs += SAMPLE_INTERVAL_US;
        if (currentUs - lastSampleTimeUs > SAMPLE_INTERVAL_US) {
            // Guard against timer rollover / lag
            lastSampleTimeUs = currentUs;
        }

        // Read raw 18-bit samples from MAX30102 FIFO
        uint32_t irVal = particleSensor.getIR();
        uint32_t redVal = particleSensor.getRed();

        // Enqueue into ring buffer
        ringBuffer[rbHead].ts = nowMs;
        ringBuffer[rbHead].ir = irVal;
        ringBuffer[rbHead].red = redVal;
        rbHead = (rbHead + 1) & (RING_BUFFER_SIZE - 1);

        // Process local peak detection for on-device OLED fallback
        processLocalPeak(irVal, nowMs);

        // Dequeue and transmit framed serial packet over UART
        if (rbTail != rbHead) {
            SampleData sample = ringBuffer[rbTail];
            rbTail = (rbTail + 1) & (RING_BUFFER_SIZE - 1);

            PpgPacket packet;
            packet.start_byte = FRAME_START_BYTE;
            packet.frame_type = FRAME_TYPE_RAW;
            packet.timestamp_ms = sample.ts;
            packet.ir_raw = sample.ir;
            packet.red_raw = sample.red;
            packet.heuristic_bpm_x10 = (uint16_t)(currentBpm * 10.0f);

            // Compute CRC16 over payload: bytes 1 to 15 (15 bytes total)
            const uint8_t* payloadPtr = (const uint8_t*)&packet + 1;
            size_t payloadLen = sizeof(PpgPacket) - 4; // Excluding start_byte, crc16, end_byte
            packet.crc16 = computeCRC16(payloadPtr, payloadLen);
            packet.end_byte = FRAME_END_BYTE;

            // Transmit raw binary packet to Raspberry Pi 4
            Serial.write((const uint8_t*)&packet, sizeof(PpgPacket));
        }
    }

    // Refresh OLED display at throttled 5 Hz interval
    if (nowMs - lastOledUpdateMs >= OLED_UPDATE_INTERVAL_MS) {
        lastOledUpdateMs = nowMs;
        updateOled(nowMs);
    }
}
