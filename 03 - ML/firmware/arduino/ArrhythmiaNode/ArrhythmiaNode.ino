/**
 * @file ArrhythmiaNode.ino
 * @brief Wearable Arrhythmia Acquisition Node — Arduino IDE Sketch (PLAN §3 & ADR-002)
 * 
 * Thesis: An Internet of Things-Based Framework for Cardiac Arrhythmia Detection
 *         via Photoplethysmography and Machine Learning
 * Authors: John Carlo Bauzon, Zidane Vincent Condino, Ranilo John Delos Angeles,
 *          Adrian Rana, Kyle Villaflor (Department of Computer Engineering, 3CPE-2A)
 * Institution: University of the East, Caloocan Campus
 * Adviser: Dr. Nelson Rodelas
 * 
 * Hardware Specifications:
 *  - Microcontroller: ESP32-C3 RISC-V (160 MHz)
 *  - Optical Biosensor: MAX30102 (Red: 660 nm, IR: 940 nm) via I2C
 *  - Local Display: 0.96" SSD1306 OLED (128x64, I2C address 0x3C)
 *  - I2C Pins: GPIO 8 (SDA), GPIO 9 (SCL)
 *  - Sampling Cadence: 100 Hz deterministic (10,000 us period)
 * 
 * Wire Protocol Contract (Matches serial_protocol.py & 03 - ML/firmware/src/main.cpp):
 *  - Packet Length: 19 bytes fixed (#pragma pack(1))
 *  - Byte 0:  0xAA (FRAME_START_BYTE)
 *  - Byte 1:  0x01 (FRAME_TYPE_RAW)
 *  - Bytes 2-5:   timestamp_ms (uint32_t, little-endian)
 *  - Bytes 6-9:   ir_raw       (uint32_t, little-endian)
 *  - Bytes 10-13: red_raw      (uint32_t, little-endian)
 *  - Bytes 14-15: heuristic_bpm_x10 (uint16_t, little-endian, e.g. 725 = 72.5 BPM)
 *  - Bytes 16-17: crc16 (uint16_t, little-endian, CRC-16-CCITT over bytes 1..15)
 *  - Byte 18: 0x55 (FRAME_END_BYTE)
 * 
 * CRITICAL ARDUINO IDE COMPILATION SETTINGS:
 *  - Board: "ESP32C3 Dev Module"
 *  - Tools -> USB CDC On Boot -> "Enabled"  (MANDATORY for /dev/ttyACM0 serial streaming)
 *  - Tools -> Flash Frequency -> "80MHz"
 *  - Tools -> CPU Frequency   -> "160MHz (WiFi)"
 */

#include <Wire.h>
#include "MAX30105.h"
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// ESP32-C3 Hardware I2C Pin Assignments
#define I2C_SDA_PIN 8
#define I2C_SCL_PIN 9

// OLED Configuration
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
#define SCREEN_ADDRESS 0x3C

// Sampling & Binary Protocol Framing
#define SAMPLING_RATE_HZ 100
#define SAMPLE_INTERVAL_US (1000000 / SAMPLING_RATE_HZ) // 10,000 us = 10.0 ms period
#define FRAME_START_BYTE 0xAA
#define FRAME_END_BYTE   0x55
#define FRAME_TYPE_RAW   0x01

// MAX30102 IR counts sit in the low thousands with no tissue present and rise
// well above this on contact. Used to suppress heuristic BPM when unworn.
#define FINGER_PRESENT_IR_THRESHOLD 50000UL

#pragma pack(push, 1)
struct PpgPacket {
    uint8_t  start_byte;        // 0xAA (Byte 0)
    uint8_t  frame_type;        // 0x01 (Byte 1)
    uint32_t timestamp_ms;      // Monotonic MCU time (Bytes 2-5)
    uint32_t ir_raw;            // 18-bit raw IR channel (Bytes 6-9)
    uint32_t red_raw;           // 18-bit raw Red channel (Bytes 10-13)
    uint16_t heuristic_bpm_x10; // e.g. 725 = 72.5 BPM (Bytes 14-15)
    uint16_t crc16;             // CRC-16-CCITT poly 0x1021 over bytes 1-15 (Bytes 16-17)
    uint8_t  end_byte;          // 0x55 (Byte 18)
};
#pragma pack(pop)

// Compile-time static check ensuring exact 19-byte alignment with serial_protocol.py
static_assert(sizeof(PpgPacket) == 19, "FATAL: PpgPacket structure must be exactly 19 bytes without compiler padding.");

// Peripheral Objects
MAX30105 particleSensor;
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// Ring Buffer for deterministic sample queuing
#define RING_BUFFER_SIZE 256
struct SampleData {
    uint32_t ts;
    uint32_t ir;
    uint32_t red;
};
volatile SampleData ringBuffer[RING_BUFFER_SIZE];
volatile uint16_t rbHead = 0;
volatile uint16_t rbTail = 0;

// On-board Heuristic Peak Detector (Local OLED fallback path)
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

// OLED Refresh Throttle (5 Hz refresh to conserve I2C bandwidth for 100 Hz sensor)
uint32_t lastOledUpdateMs = 0;
const uint32_t OLED_UPDATE_INTERVAL_MS = 200;

/**
 * @brief Computes CRC-16-CCITT (Polynomial 0x1021, Initial value 0xFFFF).
 * Matches python edge_inference/serial_protocol.py and firmware/src/main.cpp.
 */
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

/**
 * @brief On-board heuristic peak detector for standalone local OLED readout.
 * Provides resilient primary-care fallback if serial link or Pi is detached.
 */
void processLocalPeak(uint32_t ir, uint32_t nowMs) {
    // Single-pole IIR high-pass filter for DC baseline removal
    const float alpha = 0.95f;
    dcFilterIR = alpha * dcFilterIR + (1.0f - alpha) * (float)ir;
    acFilteredIR = (float)ir - dcFilterIR;

    // Peak detector with refractory period (300 ms = 200 BPM ceiling)
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

            // Rolling average
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

            // Local irregularity detection (>20 BPM fluctuation across successive beats)
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

/**
 * @brief Refreshes SSD1306 0.96" OLED display at 5 Hz.
 */
void updateOled(uint32_t nowMs) {
    display.clearDisplay();

    // Header Status Bar
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);
    display.print("UE 3CPE-2A | 100Hz");

    // Finger placement detection threshold (~50,000 raw counts on MAX30102 IR)
    uint32_t lastIr = ringBuffer[(rbHead - 1) & (RING_BUFFER_SIZE - 1)].ir;
    if (lastIr < FINGER_PRESENT_IR_THRESHOLD) {
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

    // Arrhythmia Alert Indicator (Local Fallback)
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
    // High-speed serial connection (115200 baud)
    Serial.begin(115200);
    uint32_t serialStart = millis();
    while (!Serial && (millis() - serialStart < 2500)) {
        delay(10);
    }

    // Initialize I2C Bus on ESP32-C3 designated GPIO pins (SDA=8, SCL=9)
    Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
    Wire.setClock(400000); // 400 kHz Fast-Mode I2C

    // Initialize SSD1306 OLED
    if (display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
        display.clearDisplay();
        display.setTextColor(SSD1306_WHITE);
        display.setTextSize(1);
        display.setCursor(10, 20);
        display.println("Arrhythmia IoT Node");
        display.setCursor(10, 36);
        display.println("Init MAX30102 @ 100Hz");
        display.display();
        delay(600);
    }

    // Initialize MAX30102 Biosensor
    if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
        display.clearDisplay();
        display.setCursor(5, 25);
        display.print("MAX30102 NOT FOUND!");
        display.setCursor(5, 40);
        display.print("Check SDA/SCL 8/9");
        display.display();
    } else {
        // Engineering configuration matching ANTIGRAVITY.md §4.1:
        // Red + IR mode, 100 Hz sampling rate, 411 us pulse width (18-bit ADC resolution)
        byte ledBrightness = 60;   // ~12 mA drive current
        byte sampleAverage = 1;    // 1 sample per FIFO read (no hardware averaging for pure 100 Hz)
        byte ledMode = 2;          // 2 = Red + IR dual-channel mode
        int sampleRate = 100;      // 100 Hz sampling rate
        int pulseWidth = 411;      // 411 us (18-bit ADC resolution)
        int adcRange = 4096;       // 4096 nA full-scale range

        particleSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange);
        particleSensor.setPulseAmplitudeRed(0x1F); // ~6.4 mA
        particleSensor.setPulseAmplitudeIR(0x1F);  // ~6.4 mA
    }
}

void loop() {
    static uint32_t lastSampleTimeUs = 0;
    uint32_t currentUs = micros();
    uint32_t nowMs = millis();

    // Deterministic 100 Hz Sampling Loop (every 10,000 us)
    if (currentUs - lastSampleTimeUs >= SAMPLE_INTERVAL_US) {
        lastSampleTimeUs += SAMPLE_INTERVAL_US;
        if (currentUs - lastSampleTimeUs > SAMPLE_INTERVAL_US) {
            // Guard against timer overflow / micros() jitter
            lastSampleTimeUs = currentUs;
        }

        // Read raw 18-bit samples from the MAX30102 FIFO.
        //
        // IMPORTANT: do NOT use getIR()/getRed() here. Each of those calls
        // safeCheck(), which blocks until a *new* sample lands in the FIFO, so
        // calling both consumes two sensor samples per loop pass and halves the
        // effective rate to 50 Hz (measured 49.8 Hz before this fix). The whole
        // DSP chain is configured for fs = 100 Hz, so that silently doubles every
        // reported BPM/IBI and shifts the Butterworth cutoffs.
        //
        // Instead pull one FIFO entry and read both channels from it.
        particleSensor.check();
        if (!particleSensor.available()) {
            return; // no new sample yet; try again next pass
        }
        uint32_t irVal  = particleSensor.getFIFOIR();
        uint32_t redVal = particleSensor.getFIFORed();
        particleSensor.nextSample();

        // Push to sample ring buffer
        ringBuffer[rbHead].ts = nowMs;
        ringBuffer[rbHead].ir = irVal;
        ringBuffer[rbHead].red = redVal;
        rbHead = (rbHead + 1) & (RING_BUFFER_SIZE - 1);

        // Process on-board heuristic for local OLED display
        processLocalPeak(irVal, nowMs);

        // Dequeue and transmit framed binary packet to Raspberry Pi
        if (rbTail != rbHead) {
            // Read each member individually: a volatile struct has no implicit
            // copy constructor, so whole-struct assignment does not compile.
            SampleData sample;
            sample.ts  = ringBuffer[rbTail].ts;
            sample.ir  = ringBuffer[rbTail].ir;
            sample.red = ringBuffer[rbTail].red;
            rbTail = (rbTail + 1) & (RING_BUFFER_SIZE - 1);

            PpgPacket packet;
            packet.start_byte = FRAME_START_BYTE;
            packet.frame_type = FRAME_TYPE_RAW;
            packet.timestamp_ms = sample.ts;
            packet.ir_raw = sample.ir;
            packet.red_raw = sample.red;
            // Report the local heuristic BPM only while there is actual skin
            // contact. With no finger present the MAX30102 still returns
            // low-amplitude ambient noise, and the threshold peak detector will
            // lock onto it and emit a plausible-looking heart rate. Sending 0
            // means "unknown" so the Pi never receives an invented vital sign.
            if (sample.ir < FINGER_PRESENT_IR_THRESHOLD) {
                currentBpm = 0.0f;
                for (uint8_t i = 0; i < BPM_HISTORY_SIZE; i++) bpmHistory[i] = 0.0f;
                packet.heuristic_bpm_x10 = 0;
            } else {
                packet.heuristic_bpm_x10 = (uint16_t)(currentBpm * 10.0f);
            }

            // Compute CRC-16-CCITT over 15-byte payload (bytes 1 to 15)
            const uint8_t* payloadPtr = (const uint8_t*)&packet + 1;
            size_t payloadLen = sizeof(PpgPacket) - 4; // 19 - 4 = 15 bytes
            packet.crc16 = computeCRC16(payloadPtr, payloadLen);
            packet.end_byte = FRAME_END_BYTE;

            // Stream binary packet over USB serial to Raspberry Pi (/dev/ttyACM0)
            Serial.write((const uint8_t*)&packet, sizeof(PpgPacket));
        }
    }

    // Refresh OLED display at throttled 5 Hz interval
    if (nowMs - lastOledUpdateMs >= OLED_UPDATE_INTERVAL_MS) {
        lastOledUpdateMs = nowMs;
        updateOled(nowMs);
    }
}
