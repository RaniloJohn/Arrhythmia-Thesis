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

// On-board pulse-rate estimator (local OLED readout + heuristic_bpm_x10 field).
// Pulse rate only: the rhythm decision belongs to ibi_af_v1 on the Pi, so the
// node never shows a rhythm verdict of its own.
//
// Beats are timed by sample count at the sensor's 100 Hz clock, not millis(),
// so display redraws cannot distort them. 0.5-5 Hz Butterworth band-pass, an
// adaptive threshold (50% of the largest swing in the last 2 s), a 300 ms
// refractory period, parabolic peak interpolation, and the median of the last
// 8 intervals.
struct Biquad {
    float b0, b1, b2, a1, a2, z1, z2;
    void design(float f0, float fs, bool high) {
        float w0 = 2.0f * PI * f0 / fs;
        float c = cosf(w0);
        float alpha = sinf(w0) / (2.0f * 0.70710678f);   // Q = 1/sqrt(2)
        float a0 = 1.0f + alpha;
        if (high) { b0 = (1.0f + c) / 2.0f; b1 = -(1.0f + c); }
        else      { b0 = (1.0f - c) / 2.0f; b1 =  (1.0f - c); }
        b2 = b0;
        b0 /= a0; b1 /= a0; b2 /= a0;
        a1 = -2.0f * c / a0;
        a2 = (1.0f - alpha) / a0;
        z1 = z2 = 0;
    }
    float step(float x) {
        float y = b0 * x + z1;
        z1 = b1 * x - a1 * y + z2;
        z2 = b2 * x - a2 * y;
        return y;
    }
};

#define PULSE_SETTLE_SAMPLES 100       // 1 s for the filters to settle
#define PULSE_LOSS_SAMPLES 50          // finger gone 0.5 s before resetting
#define PULSE_REFRACTORY_SAMPLES 30    // 300 ms -> 200 bpm ceiling
#define PULSE_NO_BEAT_SAMPLES 300      // 3 s without a beat -> unknown
#define PULSE_WIN 200                  // 2 s adaptive-threshold window
#define PULSE_IBI_COUNT 8

Biquad pulseHighPass, pulseLowPass;
float pulseWin[PULSE_WIN];
uint16_t pulseWinPos = 0;
float pulseIbis[PULSE_IBI_COUNT];
uint8_t pulseIbiPos = 0, pulseIbiFilled = 0;
bool pulseFingerOn = false;
float pulseIrSeed = 0;
uint32_t pulseN = 0;
float pulseY1 = 0, pulseY2 = 0;
float pulseLastPeakT = -1;
uint32_t pulseLastPeakN = 0;
float pulseBpm = 0.0f;                 // 0 = unknown

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

void resetPulseEstimator() {
    pulseHighPass.design(0.5f, SAMPLING_RATE_HZ, true);
    pulseLowPass.design(5.0f, SAMPLING_RATE_HZ, false);
    for (int i = 0; i < PULSE_WIN; i++) pulseWin[i] = 0;
    pulseWinPos = 0; pulseIbiPos = 0; pulseIbiFilled = 0;
    pulseN = 0; pulseY1 = pulseY2 = 0;
    pulseLastPeakT = -1; pulseLastPeakN = 0;
    pulseBpm = 0.0f;
}

float medianPulseIbi() {
    float s[PULSE_IBI_COUNT];
    for (int i = 0; i < pulseIbiFilled; i++) s[i] = pulseIbis[i];
    for (int i = 1; i < pulseIbiFilled; i++) {        // insertion sort, <= 8 items
        float v = s[i]; int j = i - 1;
        while (j >= 0 && s[j] > v) { s[j + 1] = s[j]; j--; }
        s[j + 1] = v;
    }
    return (pulseIbiFilled % 2) ? s[pulseIbiFilled / 2]
                                : 0.5f * (s[pulseIbiFilled / 2 - 1] + s[pulseIbiFilled / 2]);
}

/**
 * @brief Local pulse-rate estimate for the OLED readout, one call per sample.
 * Keeps working when the Pi is detached; it never classifies rhythm.
 */
void processLocalPulse(uint32_t ir) {
    static uint16_t lowCount = 0;
    static uint32_t lastGoodIr = 0;
    if (ir < FINGER_PRESENT_IR_THRESHOLD) {
        if (!pulseFingerOn) return;
        if (++lowCount >= PULSE_LOSS_SAMPLES) { pulseFingerOn = false; resetPulseEstimator(); return; }
        ir = lastGoodIr;                  // brief slip: hold the last value, keep timing intact
    } else {
        lowCount = 0;
        lastGoodIr = ir;
    }
    if (!pulseFingerOn) { pulseFingerOn = true; resetPulseEstimator(); pulseIrSeed = ir; }

    // Invert: reflected IR dips as blood volume rises, so systole becomes a peak.
    float y = -pulseLowPass.step(pulseHighPass.step((float)ir - pulseIrSeed));
    pulseN++;
    pulseWin[pulseWinPos] = fabsf(y);
    pulseWinPos = (pulseWinPos + 1) % PULSE_WIN;

    if (pulseN > PULSE_SETTLE_SAMPLES) {
        float peakSwing = 0;
        for (int i = 0; i < PULSE_WIN; i++) if (pulseWin[i] > peakSwing) peakSwing = pulseWin[i];

        bool isPeak = (pulseY1 > pulseY2) && (pulseY1 >= y) && (pulseY1 > 0.5f * peakSwing);
        bool refractoryOk = (pulseLastPeakN == 0) ||
                            (pulseN - 1 - pulseLastPeakN >= PULSE_REFRACTORY_SAMPLES);
        if (isPeak && refractoryOk) {
            float denom = pulseY2 - 2.0f * pulseY1 + y;
            float delta = (denom != 0) ? 0.5f * (pulseY2 - y) / denom : 0;
            delta = constrain(delta, -0.5f, 0.5f);
            float peakT = (float)(pulseN - 1) + delta;
            pulseLastPeakN = pulseN - 1;
            if (pulseLastPeakT >= 0) {
                float ibiMs = (peakT - pulseLastPeakT) * 1000.0f / SAMPLING_RATE_HZ;
                if (ibiMs >= 300.0f && ibiMs <= 2000.0f) {
                    pulseIbis[pulseIbiPos] = ibiMs;
                    pulseIbiPos = (pulseIbiPos + 1) % PULSE_IBI_COUNT;
                    if (pulseIbiFilled < PULSE_IBI_COUNT) pulseIbiFilled++;
                    if (pulseIbiFilled >= 2) pulseBpm = 60000.0f / medianPulseIbi();
                }
            }
            pulseLastPeakT = peakT;
        }

        if (pulseLastPeakN && (pulseN - pulseLastPeakN > PULSE_NO_BEAT_SAMPLES)) {
            pulseBpm = 0.0f; pulseIbiFilled = 0; pulseIbiPos = 0;
            pulseLastPeakT = -1; pulseLastPeakN = 0;
        }
    }
    pulseY2 = pulseY1;
    pulseY1 = y;
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

    if (!pulseFingerOn) {
        display.setCursor(10, 24);
        display.setTextSize(1);
        display.print("PLACE FINGER/WRIST");
        display.setCursor(20, 40);
        display.print("ON MAX30102...");
        display.display();
        return;
    }

    // Pulse rate only. The rhythm verdict comes from ibi_af_v1 on the Pi.
    display.setCursor(0, 18);
    display.setTextSize(4);
    if (pulseBpm > 30.0f && pulseBpm < 220.0f) {
        display.printf("%3d", (int)(pulseBpm + 0.5f));
    } else {
        display.print(" --");
    }
    display.setTextSize(1);
    display.setCursor(76, 40);
    display.print("BPM");

    display.setCursor(0, 56);
    display.print("Rhythm: see dashboard");

    display.display();
}

void setup() {
    // High-speed serial connection (115200 baud)
    Serial.begin(115200);
    uint32_t serialStart = millis();
    while (!Serial && (millis() - serialStart < 2500)) {
        delay(10);
    }

    resetPulseEstimator();

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
    uint32_t nowMs = millis();

    // The MAX30102's own 100 Hz clock is the sample clock: drain every sample
    // waiting in its FIFO on each pass and send each one. A micros() ticker
    // that reads one sample per 10 ms tick drops ticks whenever the OLED redraw
    // (~23 ms at 400 kHz) runs, the 32-deep FIFO then overflows, and the Pi
    // receives ~90 samples/s while its DSP assumes 100 — shortening every
    // inter-beat interval that ibi_af_v1 consumes.
    //
    // Do NOT use getIR()/getRed() here: each calls safeCheck(), which blocks
    // for a *new* sample, so calling both consumes two samples per pass and
    // halves the rate (measured 49.8 Hz before that fix). Read both channels
    // from one FIFO entry instead.
    particleSensor.check();
    while (particleSensor.available()) {
        uint32_t irVal  = particleSensor.getFIFOIR();
        uint32_t redVal = particleSensor.getFIFORed();
        particleSensor.nextSample();

        // Push to sample ring buffer
        ringBuffer[rbHead].ts = nowMs;
        ringBuffer[rbHead].ir = irVal;
        ringBuffer[rbHead].red = redVal;
        rbHead = (rbHead + 1) & (RING_BUFFER_SIZE - 1);

        // Local pulse-rate estimate for the OLED readout
        processLocalPulse(irVal);

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
            if (sample.ir < FINGER_PRESENT_IR_THRESHOLD || !pulseFingerOn) {
                packet.heuristic_bpm_x10 = 0;
            } else {
                packet.heuristic_bpm_x10 = (uint16_t)(pulseBpm * 10.0f + 0.5f);
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
