# Arrhythmia Wearable Node — Arduino IDE Guide

This directory provides the official Arduino IDE project for the **ESP32-C3 + MAX30102 + SSD1306** wearable photoplethysmography (PPG) acquisition node for the University of the East Computer Engineering thesis:
> **An Internet of Things-Based Framework for Cardiac Arrhythmia Detection via Photoplethysmography and Machine Learning**

---

## 1. Dual-Toolchain Architecture Note

The project maintains two firmware implementations targeting the same hardware and wire contract:

| Implementation | Location | Primary Target |
| :--- | :--- | :--- |
| **Arduino IDE Sketch** | `03 - ML/firmware/arduino/ArrhythmiaNode/ArrhythmiaNode.ino` | Standard lab flashing, quick bench testing, student development |
| **PlatformIO Project** | `03 - ML/firmware/src/main.cpp` + `platformio.ini` | Headless CI builds, automated regression compilation |

Both implementations adhere to the **identical 19-byte binary wire contract** parsed by the edge runtime (`03 - ML/edge_inference/serial_protocol.py`):
```
[0xAA][0x01][timestamp_ms: 4B][ir_raw: 4B][red_raw: 4B][bpm_x10: 2B][crc16: 2B][0x55]
 Total: 19 bytes packed (#pragma pack(push, 1))
 CRC-16-CCITT: Polynomial 0x1021, Initial 0xFFFF, calculated over bytes 1..15
```

---

## 2. Hardware Wiring Reference

The ESP32-C3 RISC-V development board utilizes dedicated hardware I2C pins:

| ESP32-C3 Pin | MAX30102 Optical Sensor | SSD1306 0.96" OLED (0x3C) | Notes |
| :--- | :--- | :--- | :--- |
| **GPIO 8** | `SDA` | `SDA` | I2C Data bus (shared) |
| **GPIO 9** | `SCL` | `SCL` | I2C Clock bus (shared, 400 kHz) |
| **3.3V** | `VIN` / `VCC` | `VCC` | Regulated 3.3V DC |
| **GND** | `GND` | `GND` | Common ground |

---

## 3. Required Arduino IDE Libraries

Install the following libraries via the Arduino IDE Library Manager (**Tools -> Manage Libraries...** or `Ctrl + Shift + I`):

1. **SparkFun MAX3010x Pulse and Proximity Sensor Library** (by SparkFun Electronics, v1.1.2+)
2. **Adafruit SSD1306** (by Adafruit, v2.5.9+)
3. **Adafruit GFX Library** (by Adafruit, v1.11.9+)

---

## 4. CRITICAL Arduino IDE Board & Port Configuration

> [!WARNING]
> **MANDATORY SETTING FOR USB SERIAL STREAMING**
> 
> The ESP32-C3 includes an integrated USB Serial/JTAG controller. If **USB CDC On Boot** is set to *Disabled*, `Serial.print()` and `Serial.write()` route to UART0 (GPIO20/GPIO21). When plugged into the Raspberry Pi via USB-C, no packets will be received and `/dev/ttyACM0` will appear silent.
>
> You **MUST** set **USB CDC On Boot** to **Enabled**.

Open `ArrhythmiaNode.ino` in Arduino IDE and set the following under the **Tools** menu:

- **Board:** `ESP32 Arduino` -> `ESP32C3 Dev Module`
- **USB CDC On Boot:** `Enabled`  <--- *(CRITICAL: enables /dev/ttyACM0 serial streaming)*
- **CPU Frequency:** `160MHz (WiFi)`
- **Flash Frequency:** `80MHz`
- **Flash Mode:** `QIO`
- **Partition Scheme:** `Default 4MB with spiffs (1.2MB APP/1.5MB SPIFFS)`
- **Core Debug Level:** `None`
- **Port:** Select the COM port corresponding to your ESP32-C3 board

---

## 5. Step-by-Step Flashing Procedure

1. Connect the ESP32-C3 board to your computer using a data-capable USB-C cable.
2. If the board is not detected in bootloader mode:
   - Hold down the **BOOT** button.
   - Press and release the **RESET** button.
   - Release the **BOOT** button.
3. In Arduino IDE, click **Verify** (`Ctrl + R`) to compile the sketch and confirm clean build output.
4. Click **Upload** (`Ctrl + U`).
5. After upload completes, disconnect the board and connect it to one of the USB ports on the Raspberry Pi 5.
6. On the Raspberry Pi, verify the device node:
   ```bash
   ls -la /dev/ttyACM0
   ```
7. Check that the edge inference service locks and streams packets:
   ```bash
   sudo systemctl status arrhythmia-edge.service
   journalctl -u arrhythmia-edge.service -n 30 --no-pager
   ```
