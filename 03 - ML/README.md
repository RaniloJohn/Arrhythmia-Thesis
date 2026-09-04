# ML / Edge Source Code

The edge ML/DSP/firmware codebase — implemented by Antigravity, deployed and running
live on the Raspberry Pi (`/home/ranilo/Arrhythmia Thesis/`, Tailscale `raspberrypi`).
This folder does **not** contain the website — see [[../07 - Website/README|07 - Website/]]
(Node.js backend + HTML/CSS/JS frontend, a separate codebase per the 2026-09-03
folder-placement decision recorded in [[../PLAN.md|PLAN.md]]).

## Layout

- `firmware/src/main.cpp` — ESP32-C3 firmware (MAX30102 acquisition over I2C, SSD1306
  OLED BPM/alert display, 100 Hz framed binary serial protocol to the Pi).
- `signal_processing/` — Butterworth bandpass filtering, detrending, Signal Quality
  Index (SQI), Elgendi peak detection / IBI extraction (Python).
- `model/` — 1D-CNN inference (`inference_model.py`) and 1D Grad-CAM explainability
  (`grad_cam.py`).
- `edge_inference/` — `runner.py`, the real-time pipeline: serial ingestion (auto-
  detects `/dev/ttyUSB*`/`/dev/ttyACM*`, with a `--simulate` synthetic fallback) → DSP →
  1D-CNN → Grad-CAM → local TCP pub/sub bridge to the website backend. Also
  `serial_protocol.py` (the `0xAA`/`0x55` + CRC16 framed packet parser matching the
  firmware exactly).
- `storage/` — `db_manager.py`: SQLite schema + SHA-256 backward hash-chaining for the
  `arrhythmia_events` table (WAL mode, per ADR-001). Node.js owns the `patients` table
  in the same database file — see ADR-001 for the ownership split.
- `deploy/systemd/` — `arrhythmia-edge.service` and `arrhythmia-website.service`, the
  boot-time automation units installed on the Pi (`systemctl enable --now`) so both the
  edge pipeline and the website start automatically on every boot.

## Reference

- Architecture rationale: [[../06 - Antigravity Notes/ADR-001 - Edge Processing Topology & Local Cryptographic Hash Chaining for Offline Primary Care|ADR-001]]
- Full engineering blueprint: [[../06 - Antigravity Notes/2026-09-03 - Dual-Agent System Architecture & Engineering Blueprint|Dual-Agent System Architecture & Engineering Blueprint]]
- Implementation checklist & status: [[../PLAN.md|PLAN.md]]
- Formal review log: [[../02 - Code Review/Index|Code Review Index]]
