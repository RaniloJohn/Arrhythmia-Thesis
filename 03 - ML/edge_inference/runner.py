"""
Edge Inference Pipeline & Local Pub/Sub Bridge (ADR-001 & PLAN §2)
==================================================================
Thesis: An Internet of Things-Based Framework for Cardiac Arrhythmia Detection
        via Photoplethysmography and Machine Learning (3CPE-2A, Univ. of the East)

Architecture:
- Ingests raw 100 Hz frames from ESP32-C3 via Serial (or synthetic simulation).
- Slices continuous stream into sliding 10-second windows (N=1000 @ 100 Hz).
- Executes DSP: Butterworth 0.5-5Hz bandpass, detrending, and SQI calculation.
- Runs 1D-CNN inference and 1D Grad-CAM explainability (<25ms budget).
- Persists AF events into SQLite with cryptographic SHA-256 backward hash chaining.
- Bridges telemetry to the Node.js Clinician Dashboard via local pub/sub.
"""

import os
import sys
import time
import json
import socket
import glob
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Union

try:
    import serial
    import serial.tools.list_ports
    HAS_PYSERIAL = True
except ImportError:
    HAS_PYSERIAL = False

# Add parent directory to module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from storage.db_manager import DatabaseManager
from signal_processing.filter import ButterBandpassFilter, detrend_ppg, zscore_normalize
from signal_processing.sqi import SignalQualityAssessor
from signal_processing.peak_detection import ElgendiPeakDetector
from model.inference_model import Arrhythmia1DCNN
from model.grad_cam import GradCAM1D
from edge_inference.serial_protocol import StreamPacketParser, pack_ppg_frame


def find_esp32_port() -> Optional[str]:
    """
    Auto-detects connected ESP32 serial port.
    Matches /dev/ttyUSB* or /dev/ttyACM* on Linux/Raspberry Pi per PLAN §7,
    or matching USB-UART bridge descriptors across platforms.
    """
    if HAS_PYSERIAL:
        try:
            ports = list(serial.tools.list_ports.comports())
            # 1. Match known ESP32 / USB-UART bridge descriptors
            for p in ports:
                desc = (p.description or "").lower()
                hwid = (p.hwid or "").lower()
                if any(k in desc or k in hwid for k in ["esp32", "cp210", "ch340", "ch341", "ftdi", "usb-serial", "jtag", "uart"]):
                    return p.device
            # 2. Check for Linux /dev/ttyUSB* or /dev/ttyACM*
            for p in ports:
                if p.device.startswith("/dev/ttyUSB") or p.device.startswith("/dev/ttyACM"):
                    return p.device
        except Exception:
            pass

    # Direct filesystem glob fallback for Linux/Raspberry Pi (/dev/ttyUSB* or /dev/ttyACM*)
    linux_candidates = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
    if linux_candidates:
        return linux_candidates[0]

    return None


class SyntheticPPGGenerator:
    """
    Generates realistic physiological PPG signals for bench-testing and simulation.
    Can toggle between Normal Sinus Rhythm (NSR) and Atrial Fibrillation (AF).
    AF morphology: Irregular RR intervals, amplitude variability, absent dicrotic notch.
    """
    def __init__(self, fs: float = 100.0):
        self.fs = fs
        self.t = 0.0
        self.phase = 0.0
        self.mode = "nsr"  # 'nsr' or 'af'
        self.base_hr = 72.0
        self.dc_baseline = 52000.0

    def set_mode(self, mode: str):
        if mode in ("nsr", "af"):
            self.mode = mode

    def next_sample(self) -> Dict[str, Any]:
        dt = 1.0 / self.fs
        self.t += dt

        if self.mode == "nsr":
            # Normal Sinus Rhythm: stable ~72 BPM with respiratory sinus arrhythmia (RSA)
            hr = self.base_hr + 3.0 * np.sin(2 * np.pi * 0.2 * self.t)
            freq = hr / 60.0
            self.phase += 2 * np.pi * freq * dt
            # Typical PPG pulse: systolic peak + dicrotic wave
            systolic = np.exp(-(( (self.phase % (2 * np.pi)) - 1.2 ) ** 2) / 0.15)
            dicrotic = 0.35 * np.exp(-(( (self.phase % (2 * np.pi)) - 2.3 ) ** 2) / 0.25)
            pulse = systolic + dicrotic
            noise = np.random.normal(0, 0.02)
        else:
            # Atrial Fibrillation: irregular erratic intervals, pulse amplitude variation, no distinct dicrotic wave
            # Rapid random frequency variation (90 to 140 BPM)
            instant_freq = (95.0 + 35.0 * np.sin(self.t * 1.7) + np.random.uniform(-15, 15)) / 60.0
            self.phase += 2 * np.pi * instant_freq * dt
            # Erratic pulse amplitude
            amp = 0.65 + 0.45 * np.sin(self.t * 2.3)
            systolic = amp * np.exp(-(( (self.phase % (2 * np.pi)) - 1.2 ) ** 2) / 0.22)
            # Absent or disorganized dicrotic notch
            dicrotic = 0.10 * np.exp(-(( (self.phase % (2 * np.pi)) - 2.8 ) ** 2) / 0.40)
            noise = np.random.normal(0, 0.08)
            pulse = systolic + dicrotic + noise

        # Add respiratory baseline wander (0.25 Hz)
        wander = 400.0 * np.sin(2 * np.pi * 0.25 * self.t)
        ir_ac = pulse * 1800.0
        ir_total = int(self.dc_baseline + wander + ir_ac)
        red_total = int(self.dc_baseline * 0.85 + wander * 0.85 + ir_ac * 0.80)

        now_ms = int(time.time() * 1000)
        return {
            "timestamp_ms": now_ms,
            "ir_raw": ir_total,
            "red_raw": red_total,
            "heuristic_bpm": round(self.base_hr, 1)
        }


class EdgeInferenceRunner:
    def __init__(self,
                 db_path: Optional[str] = None,
                 patient_id: str = "PAT-CAL-001",
                 device_id: str = "ESP32C3-NODE-01",
                 window_size: int = 1000,
                 step_size: int = 100,  # 1-second step @ 100Hz
                 dashboard_port: int = 5051,
                 http_ingest_url: Optional[str] = None,
                 weights_path: Optional[Union[str, Path]] = None):
        self.patient_id = patient_id
        self.device_id = device_id
        self.window_size = window_size
        self.step_size = step_size
        self.dashboard_port = dashboard_port
        self.http_ingest_url = http_ingest_url or os.environ.get("HTTP_INGEST_URL", "http://127.0.0.1:8080/api/edge/ingest-window")

        # Modules
        self.db = DatabaseManager(db_path) if db_path else DatabaseManager()
        self.bp_filter = ButterBandpassFilter(lowcut=0.5, highcut=5.0, fs=100.0, order=4)
        self.sqi_assessor = SignalQualityAssessor()
        self.peak_detector = ElgendiPeakDetector(fs=100.0)
        self.model = Arrhythmia1DCNN(input_length=window_size)

        # Resolve weights path: explicit argument -> env var ARRHYTHMIA_WEIGHTS -> default file
        resolved_weights: Optional[Path] = None
        if weights_path is not None:
            w_path = Path(weights_path)
            if not w_path.exists():
                raise FileNotFoundError(f"Explicitly specified weights file not found: {w_path.resolve()}")
            resolved_weights = w_path
        elif "ARRHYTHMIA_WEIGHTS" in os.environ and os.environ["ARRHYTHMIA_WEIGHTS"].strip():
            env_w = Path(os.environ["ARRHYTHMIA_WEIGHTS"].strip())
            if env_w.exists():
                resolved_weights = env_w
            else:
                print(f"[Edge Runner WARNING] ARRHYTHMIA_WEIGHTS set to '{env_w}' but file does not exist.")

        if resolved_weights is None:
            default_weights = Path(__file__).resolve().parent.parent / "model" / "weights" / "cnn_af_v1.npz"
            if default_weights.exists():
                resolved_weights = default_weights

        if resolved_weights is not None and resolved_weights.exists():
            self.model.load_weights(resolved_weights)
            print(f"[Edge Runner] Successfully loaded trained weights from: {resolved_weights.resolve()}")
            print(f"[Edge Runner] Model operational status: WEIGHTS_LOADED | Calibrated threshold: {self.model.threshold:.4f} (source: {self.model.weights_source})")
        else:
            print("\n" + "!" * 78)
            print("[Edge Runner WARNING] NO TRAINED WEIGHTS FILE FOUND!")
            print("[Edge Runner WARNING] Running on UNTRAINED random He-init weights.")
            print("[Edge Runner WARNING] Model predictions are for demo purposes and NOT clinically valid.")
            print("[Edge Runner WARNING] Specify weights via --weights <path> or ARRHYTHMIA_WEIGHTS env var.")
            print("!" * 78 + "\n")

        self.grad_cam = GradCAM1D(self.model)

        # Consensus tracking buffer (rolling window of 5 decisions for temporal stability)
        self.decision_history: List[int] = []
        self.consensus_window_size: int = 5
        self.consensus_k: int = 3  # 3 out of 5 majority

        # Buffers
        self.raw_ir_buffer: List[int] = []
        self.raw_red_buffer: List[int] = []
        self.timestamps_buffer: List[int] = []
        self.sample_count = 0

        # TCP pub/sub bridge socket (to Node.js dashboard)
        self.tcp_bridge_sock: Optional[socket.socket] = None

    def connect_bridge(self):
        """Attempts to establish local TCP pub/sub bridge to Node.js backend."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            sock.connect(("127.0.0.1", self.dashboard_port))
            sock.setblocking(False)
            self.tcp_bridge_sock = sock
            print(f"[Bridge] Connected to Node.js dashboard TCP pub/sub on port {self.dashboard_port}")
        except Exception:
            self.tcp_bridge_sock = None

    def broadcast_telemetry(self, payload: Dict[str, Any]):
        """Transmits JSON telemetry to Node.js dashboard via TCP bridge or HTTP POST."""
        json_data = json.dumps(payload) + "\n"
        # 1. Try TCP bridge socket
        if self.tcp_bridge_sock:
            try:
                self.tcp_bridge_sock.sendall(json_data.encode("utf-8"))
                return
            except Exception:
                self.tcp_bridge_sock = None

        # 2. Reconnection attempt if disconnected
        self.connect_bridge()
        if self.tcp_bridge_sock:
            try:
                self.tcp_bridge_sock.sendall(json_data.encode("utf-8"))
                return
            except Exception:
                pass

        # 3. HTTP Fallback via urllib
        try:
            import urllib.request
            req = urllib.request.Request(
                self.http_ingest_url,
                data=json_data.encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=0.2):
                pass
        except Exception:
            pass

    def ingest_sample(self, ts_ms: int, ir_val: int, red_val: int) -> Optional[Dict[str, Any]]:
        """
        Ingests a single 100 Hz sample. When window reaches window_size (1000)
        and step_size (100) is reached, executes the full edge pipeline.
        """
        self.raw_ir_buffer.append(ir_val)
        self.raw_red_buffer.append(red_val)
        self.timestamps_buffer.append(ts_ms)
        self.sample_count += 1

        # Keep buffer length at window_size
        if len(self.raw_ir_buffer) > self.window_size:
            self.raw_ir_buffer.pop(0)
            self.raw_red_buffer.pop(0)
            self.timestamps_buffer.pop(0)

        # Check if full window is ready and step reached
        if len(self.raw_ir_buffer) == self.window_size and (self.sample_count % self.step_size == 0):
            return self.process_window()

        return None

    def process_window(self) -> Dict[str, Any]:
        """
        Executes DSP -> SQI -> Peak Detection -> 1D-CNN -> Grad-CAM -> Hash Chaining -> Pub/Sub.
        """
        t_start = time.perf_counter_ns()

        raw_ir = np.array(self.raw_ir_buffer, dtype=np.float32)
        raw_red = np.array(self.raw_red_buffer, dtype=np.float32)
        ts_latest = self.timestamps_buffer[-1]

        # 1. DSP Stage: Detrending + Zero-Phase Butterworth Bandpass Filter (0.5 - 5.0 Hz)
        t_dsp_start = time.perf_counter_ns()
        detrended = detrend_ppg(raw_ir)
        filtered = self.bp_filter.apply(detrended)
        normalized = zscore_normalize(filtered)
        t_dsp_end = time.perf_counter_ns()
        dsp_latency_ms = (t_dsp_end - t_dsp_start) / 1_000_000.0

        # 2. Signal Quality Index (SQI)
        sqi_res = self.sqi_assessor.compute_metrics(raw_ir, filtered)

        # 3. Peak Detection & Real-Time BPM
        peak_res = self.peak_detector.analyze_intervals(
            self.peak_detector.detect_peaks(filtered)
        )
        current_bpm = peak_res["bpm"] if peak_res["bpm"] > 0 else 72.0

        # 4. 1D-CNN Inference & 1D Grad-CAM Explainability
        t_ml_start = time.perf_counter_ns()
        gradcam_res = self.grad_cam.explain(normalized)
        t_ml_end = time.perf_counter_ns()
        inference_latency_ms = (t_ml_end - t_ml_start) / 1_000_000.0

        af_prob = gradcam_res["af_probability"]
        af_detected = gradcam_res["af_detected"]
        gradcam_weights = gradcam_res["weights"]

        # Update rolling decision history for temporal consensus
        self.decision_history.append(int(af_detected))
        if len(self.decision_history) > self.consensus_window_size:
            self.decision_history.pop(0)
        consensus_af = int(sum(self.decision_history) >= self.consensus_k) if len(self.decision_history) >= self.consensus_k else int(af_detected)

        # 5. Cryptographic Hash Chaining Persistence (if AF detected or periodic anchor)
        db_event_id = None
        if af_detected:
            # Serialise Grad-CAM top regions for clinical record
            event_rec = self.db.record_event(
                patient_id=self.patient_id,
                device_id=self.device_id,
                bpm=current_bpm,
                af_detected=af_detected,
                confidence=af_prob,
                gradcam_path=json.dumps({"high_regions": gradcam_res["high_relevance_regions"][:5]})
            )
            db_event_id = event_rec["event_id"]

        t_total_end = time.perf_counter_ns()
        total_latency_ms = (t_total_end - t_start) / 1_000_000.0

        # Downsample waveform for transmission efficiency (1000 points -> 250 points, 25 Hz visual display)
        # while preserving full fidelity for peaks
        downsample_factor = 4
        display_raw = [round(float(v), 1) for v in filtered[::downsample_factor]]
        display_cam = [round(float(w), 3) for w in gradcam_weights[::downsample_factor]]

        telemetry_payload = {
            "ts": ts_latest,
            "iso_time": datetime.now(timezone.utc).isoformat(),
            "patient_id": self.patient_id,
            "device_id": self.device_id,
            "bpm": round(current_bpm, 1),
            "af_detected": int(af_detected),
            "af_consensus": consensus_af,
            "af_probability": round(af_prob, 4),
            "weights_loaded": bool(self.model.weights_loaded),
            "model_trained": bool(self.model.weights_loaded),
            "model_version": getattr(self.model, "model_version", "unknown"),
            "training_dataset": getattr(self.model, "training_dataset", "unknown"),
            "decision_threshold": round(float(self.model.threshold), 4),
            "sqi": sqi_res,
            "hrv": {
                "rmssd_ms": peak_res["rmssd_ms"],
                "cv_ibi": peak_res["cv_ibi"],
                "irregular": peak_res["irregular_intervals"]
            },
            "raw_window": display_raw,
            "gradcam_weights": display_cam,
            "event_id": db_event_id,
            "latencies": {
                "dsp_ms": round(dsp_latency_ms, 2),
                "inference_ms": round(inference_latency_ms, 2),
                "total_edge_ms": round(total_latency_ms, 2),
                "budget_ms": 25.0,
                "budget_met": bool(inference_latency_ms < 25.0)
            }
        }

        # 6. Publish to Node.js Clinician Dashboard
        self.broadcast_telemetry(telemetry_payload)

        return telemetry_payload

    def run_simulation(self, duration_sec: int = 30, mode_toggle_sec: int = 15, prefill: bool = False):
        """Runs realistic simulation streaming 100 Hz synthetic samples."""
        gen = SyntheticPPGGenerator(fs=100.0)
        print(f"[Edge Runner] Starting simulation for {duration_sec}s (Mode switch every {mode_toggle_sec}s)...")
        print(f"[Edge Runner] Target patient: {self.patient_id} | Device: {self.device_id}")

        if prefill:
            # Prefill 900 samples for instantaneous test window emission
            print("[Edge Runner] Prefilling buffer with 900 baseline samples...")
            for _ in range(900):
                s = gen.next_sample()
                self.ingest_sample(s["timestamp_ms"], s["ir_raw"], s["red_raw"])

        start_time = time.time()
        sample_interval = 0.010  # 10 ms = 100 Hz
        next_tick = time.time() + sample_interval

        while time.time() - start_time < duration_sec:
            elapsed = time.time() - start_time
            # Toggle between NSR and AF
            current_mode = "af" if int(elapsed // mode_toggle_sec) % 2 == 1 else "nsr"
            gen.set_mode(current_mode)

            sample = gen.next_sample()
            res = self.ingest_sample(sample["timestamp_ms"], sample["ir_raw"], sample["red_raw"])
            if res:
                print(f"[Edge Window] Mode={current_mode.upper()} | BPM={res['bpm']} | "
                      f"AF_Prob={res['af_probability']:.2%} | AF={res['af_detected']} | "
                      f"SQI={res['sqi']['sqi_score']} | Infer={res['latencies']['inference_ms']}ms (<25ms: {res['latencies']['budget_met']})")

            sleep_time = next_tick - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            next_tick += sample_interval

    def run_serial(self,
                   port: Optional[str] = None,
                   baud: int = 115200,
                   duration_sec: Optional[int] = None,
                   prefill: bool = False):
        """
        Runs real serial ingestion from connected ESP32-C3 microcontroller.
        Feeds incoming bytes into StreamPacketParser and triggers window inference.
        Auto-detects /dev/ttyUSB* or /dev/ttyACM* if port is not explicitly specified.
        Tolerates device disconnects or delayed boot by retrying every 2 seconds.
        """
        if not HAS_PYSERIAL:
            print("[Edge Runner ERROR] 'pyserial' is not installed. Please run: pip install pyserial")
            sys.exit(1)

        print(f"[Edge Runner] Starting real serial ingestion mode (baud={baud})...", flush=True)
        print(f"[Edge Runner] Target patient: {self.patient_id} | Device: {self.device_id}", flush=True)

        if prefill:
            print("[Edge Runner] Prefilling buffer with 900 baseline samples...", flush=True)
            gen = SyntheticPPGGenerator(fs=100.0)
            for _ in range(900):
                s = gen.next_sample()
                self.ingest_sample(s["timestamp_ms"], s["ir_raw"], s["red_raw"])

        parser = StreamPacketParser()
        start_time = time.time()
        last_poll_log = 0.0

        while True:
            if duration_sec and (time.time() - start_time >= duration_sec):
                print(f"[Edge Runner] Target duration of {duration_sec}s reached.", flush=True)
                break

            target_port = port or find_esp32_port()

            if not target_port:
                now = time.time()
                if now - last_poll_log >= 2.0:
                    print("[Edge Runner] Waiting for ESP32 serial device (/dev/ttyUSB* or /dev/ttyACM*)... Retrying in 2s", flush=True)
                    last_poll_log = now
                time.sleep(2.0)
                continue

            print(f"[Edge Runner] Found serial device: {target_port}. Attempting connection at {baud} baud...", flush=True)
            try:
                with serial.Serial(target_port, baudrate=baud, timeout=1.0) as ser:
                    print(f"[Edge Runner] Serial port {target_port} open. Ingesting raw 100 Hz frames...", flush=True)
                    ser.reset_input_buffer()
                    frames_received = 0

                    while True:
                        if duration_sec and (time.time() - start_time >= duration_sec):
                            print(f"[Edge Runner] Target duration of {duration_sec}s reached.", flush=True)
                            return

                        available = ser.in_waiting
                        raw_data = ser.read(max(available, 1))

                        if not raw_data:
                            continue

                        parser.feed(raw_data)

                        while True:
                            frame = parser.parse_next()
                            if not frame:
                                break

                            frames_received += 1
                            res = self.ingest_sample(
                                frame["timestamp_ms"],
                                frame["ir_raw"],
                                frame["red_raw"]
                            )
                            if res:
                                print(f"[Edge Window] Source=SERIAL | Frames={frames_received} | BPM={res['bpm']} | "
                                      f"AF_Prob={res['af_probability']:.2%} | AF={res['af_detected']} | "
                                      f"SQI={res['sqi']['sqi_score']} | Infer={res['latencies']['inference_ms']}ms (<25ms: {res['latencies']['budget_met']})", flush=True)
            except (serial.SerialException, OSError, IOError) as err:
                print(f"[Edge Runner] Serial disconnect or error on {target_port}: {err}. Retrying in 2s...", flush=True)
                time.sleep(2.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge Inference Runner & Pub/Sub Bridge")
    parser.add_argument("--simulate", action="store_true", default=False, help="Run in synthetic simulation mode (fallback)")
    parser.add_argument("--port", type=str, default=None, help="Serial port to connect to (e.g. /dev/ttyUSB0, /dev/ttyACM0). Auto-detected if not specified.")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate (default: 115200)")
    parser.add_argument("--duration", type=int, default=None, help="Execution duration in seconds (default: indefinite for serial, 30s for simulation)")
    parser.add_argument("--patient", type=str, default="PAT-CAL-001", help="Patient ID")
    parser.add_argument("--device", type=str, default="ESP32C3-NODE-01", help="Device ID")
    parser.add_argument("--prefill", action="store_true", default=False, help="Prefill initial window for instant emission")
    parser.add_argument("--weights", type=str, default=None, help="Path to trained model weights .npz file (default: 03 - ML/model/weights/cnn_af_v1.npz)")
    args = parser.parse_args()

    runner = EdgeInferenceRunner(
        patient_id=args.patient,
        device_id=args.device,
        weights_path=args.weights,
    )
    try:
        if args.simulate:
            sim_duration = args.duration if args.duration is not None else 30
            runner.run_simulation(duration_sec=sim_duration, prefill=args.prefill)
        else:
            runner.run_serial(port=args.port, baud=args.baud, duration_sec=args.duration, prefill=args.prefill)
    except KeyboardInterrupt:
        print("\n[Edge Runner] Stopped by user (SIGINT). Exiting.")

