"""
On-device latency benchmark for the edge pipeline (RQ5, Performance Efficiency).

Replays recorded raw MAX30102 captures (scripts/capture_raw.py output) through the real
`EdgeInferenceRunner.process_window` at the live cadence (one 10 s window per 1 s step), and
reports per-window latency for DSP, inference + explanation, and the end-to-end window.

Isolation: the runner gets a throwaway SQLite database in the storage folder (same SD card
as the live ledger, so a ledger write costs what it really costs) that is deleted afterwards;
dashboard broadcast and raw auto-capture are disabled. The live service keeps running, so
the figures include real background load.

    python3 scripts/benchmark_pipeline.py data/device_captures/*.npz [--repeat 3]
"""
import argparse
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ML_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ML_DIR))
os.environ["RAW_CAPTURE_DIR"] = "off"
os.environ.pop("TARGET_PATIENT_ID", None)

from edge_inference.runner import EdgeInferenceRunner  # noqa: E402


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("captures", nargs="+")
    ap.add_argument("--repeat", type=int, default=3, help="passes over all captures")
    a = ap.parse_args()

    tmp = tempfile.NamedTemporaryFile(suffix=".db", dir=ML_DIR / "storage", delete=False)
    tmp.close()
    try:
        runner = EdgeInferenceRunner(db_path=tmp.name)
        runner.broadcast_telemetry = lambda payload: None
        W, step = runner.window_size, runner.step_size

        total, dsp, inf, outcomes = [], [], [], {}
        for _ in range(a.repeat):
            for path in a.captures:
                z = np.load(path)
                ir, red, ts = z["ir"], z["red"], z["timestamp_ms"]
                for s in range(0, len(ir) - W + 1, step):
                    runner.raw_ir_buffer = [int(v) for v in ir[s:s + W]]
                    runner.raw_red_buffer = [int(v) for v in red[s:s + W]]
                    runner.timestamps_buffer = [int(v) for v in ts[s:s + W]]
                    runner.sample_count += step
                    t0 = time.perf_counter_ns()
                    res = runner.process_window()
                    wall = (time.perf_counter_ns() - t0) / 1e6
                    key = (res.get("invalid_reason") if res.get("measurement_valid") is False
                           else ("AF" if res.get("af_detected") else "no AF"))
                    outcomes[key] = outcomes.get(key, 0) + 1
                    total.append(wall)
                    lat = res.get("latencies")
                    if lat:  # only classified windows run the full chain
                        dsp.append(lat["dsp_ms"])
                        inf.append(lat["inference_ms"])

        # Ledger write cost on its own (hash-chained insert on the SD card).
        ledger = []
        for _ in range(50):
            t0 = time.perf_counter_ns()
            runner.db.record_event(patient_id=None, device_id="BENCH", bpm=70.0,
                                   af_detected=1, confidence=0.9, gradcam_path="{}")
            ledger.append((time.perf_counter_ns() - t0) / 1e6)

        print(f"host: {platform.node()} | {platform.machine()} | Python {platform.python_version()} "
              f"| numpy {np.__version__}")
        print(f"captures: {len(a.captures)} x {a.repeat} passes -> {len(total)} windows; outcomes {outcomes}")
        print(f"{'stage':34s} {'n':>5s} {'median':>8s} {'p95':>8s} {'p99':>8s} {'max':>8s}  (ms)")
        for name, arr in (("end-to-end, every window", total),
                          ("DSP (classified windows)", dsp),
                          ("inference + explanation", inf),
                          ("ledger write (hash-chained)", ledger)):
            print(f"{name:34s} {len(arr):5d} {pct(arr, 50):8.2f} {pct(arr, 95):8.2f} "
                  f"{pct(arr, 99):8.2f} {max(arr) if arr else float('nan'):8.2f}")
        over = sum(t >= 25.0 for t in total)
        print(f"windows at or over the 25 ms budget: {over} of {len(total)}")
    finally:
        import gc
        runner = None  # drop open SQLite handles before deleting (required on Windows)
        gc.collect()
        for suffix in ("", "-wal", "-shm"):
            try:
                os.remove(tmp.name + suffix)
            except FileNotFoundError:
                pass
            except PermissionError:
                print(f"note: could not delete {tmp.name + suffix}; remove it by hand")


if __name__ == "__main__":
    main()
