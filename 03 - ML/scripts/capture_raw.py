"""
Raw MAX30102 capture for offline signal-quality analysis.

Pauses the edge runner (SIGSTOP), reads framed samples straight from the serial port for
a fixed duration, saves them to an .npz, then resumes the runner (SIGCONT) — the service is
never restarted. The runner must be owned by the invoking user.

    python3 scripts/capture_raw.py --seconds 60 --out /tmp/capture.npz [--label nsr]
"""
import argparse, os, signal, subprocess, sys, time
import numpy as np
import serial

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from edge_inference.serial_protocol import StreamPacketParser


def runner_pids():
    # Anchor on the interpreter so a shell whose command line merely mentions runner.py
    # (e.g. an ssh one-liner) is never paused by mistake.
    out = subprocess.run(["pgrep", "-f", r"^\S*python3? edge_inference/runner\.py"],
                         capture_output=True, text=True)
    return [int(p) for p in out.stdout.split()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="unlabelled", help="e.g. nsr / af / unknown")
    a = ap.parse_args()

    pids = runner_pids()
    for p in pids:
        os.kill(p, signal.SIGSTOP)
    ts, ir, red = [], [], []
    try:
        parser = StreamPacketParser()
        with serial.Serial(a.port, 115200, timeout=0.5) as ser:
            ser.reset_input_buffer()
            t_end = time.time() + a.seconds
            next_note = time.time() + 5
            while time.time() < t_end:
                parser.feed(ser.read(max(ser.in_waiting, 1)))
                while True:
                    f = parser.parse_next()
                    if f is None:
                        break
                    ts.append(f["timestamp_ms"]); ir.append(f["ir_raw"]); red.append(f["red_raw"])
                if time.time() >= next_note:
                    print(f"  {len(ir)} frames, last IR {ir[-1] if ir else '-'}", flush=True)
                    next_note += 5
    finally:
        for p in pids:
            os.kill(p, signal.SIGCONT)
    np.savez(a.out, timestamp_ms=np.array(ts, np.int64), ir=np.array(ir, np.int64),
             red=np.array(red, np.int64), label=a.label)
    print(f"saved {len(ir)} frames to {a.out}; runner resumed ({pids})")


if __name__ == "__main__":
    main()
