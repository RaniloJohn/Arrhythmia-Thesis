#!/usr/bin/env python3
"""
Purge Synthetic Demo Data & Re-verify Cryptographic Hash Chain Ledger
Thesis: An Internet of Things-Based Framework for Cardiac Arrhythmia Detection
        via Photoplethysmography and Machine Learning (3CPE-2A, University of the East)

This maintenance utility:
 1. Creates a timestamped safety backup of arrhythmia_edge.db.
 2. Displays existing records (synthetic events and seeded patients).
 3. Purges all synthetic events from `arrhythmia_events`.
 4. Purges the 3 seeded mock patients (`PAT-CAL-001`, `PAT-CAL-002`, `PAT-CAL-003`).
 5. Validates the genesis cryptographic SHA-256 hash chain (ensuring valid=True, count=0).
"""

import os
import sys
import shutil
import sqlite3
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from storage.db_manager import DatabaseManager

def main():
    db_path = os.path.join(BASE_DIR, "storage", "arrhythmia_edge.db")
    if not os.path.exists(db_path):
        print(f"[ERROR] Database file not found at: {db_path}")
        sys.exit(1)

    print("=" * 66)
    print(" ARRHYTHMIA THESIS — SYNTHETIC DATA PURGE & LEDGER RESET UTILITY")
    print(f" Database: {db_path}")
    print("=" * 66)

    # 1. Create safety backup
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{db_path}.bak_{timestamp_str}"
    shutil.copy2(db_path, backup_path)
    print(f"\n[1/5] Created safety database backup at:\n      -> {backup_path}")

    # 2. Inspect current database state
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    events = cursor.execute("SELECT event_id, patient_id, timestamp, af_detected, record_hash FROM arrhythmia_events;").fetchall()
    patients = cursor.execute("SELECT patient_id, name, age, gender FROM patients;").fetchall()
    users = cursor.execute("SELECT user_id, username, role FROM users;").fetchall()

    print(f"\n[2/5] Current Database State:")
    print(f"      • Events: {len(events)} record(s)")
    for ev in events:
        print(f"        - ID: {ev['event_id'][:8]}... | Patient: {ev['patient_id']} | AF: {ev['af_detected']} | Date: {ev['timestamp']}")

    print(f"      • Patients: {len(patients)} record(s)")
    for pt in patients:
        print(f"        - ID: {pt['patient_id']} | Name: {pt['name']} | Age: {pt['age']} | Gender: {pt['gender']}")

    print(f"      • Users: {len(users)} record(s)")
    for u in users:
        print(f"        - User: {u['username']} ({u['role']})")

    # 3. Purge synthetic events
    print("\n[3/5] Purging synthetic records from arrhythmia_events...")
    cursor.execute("DELETE FROM arrhythmia_events;")
    conn.commit()
    print("      -> Successfully purged all synthetic events.")

    # 4. Purge seeded mock patients
    print("\n[4/5] Purging seeded mock patients (PAT-CAL-001..003)...")
    cursor.execute("DELETE FROM patients WHERE patient_id IN ('PAT-CAL-001', 'PAT-CAL-002', 'PAT-CAL-003');")
    conn.commit()
    print("      -> Seeded demo patient records removed.")

    remaining_patients = cursor.execute("SELECT count(*) FROM patients;").fetchone()[0]
    print(f"      -> Remaining registered patients: {remaining_patients}")
    conn.close()

    # 5. Cryptographic hash chain validation
    print("\n[5/5] Re-verifying cryptographic hash chain continuity...")
    db_mgr = DatabaseManager(db_path=db_path)
    valid, msg, count = db_mgr.validate_chain()
    print(f"      -> Chain Valid: {valid}")
    print(f"      -> Event Count: {count}")
    print(f"      -> Details:     {msg}")

    if valid and count == 0:
        print("\n" + "=" * 66)
        print(" [SUCCESS] Database is pristine, unseeded, and genesis-ready!")
        print(" Next events recorded from genuine PPG sensing will chain from genesis.")
        print("=" * 66)
    else:
        print("\n[WARNING] Unexpected ledger validation state:", valid, msg, count)

if __name__ == "__main__":
    main()
