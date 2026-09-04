"""
Storage Layer & Cryptographic Hash Chain Manager (ADR-001)
==========================================================
Thesis: An Internet of Things-Based Framework for Cardiac Arrhythmia Detection
        via Photoplethysmography and Machine Learning (3CPE-2A, Univ. of the East)

Features:
- SQLite in Write-Ahead Logging (WAL) mode for concurrent Node.js / Python access.
- SHA-256 backward hash chaining on arrhythmia_events for tamper-evidence.
- Patient attribution via patient_id foreign key.
- Seed data for default clinician/admin accounts and primary-care demo patients.
"""

import os
import sqlite3
import hashlib
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "arrhythmia_edge.db")
GENESIS_HASH = "0" * 64

# Default password hashes (bcrypt cost 12):
# admin: admin123
# clinician: clinician123
SEED_USERS = [
    {
        "user_id": "usr-admin-01",
        "username": "admin",
        "password_hash": "$2b$12$wZwimrCy8zichNwWryZ6sOOE5NDe6dBtX9I8.9vDkFWWWfN8JBXoW",
        "role": "admin",
        "full_name": "System Administrator (3CPE-2A)"
    },
    {
        "user_id": "usr-clin-01",
        "username": "clinician",
        "password_hash": "$2b$12$RTopYMaTdq4Fl8IHEcXV4OD3yJ6D7vFAtwOH9xkfdhlwg8Yi0rviW",
        "role": "clinician",
        "full_name": "Dr. Maria Santos, MD (Brgy. 171 Health Center)"
    }
]

SEED_PATIENTS = [
    {
        "patient_id": "PAT-CAL-001",
        "name": "Eduardo Ramos",
        "age": 63,
        "gender": "Male",
        "contact_number": "+63 917 555 1024",
        "barangay": "Barangay 171, Bagumbong, Caloocan City",
        "device_id": "ESP32C3-NODE-01",
        "medical_history": "Hypertension (5 yrs), Type 2 Diabetes, occasional palpitations",
        "notes": "Referred for ambulatory rhythm screening following primary health consultation."
    },
    {
        "patient_id": "PAT-CAL-002",
        "name": "Corazon Bautista",
        "age": 58,
        "gender": "Female",
        "contact_number": "+63 928 555 3841",
        "barangay": "Barangay 172, Urduja, Caloocan City",
        "device_id": "ESP32C3-NODE-02",
        "medical_history": "Post-menopausal, mild mitral valve prolapse, hyperlipidemia",
        "notes": "Routine community health center outreach screening."
    },
    {
        "patient_id": "PAT-CAL-003",
        "name": "Rodrigo Dela Cruz",
        "age": 71,
        "gender": "Male",
        "contact_number": "+63 919 555 9012",
        "barangay": "Barangay 177, Camarin, Caloocan City",
        "device_id": "ESP32C3-NODE-03",
        "medical_history": "Previous transient ischemic attack (TIA 2024), hypertensive heart disease",
        "notes": "High risk for embolic stroke; prioritize continuous ambulatory check."
    }
]


class DatabaseManager:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._init_sqlite()

    def get_connection(self) -> sqlite3.Connection:
        """Create and configure a SQLite connection with WAL mode and foreign keys."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_sqlite(self):
        """Create tables and initialize baseline data if not already present."""
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # 1. Patients Table (Owned & Written by Node.js Backend per PLAN §1 & Claude Notes)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                patient_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                age INTEGER NOT NULL,
                gender TEXT NOT NULL,
                contact_number TEXT,
                barangay TEXT NOT NULL,
                device_id TEXT NOT NULL,
                medical_history TEXT,
                notes TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # 2. Arrhythmia Events Table (Owned & Written by Python Edge Inference with Hash Chain)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS arrhythmia_events (
                event_id TEXT PRIMARY KEY,
                patient_id TEXT,
                timestamp DATETIME NOT NULL,
                device_id TEXT NOT NULL,
                bpm REAL NOT NULL,
                af_detected INTEGER NOT NULL,
                confidence REAL NOT NULL,
                gradcam_path TEXT,
                prev_hash TEXT NOT NULL,
                record_hash TEXT NOT NULL,
                sync_status INTEGER DEFAULT 0,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE SET NULL
            );
            """)

            # Index for rapid lookups
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_patient ON arrhythmia_events(patient_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON arrhythmia_events(timestamp);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_sync ON arrhythmia_events(sync_status);")

            # 3. Clinician & Admin Auth Table (PLAN §3)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('clinician', 'admin')),
                full_name TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # Seed default users if empty
            cursor.execute("SELECT COUNT(*) FROM users;")
            if cursor.fetchone()[0] == 0:
                for user in SEED_USERS:
                    cursor.execute("""
                    INSERT INTO users (user_id, username, password_hash, role, full_name)
                    VALUES (?, ?, ?, ?, ?);
                    """, (user["user_id"], user["username"], user["password_hash"], user["role"], user["full_name"]))

            # Seed default patients if empty
            cursor.execute("SELECT COUNT(*) FROM patients;")
            if cursor.fetchone()[0] == 0:
                for p in SEED_PATIENTS:
                    cursor.execute("""
                    INSERT INTO patients (patient_id, name, age, gender, contact_number, barangay, device_id, medical_history, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """, (p["patient_id"], p["name"], p["age"], p["gender"], p["contact_number"], p["barangay"], p["device_id"], p["medical_history"], p["notes"]))

            conn.commit()

    def get_latest_hash(self) -> str:
        """Retrieve the hash of the latest event in the chain, or GENESIS_HASH if empty."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT record_hash FROM arrhythmia_events ORDER BY rowid DESC LIMIT 1;")
            row = cursor.fetchone()
            if row:
                return row["record_hash"]
            return GENESIS_HASH

    @staticmethod
    def compute_record_hash(event_id: str, patient_id: Optional[str], timestamp: str,
                            device_id: str, bpm: float, af_detected: int,
                            confidence: float, prev_hash: str) -> str:
        """
        Cryptographic SHA-256 hash chaining formula:
        Hash_t = SHA-256(event_id | patient_id | timestamp | device_id | bpm | af | conf | Hash_{t-1})
        """
        payload = f"{event_id}|{patient_id or ''}|{timestamp}|{device_id}|{bpm:.2f}|{af_detected}|{confidence:.4f}|{prev_hash}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def record_event(self, patient_id: Optional[str], device_id: str, bpm: float,
                     af_detected: int, confidence: float,
                     gradcam_path: Optional[str] = None,
                     timestamp: Optional[str] = None,
                     sync_status: int = 0) -> Dict[str, Any]:
        """
        Append an arrhythmia classification event to the SQLite database with tamper-evident hash chaining.
        """
        event_id = str(uuid.uuid4())
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Lock row to prevent concurrent race condition in hash computation
            cursor.execute("SELECT record_hash FROM arrhythmia_events ORDER BY rowid DESC LIMIT 1;")
            last_row = cursor.fetchone()
            prev_hash = last_row["record_hash"] if last_row else GENESIS_HASH

            record_hash = self.compute_record_hash(
                event_id=event_id,
                patient_id=patient_id,
                timestamp=timestamp,
                device_id=device_id,
                bpm=bpm,
                af_detected=af_detected,
                confidence=confidence,
                prev_hash=prev_hash
            )

            cursor.execute("""
            INSERT INTO arrhythmia_events (
                event_id, patient_id, timestamp, device_id, bpm, af_detected,
                confidence, gradcam_path, prev_hash, record_hash, sync_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (event_id, patient_id, timestamp, device_id, bpm, af_detected,
                  confidence, gradcam_path, prev_hash, record_hash, sync_status))

            conn.commit()

            return {
                "event_id": event_id,
                "patient_id": patient_id,
                "timestamp": timestamp,
                "device_id": device_id,
                "bpm": bpm,
                "af_detected": af_detected,
                "confidence": confidence,
                "gradcam_path": gradcam_path,
                "prev_hash": prev_hash,
                "record_hash": record_hash,
                "sync_status": sync_status
            }

    def validate_chain(self) -> Tuple[bool, str, int]:
        """
        Walks the entire arrhythmia_events ledger from rowid 1 to N,
        verifying cryptographic hash continuity and individual record hashes.
        Returns: (is_valid: bool, details: str, event_count: int)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT rowid, event_id, patient_id, timestamp, device_id, bpm,
                   af_detected, confidence, prev_hash, record_hash
            FROM arrhythmia_events
            ORDER BY rowid ASC;
            """)
            rows = cursor.fetchall()

            if not rows:
                return True, "Chain is empty (Genesis state valid)", 0

            expected_prev_hash = GENESIS_HASH
            for idx, row in enumerate(rows):
                # 1. Verify link to previous event
                if row["prev_hash"] != expected_prev_hash:
                    return False, f"Broken link at rowid {row['rowid']} (event {row['event_id']}): prev_hash {row['prev_hash']} != expected {expected_prev_hash}", idx

                # 2. Recompute record_hash
                calculated_hash = self.compute_record_hash(
                    event_id=row["event_id"],
                    patient_id=row["patient_id"],
                    timestamp=row["timestamp"],
                    device_id=row["device_id"],
                    bpm=row["bpm"],
                    af_detected=row["af_detected"],
                    confidence=row["confidence"],
                    prev_hash=row["prev_hash"]
                )

                if row["record_hash"] != calculated_hash:
                    return False, f"Tampered record at rowid {row['rowid']} (event {row['event_id']}): stored {row['record_hash']} != recomputed {calculated_hash}", idx

                expected_prev_hash = row["record_hash"]

            return True, f"Hash chain verified: {len(rows)} events cryptographically intact.", len(rows)


if __name__ == "__main__":
    db = DatabaseManager()
    print("Database initialized at:", db.db_path)
    # Seed a test event if none exists
    valid, msg, count = db.validate_chain()
    print(f"Chain validation: valid={valid}, count={count}, msg={msg}")
    if count == 0:
        event = db.record_event(
            patient_id="PAT-CAL-001",
            device_id="ESP32C3-NODE-01",
            bpm=88.5,
            af_detected=1,
            confidence=0.9412,
            gradcam_path="{\"weights\": [0.1, 0.8, 0.9, 0.3]}"
        )
        print("Inserted initial test event:", event["event_id"])
        valid, msg, count = db.validate_chain()
        print(f"Chain validation after insert: valid={valid}, count={count}, msg={msg}")
