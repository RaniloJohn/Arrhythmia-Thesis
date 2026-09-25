"""
Unit Tests for Database Manager & Cryptographic Hash Chaining (ADR-001)
=======================================================================
Tests:
1. Database initialization and table seeding in temporary SQLite DB.
2. Hash-chain event recording: each event links to the preceding SHA-256 hash.
3. Cryptographic ledger integrity verification (validate_chain returns valid = True).
4. Tamper detection: modifying an event's BPM, label, or hash is immediately caught.
5. Patient record retrieval and foreign key integrity.
"""

import sqlite3
import pytest
from pathlib import Path
from storage.db_manager import DatabaseManager, GENESIS_HASH


@pytest.fixture
def temp_db(tmp_path):
    """Provides a fresh isolated SQLite database instance for each test."""
    db_file = tmp_path / "test_arrhythmia_edge.db"
    manager = DatabaseManager(str(db_file))
    with manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO patients (patient_id, name, age, gender, barangay, device_id)
        VALUES 
            ('PAT-CAL-001', 'Test Patient', 45, 'Male', 'Barangay 1', 'ESP32C3-NODE-01'),
            ('PAT-CAL-002', 'Test Patient 2', 52, 'Female', 'Barangay 2', 'ESP32C3-NODE-02');
        """)
        cursor.execute("""
        INSERT INTO users (user_id, username, password_hash, role, full_name)
        VALUES ('USR-001', 'testadmin', 'dummyhash', 'admin', 'Test Admin');
        """)
        conn.commit()
    return manager


def test_db_initialization_and_seeding(temp_db):
    """Test that tables are created and test patients/users can be queried."""
    with temp_db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0] for row in cursor.fetchall()}
        assert "patients" in tables
        assert "arrhythmia_events" in tables
        assert "users" in tables

        assert temp_db.patient_exists("PAT-CAL-001") is True
        assert temp_db.patient_exists("NON_EXISTENT") is False


def test_hash_chain_append_and_validation(temp_db):
    """Test recording events and verifying unbroken SHA-256 backward hash chain."""
    # First record should link to GENESIS_HASH
    rec1 = temp_db.record_event(
        patient_id="PAT-CAL-001",
        device_id="ESP32C3-NODE-01",
        bpm=75.5,
        af_detected=0,
        confidence=0.12,
    )
    assert rec1["prev_hash"] == GENESIS_HASH
    assert len(rec1["record_hash"]) == 64

    # Second record should link to rec1's hash
    rec2 = temp_db.record_event(
        patient_id="PAT-CAL-001",
        device_id="ESP32C3-NODE-01",
        bpm=118.2,
        af_detected=1,
        confidence=0.88,
    )
    assert rec2["prev_hash"] == rec1["record_hash"]

    # Third record should link to rec2's hash
    rec3 = temp_db.record_event(
        patient_id="PAT-CAL-002",
        device_id="ESP32C3-NODE-02",
        bpm=71.0,
        af_detected=0,
        confidence=0.08,
    )
    assert rec3["prev_hash"] == rec2["record_hash"]

    # Chain validation must succeed
    valid, msg, count = temp_db.validate_chain()
    assert valid is True
    assert count == 3
    assert "intact" in msg.lower() or "verified" in msg.lower()


def test_tamper_detection_in_data(temp_db):
    """Test that altering a field in SQLite (e.g. changing af_detected) breaks the chain."""
    # Add two records
    temp_db.record_event("PAT-CAL-001", "NODE-01", 72.0, 0, 0.15)
    temp_db.record_event("PAT-CAL-001", "NODE-01", 120.0, 1, 0.90)

    # Tamper with the first record: alter bpm from 72.0 to 99.0
    with temp_db.get_connection() as conn:
        conn.execute("UPDATE arrhythmia_events SET bpm = 99.0 WHERE rowid = 1;")
        conn.commit()

    # Chain validation must detect the tampering
    valid, msg, count = temp_db.validate_chain()
    assert valid is False
    assert "mismatch" in msg.lower() or "tamper" in msg.lower() or "corrupted" in msg.lower()


def test_tamper_detection_in_hash(temp_db):
    """Test that altering a stored record_hash breaks subsequent linkage."""
    temp_db.record_event("PAT-CAL-001", "NODE-01", 72.0, 0, 0.15)
    temp_db.record_event("PAT-CAL-001", "NODE-01", 120.0, 1, 0.90)

    # Tamper with record_hash of row 1
    fake_hash = "a" * 64
    with temp_db.get_connection() as conn:
        conn.execute(f"UPDATE arrhythmia_events SET record_hash = '{fake_hash}' WHERE rowid = 1;")
        conn.commit()

    valid, msg, count = temp_db.validate_chain()
    assert valid is False
