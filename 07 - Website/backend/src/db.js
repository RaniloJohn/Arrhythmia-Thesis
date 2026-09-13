/**
 * SQLite Database Connection & Query Interface
 * Thesis: An Internet of Things-Based Framework for Cardiac Arrhythmia Detection
 *         via Photoplethysmography and Machine Learning (3CPE-2A, Univ. of the East)
 * 
 * Contract:
 * - WAL mode enabled per ADR-001 for non-blocking concurrent Python/Node.js access.
 * - Node.js owns write access to `patients`.
 * - Python owns write access to `arrhythmia_events` (tamper-evident hash chain).
 * - Node.js provides read access & cryptographic verification of the hash chain.
 */

const { DatabaseSync } = require('node:sqlite');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const bcrypt = require('bcryptjs');
const config = require('./config');

const GENESIS_HASH = '0'.repeat(64);

class DatabaseService {
  constructor(dbPath = config.DB_PATH) {
    this.dbPath = dbPath;
    this._init();
  }

  _init() {
    const dir = path.dirname(this.dbPath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }

    this.db = new DatabaseSync(this.dbPath);

    // Apply SQLite performance and concurrency pragmas per ADR-001
    this.db.exec('PRAGMA journal_mode = WAL;');
    this.db.exec('PRAGMA synchronous = NORMAL;');
    this.db.exec('PRAGMA foreign_keys = ON;');

    this._ensureSchema();
  }

  _ensureSchema() {
    // 1. Patients Table (Owned and Written by Website Backend)
    this.db.exec(`
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
    `);

    // 2. Arrhythmia Events Table (Owned & Written by Edge Python Process)
    this.db.exec(`
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
    `);

    this.db.exec(`
      CREATE INDEX IF NOT EXISTS idx_events_patient ON arrhythmia_events(patient_id);
      CREATE INDEX IF NOT EXISTS idx_events_timestamp ON arrhythmia_events(timestamp);
      CREATE INDEX IF NOT EXISTS idx_events_sync ON arrhythmia_events(sync_status);
    `);

    // 3. Clinician Users Table
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL CHECK (role IN ('clinician', 'admin')),
        full_name TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
      );
    `);

    // First-run admin bootstrap (PLAN §2)
    // If users table is empty, read ADMIN_USERNAME & ADMIN_PASSWORD from environment,
    // hash with bcrypt cost 12 at runtime, and create exactly one admin account.
    // Never commit a password hash or auto-seed fake accounts.
    try {
      const userCount = this.db.prepare('SELECT COUNT(*) AS count FROM users').get().count;
      if (userCount === 0) {
        const adminUser = process.env.ADMIN_USERNAME;
        const adminPass = process.env.ADMIN_PASSWORD;
        if (adminUser && adminPass) {
          const salt = bcrypt.genSaltSync(12);
          const hash = bcrypt.hashSync(adminPass, salt);
          const insertUser = this.db.prepare(`
            INSERT INTO users (user_id, username, password_hash, role, full_name)
            VALUES (?, ?, ?, ?, ?)
          `);
          insertUser.run(
            'usr-admin-01',
            adminUser,
            hash,
            'admin',
            'System Administrator'
          );
          console.log(`[DB] Bootstrapped initial admin user '${adminUser}' from environment.`);
        } else {
          console.warn('[DB] Notice: users table is empty and ADMIN_USERNAME / ADMIN_PASSWORD are not set in environment. No admin account created. Supply ADMIN_USERNAME and ADMIN_PASSWORD to bootstrap.');
        }
      }
    } catch (bootstrapErr) {
      console.warn('[DB] User bootstrap notice:', bootstrapErr.message);
    }
  }

  // User Queries
  getUserByUsername(username) {
    const stmt = this.db.prepare('SELECT * FROM users WHERE username = ?');
    return stmt.get(username);
  }

  getUserById(userId) {
    const stmt = this.db.prepare('SELECT user_id, username, role, full_name, created_at FROM users WHERE user_id = ?');
    return stmt.get(userId);
  }

  createUser(userId, username, passwordHash, role, fullName) {
    const stmt = this.db.prepare(`
      INSERT INTO users (user_id, username, password_hash, role, full_name)
      VALUES (?, ?, ?, ?, ?)
    `);
    stmt.run(userId, username, passwordHash, role, fullName);
    return this.getUserByUsername(username);
  }

  clearEventsLedger() {
    this.db.exec('DELETE FROM arrhythmia_events;');
  }

  // Patient Queries
  getAllPatients(search = '') {
    let sql = `
      SELECT p.*, 
             COUNT(e.event_id) AS total_events,
             SUM(CASE WHEN e.af_detected = 1 THEN 1 ELSE 0 END) AS af_episodes,
             MAX(e.timestamp) AS last_event_time
      FROM patients p
      LEFT JOIN arrhythmia_events e ON p.patient_id = e.patient_id
    `;
    const params = [];
    if (search && search.trim() !== '') {
      sql += ` WHERE p.name LIKE ? OR p.patient_id LIKE ? OR p.device_id LIKE ? OR p.barangay LIKE ?`;
      const term = `%${search.trim()}%`;
      params.push(term, term, term, term);
    }
    sql += ` GROUP BY p.patient_id ORDER BY p.updated_at DESC`;

    const stmt = this.db.prepare(sql);
    return stmt.all(...params);
  }

  getPatientById(patientId) {
    const stmt = this.db.prepare(`
      SELECT p.*,
             COUNT(e.event_id) AS total_events,
             SUM(CASE WHEN e.af_detected = 1 THEN 1 ELSE 0 END) AS af_episodes,
             MAX(e.timestamp) AS last_event_time
      FROM patients p
      LEFT JOIN arrhythmia_events e ON p.patient_id = e.patient_id
      WHERE p.patient_id = ?
      GROUP BY p.patient_id
    `);
    return stmt.get(patientId);
  }

  createPatient({ patientId, name, age, gender, contactNumber, barangay, deviceId, medicalHistory, notes }) {
    const id = patientId || `PAT-CAL-${Math.floor(100 + Math.random() * 900)}`;
    const now = new Date().toISOString();
    const stmt = this.db.prepare(`
      INSERT INTO patients (
        patient_id, name, age, gender, contact_number, barangay, device_id, medical_history, notes, created_at, updated_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);
    stmt.run(
      id,
      name,
      Number(age),
      gender,
      contactNumber || null,
      barangay,
      deviceId,
      medicalHistory || null,
      notes || null,
      now,
      now
    );
    return this.getPatientById(id);
  }

  updatePatient(patientId, { name, age, gender, contactNumber, barangay, deviceId, medicalHistory, notes }) {
    const now = new Date().toISOString();
    const stmt = this.db.prepare(`
      UPDATE patients
      SET name = ?, age = ?, gender = ?, contact_number = ?, barangay = ?, device_id = ?,
          medical_history = ?, notes = ?, updated_at = ?
      WHERE patient_id = ?
    `);
    stmt.run(
      name,
      Number(age),
      gender,
      contactNumber || null,
      barangay,
      deviceId,
      medicalHistory || null,
      notes || null,
      now,
      patientId
    );
    return this.getPatientById(patientId);
  }

  deletePatient(patientId) {
    const stmt = this.db.prepare('DELETE FROM patients WHERE patient_id = ?');
    return stmt.run(patientId);
  }

  // Arrhythmia Event Queries (Read-Only ledger from Node.js)
  getEvents({ patientId, limit = 50, offset = 0, afOnly = false, syncStatus = null }) {
    let sql = `
      SELECT e.*, p.name AS patient_name, p.barangay
      FROM arrhythmia_events e
      LEFT JOIN patients p ON e.patient_id = p.patient_id
      WHERE 1=1
    `;
    const params = [];

    if (patientId) {
      sql += ` AND e.patient_id = ?`;
      params.push(patientId);
    }
    if (afOnly) {
      sql += ` AND e.af_detected = 1`;
    }
    if (syncStatus !== null && syncStatus !== undefined && syncStatus !== '') {
      sql += ` AND e.sync_status = ?`;
      params.push(Number(syncStatus));
    }

    sql += ` ORDER BY e.timestamp DESC LIMIT ? OFFSET ?`;
    params.push(Number(limit), Number(offset));

    const stmt = this.db.prepare(sql);
    return stmt.all(...params);
  }

  getEventById(eventId) {
    const stmt = this.db.prepare(`
      SELECT e.*, p.name AS patient_name, p.age AS patient_age, p.gender AS patient_gender, p.barangay
      FROM arrhythmia_events e
      LEFT JOIN patients p ON e.patient_id = p.patient_id
      WHERE e.event_id = ?
    `);
    return stmt.get(eventId);
  }

  // Cryptographic Hash Chain Verification (ADR-001 & PLAN §4)
  computeRecordHash(eventId, patientId, timestamp, deviceId, bpm, afDetected, confidence, prevHash) {
    const payload = [
      eventId,
      patientId || '',
      timestamp,
      deviceId,
      Number(bpm).toFixed(2),
      afDetected,
      Number(confidence).toFixed(4),
      prevHash
    ].join('|');
    return crypto.createHash('sha256').update(payload, 'utf8').digest('hex');
  }

  validateLedgerChain() {
    const stmt = this.db.prepare(`
      SELECT rowid, event_id, patient_id, timestamp, device_id, bpm, af_detected, confidence, prev_hash, record_hash
      FROM arrhythmia_events
      ORDER BY rowid ASC
    `);
    const rows = stmt.all();

    if (!rows || rows.length === 0) {
      return {
        valid: true,
        count: 0,
        message: 'Chain is empty (Genesis state valid).'
      };
    }

    let expectedPrevHash = GENESIS_HASH;
    for (let i = 0; i < rows.length; i++) {
      const row = rows[i];

      // 1. Check backward link
      if (row.prev_hash !== expectedPrevHash) {
        return {
          valid: false,
          count: rows.length,
          brokenIndex: i,
          eventId: row.event_id,
          message: `Broken link at rowid ${row.rowid} (event ${row.event_id}): prev_hash ${row.prev_hash} != expected ${expectedPrevHash}`
        };
      }

      // 2. Recompute record hash
      const calculated = this.computeRecordHash(
        row.event_id,
        row.patient_id,
        row.timestamp,
        row.device_id,
        row.bpm,
        row.af_detected,
        row.confidence,
        row.prev_hash
      );

      if (row.record_hash !== calculated) {
        return {
          valid: false,
          count: rows.length,
          brokenIndex: i,
          eventId: row.event_id,
          message: `Tampered record detected at rowid ${row.rowid} (event ${row.event_id}): stored ${row.record_hash} != calculated ${calculated}`
        };
      }

      expectedPrevHash = row.record_hash;
    }

    return {
      valid: true,
      count: rows.length,
      latestHash: expectedPrevHash,
      message: `Cryptographic hash chain intact: ${rows.length} sequential diagnostic event(s) mathematically verified.`
    };
  }
}

module.exports = new DatabaseService();
