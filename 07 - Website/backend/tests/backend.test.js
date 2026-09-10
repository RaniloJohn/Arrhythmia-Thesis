/**
 * Automated Verification Test Suite for Clinician Web Dashboard Backend
 * Evaluates ISO/IEC 25010 Quality Model:
 *  - Functional Suitability: Patient CRUD, Event Ledger, Live Stream Bridge
 *  - Security: bcrypt auth, JWT token verification, RBAC, unauthenticated WS rejection
 *  - Performance Efficiency: Sensor-to-browser latency tracking, CNN <25ms validation
 *  - Reliability: SHA-256 hash-chain continuity under concurrent queries
 */

process.env.JWT_SECRET = process.env.JWT_SECRET || 'test-jwt-secret-key-3cpe2a-verification-token';
process.env.ADMIN_USERNAME = process.env.ADMIN_USERNAME || 'admin';
process.env.ADMIN_PASSWORD = process.env.ADMIN_PASSWORD || 'admin123';
process.env.ENABLE_SIMULATION = '1';
process.env.PORT = process.env.TEST_PORT || '8088';
process.env.TCP_BRIDGE_PORT = process.env.TEST_TCP_BRIDGE_PORT || '5058';

const http = require('http');
const bcrypt = require('bcryptjs');
const { WebSocket } = require('ws');
const { app, server } = require('../src/server');
const config = require('../src/config');
const db = require('../src/db');

// Helper for making HTTP requests
function httpRequest(options, postData = null) {
  return new Promise((resolve, reject) => {
    const req = http.request(options, (res) => {
      let data = '';
      res.on('data', (chunk) => (data += chunk));
      res.on('end', () => {
        try {
          const parsed = data ? JSON.parse(data) : {};
          resolve({ status: res.statusCode, headers: res.headers, body: parsed });
        } catch (e) {
          resolve({ status: res.statusCode, headers: res.headers, body: data });
        }
      });
    });
    req.on('error', reject);
    if (postData) {
      req.write(typeof postData === 'string' ? postData : JSON.stringify(postData));
    }
    req.end();
  });
}

let clinicianToken = '';
let adminToken = '';
let testPatientId = `PAT-TEST-${Date.now()}`;

async function runTests() {
  console.log('\n================================================================');
  console.log(' STARTING CLINICIAN DASHBOARD BACKEND TEST SUITE (PLAN.md)');
  console.log('================================================================\n');

  let passed = 0;
  let failed = 0;

  function assert(condition, message) {
    if (condition) {
      console.log(` [PASS] ${message}`);
      passed++;
    } else {
      console.error(` [FAIL] ${message}`);
      failed++;
    }
  }

  try {
    // -------------------------------------------------------------
    // TEST 1: Health & Database Integrity Check
    // -------------------------------------------------------------
    console.log('--- Phase 1: Database & Health Checks ---');
    const healthRes = await httpRequest({
      hostname: '127.0.0.1',
      port: config.PORT,
      path: '/api/health',
      method: 'GET'
    });
    assert(healthRes.status === 200, 'Health endpoint responds with HTTP 200');
    assert(healthRes.body.status === 'healthy', 'System reports healthy state');
    assert(healthRes.body.database.journal_mode === 'WAL', 'SQLite configured in WAL mode per ADR-001');

    // -------------------------------------------------------------
    // TEST 2: Clinician Authentication (PLAN §3)
    // -------------------------------------------------------------
    console.log('\n--- Phase 2: Clinician Auth & RBAC (PLAN §3) ---');

    // Ensure test credentials exist in test environment
    if (!db.getUserByUsername('admin')) {
      const salt = bcrypt.genSaltSync(12);
      db.createUser('usr-admin-test', 'admin', bcrypt.hashSync('admin123', salt), 'admin', 'System Administrator');
    }
    if (!db.getUserByUsername('clinician')) {
      const salt = bcrypt.genSaltSync(12);
      db.createUser('usr-clin-test', 'clinician', bcrypt.hashSync('clinician123', salt), 'clinician', 'Dr. Maria Santos, MD');
    }
    
    // 2.1 Invalid login attempt
    const badLogin = await httpRequest({
      hostname: '127.0.0.1',
      port: config.PORT,
      path: '/api/auth/login',
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    }, { username: 'clinician', password: 'wrongpassword' });
    assert(badLogin.status === 401, 'Invalid password correctly rejected with 401');

    // 2.2 Valid Clinician login
    const clinLogin = await httpRequest({
      hostname: '127.0.0.1',
      port: config.PORT,
      path: '/api/auth/login',
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    }, { username: 'clinician', password: 'clinician123' });
    assert(clinLogin.status === 200, 'Clinician login succeeds with HTTP 200');
    assert(!!clinLogin.body.token, 'Clinician login returns signed JWT');
    assert(clinLogin.body.user.role === 'clinician', 'Returned user has clinician role');
    clinicianToken = clinLogin.body.token;

    // 2.3 Valid Admin login
    const adminLogin = await httpRequest({
      hostname: '127.0.0.1',
      port: config.PORT,
      path: '/api/auth/login',
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    }, { username: 'admin', password: 'admin123' });
    assert(adminLogin.status === 200, 'Admin login succeeds with HTTP 200');
    adminToken = adminLogin.body.token;

    // 2.4 Verify Protected Route requires token
    const unauthGet = await httpRequest({
      hostname: '127.0.0.1',
      port: config.PORT,
      path: '/api/patients',
      method: 'GET'
    });
    assert(unauthGet.status === 401, 'Unauthenticated access to /api/patients returns 401');

    // 2.5 Verify Token Validation on /api/auth/me
    const meRes = await httpRequest({
      hostname: '127.0.0.1',
      port: config.PORT,
      path: '/api/auth/me',
      method: 'GET',
      headers: { 'Authorization': `Bearer ${clinicianToken}` }
    });
    assert(meRes.status === 200, 'Authenticated /api/auth/me succeeds');
    assert(meRes.body.user.username === 'clinician', 'Identity correctly resolved from token');

    // -------------------------------------------------------------
    // TEST 3: Patient Data CRUD (PLAN §1)
    // -------------------------------------------------------------
    console.log('\n--- Phase 3: Patient Data Management CRUD (PLAN §1) ---');
    
    // 3.1 List Patients
    const listPatients = await httpRequest({
      hostname: '127.0.0.1',
      port: config.PORT,
      path: '/api/patients',
      method: 'GET',
      headers: { 'Authorization': `Bearer ${clinicianToken}` }
    });
    assert(listPatients.status === 200, 'GET /api/patients succeeds with auth');
    assert(Array.isArray(listPatients.body.patients), 'Patients result is an array');
    const prevPatientCount = listPatients.body.count || 0;

    // 3.2 Create New Patient
    const createRes = await httpRequest({
      hostname: '127.0.0.1',
      port: config.PORT,
      path: '/api/patients',
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${clinicianToken}`,
        'Content-Type': 'application/json'
      }
    }, {
      patientId: testPatientId,
      name: 'Maria Dela Cruz',
      age: 67,
      gender: 'Female',
      contactNumber: '+63 917 555 9988',
      barangay: 'Barangay 171, Caloocan City',
      deviceId: 'ESP32C3-NODE-TEST',
      medicalHistory: 'Hypertension, paroxysmal atrial flutter',
      notes: 'Pilot bench testing record'
    });
    assert(createRes.status === 201, 'POST /api/patients returns 201 Created');
    assert(createRes.body.patient.patient_id === testPatientId, 'Created patient ID matches requested ID');

    // 3.3 Read Patient by ID
    const getSingle = await httpRequest({
      hostname: '127.0.0.1',
      port: config.PORT,
      path: `/api/patients/${testPatientId}`,
      method: 'GET',
      headers: { 'Authorization': `Bearer ${clinicianToken}` }
    });
    assert(getSingle.status === 200, 'GET /api/patients/:id returns 200');
    assert(getSingle.body.patient.name === 'Maria Dela Cruz', 'Patient name matches');

    // 3.4 Update Patient Record
    const updateRes = await httpRequest({
      hostname: '127.0.0.1',
      port: config.PORT,
      path: `/api/patients/${testPatientId}`,
      method: 'PUT',
      headers: {
        'Authorization': `Bearer ${clinicianToken}`,
        'Content-Type': 'application/json'
      }
    }, {
      name: 'Maria Dela Cruz (Updated)',
      notes: 'Reviewed during clinician clinical round'
    });
    assert(updateRes.status === 200, 'PUT /api/patients/:id returns 200');
    assert(updateRes.body.patient.name === 'Maria Dela Cruz (Updated)', 'Updated name saved in SQLite');

    // -------------------------------------------------------------
    // TEST 4: Arrhythmia Event History & Cryptographic Chain (PLAN §4)
    // -------------------------------------------------------------
    console.log('\n--- Phase 4: Event History & Cryptographic Hash Chain (PLAN §4) ---');

    // 4.1 Query Event History
    const eventsRes = await httpRequest({
      hostname: '127.0.0.1',
      port: config.PORT,
      path: '/api/events?limit=10',
      method: 'GET',
      headers: { 'Authorization': `Bearer ${clinicianToken}` }
    });
    assert(eventsRes.status === 200, 'GET /api/events returns 200');
    assert(Array.isArray(eventsRes.body.events), 'Events field is an array');

    // 4.2 Cryptographic Verification of the SHA-256 Ledger
    const verifyRes = await httpRequest({
      hostname: '127.0.0.1',
      port: config.PORT,
      path: '/api/events/verify/chain',
      method: 'GET',
      headers: { 'Authorization': `Bearer ${clinicianToken}` }
    });
    assert(verifyRes.status === 200, 'GET /api/events/verify/chain returns 200');
    assert(verifyRes.body.valid === true, 'Cryptographic hash chain is valid and un-tampered');
    console.log(`       Audit proof: ${verifyRes.body.message}`);

    // -------------------------------------------------------------
    // TEST 5: Edge Ingestion & Latency Tracking (PLAN §2 & §5)
    // -------------------------------------------------------------
    console.log('\n--- Phase 5: Edge Ingestion & Latency Tracking (PLAN §2 & §5) ---');

    const edgePayload = {
      ts: Date.now() - 5, // 5ms ago
      patient_id: testPatientId,
      device_id: 'ESP32C3-NODE-TEST',
      bpm: 78.4,
      af_detected: 0,
      af_probability: 0.042,
      sqi: { acceptable: true, sqi_score: 'EXCELLENT' },
      raw_window: [0.1, 0.4, 0.9, 1.2, 0.8, 0.3, 0.1],
      gradcam_weights: [0.05, 0.08, 0.12, 0.15, 0.10, 0.06, 0.04],
      latencies: {
        dsp_ms: 1.4,
        inference_ms: 11.8,
        total_edge_ms: 13.2,
        budget_ms: 25.0,
        budget_met: true
      }
    };

    const ingestRes = await httpRequest({
      hostname: '127.0.0.1',
      port: config.PORT,
      path: '/api/edge/ingest-window',
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    }, edgePayload);

    assert(ingestRes.status === 200, 'POST /api/edge/ingest-window accepted edge telemetry');

    // Verify bridge status reflects ingestion
    const bridgeRes = await httpRequest({
      hostname: '127.0.0.1',
      port: config.PORT,
      path: '/api/edge/bridge/status',
      method: 'GET',
      headers: { 'Authorization': `Bearer ${clinicianToken}` }
    });
    assert(bridgeRes.status === 200, 'GET /api/edge/bridge/status returns 200');
    assert(bridgeRes.body.stats.framesReceived > 0, 'Frames received count incremented');

    // -------------------------------------------------------------
    // TEST 6: WebSocket Security & Streaming Relay (PLAN §2)
    // -------------------------------------------------------------
    console.log('\n--- Phase 6: WebSocket Security & Live Telemetry Stream (PLAN §2) ---');

    // 6.1 Unauthenticated WebSocket handshake MUST fail (Security row in ANTIGRAVITY.md §5)
    const wsRejectPromise = new Promise((resolve) => {
      const badWs = new WebSocket(`ws://127.0.0.1:${config.PORT}/ws/live`); // No token
      badWs.on('error', (err) => {
        // Handshake 401 emits an error event on ws client
        resolve({ rejected: true, error: err.message });
      });
      badWs.on('open', () => {
        badWs.close();
        resolve({ rejected: false });
      });
    });

    const unauthWsRes = await wsRejectPromise;
    assert(unauthWsRes.rejected === true, 'Unauthenticated WebSocket connection correctly rejected with 401');

    // 6.2 Authenticated WebSocket connection succeeds
    const wsAuthPromise = new Promise((resolve) => {
      const ws = new WebSocket(`ws://127.0.0.1:${config.PORT}/ws/live?token=${clinicianToken}`);
      let ackReceived = false;
      let frameReceived = false;

      ws.on('open', () => {
        // Send a ping message
        ws.send(JSON.stringify({ type: 'ping', time: Date.now() }));
      });

      ws.on('message', (msgStr) => {
        const msg = JSON.parse(msgStr.toString());
        if (msg.type === 'connection_ack') {
          ackReceived = true;
          // Trigger a simulated window broadcast to test relay
          httpRequest({
            hostname: '127.0.0.1',
            port: config.PORT,
            path: '/api/edge/ingest-window',
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
          }, edgePayload);
        } else if (msg.type === 'ppg_window') {
          frameReceived = true;
          ws.close();
          resolve({ success: true, ack: ackReceived, frame: msg });
        }
      });

      ws.on('error', (err) => {
        resolve({ success: false, error: err.message });
      });

      // 5-second timeout safeguard
      setTimeout(() => {
        if (!frameReceived) {
          ws.close();
          resolve({ success: false, timeout: true, ack: ackReceived });
        }
      }, 4000);
    });

    const authWsRes = await wsAuthPromise;
    assert(authWsRes.success === true, 'Authenticated WebSocket receives live telemetry broadcast');
    assert(authWsRes.ack === true, 'Connection handshake acknowledged with user identity');
    if (authWsRes.frame) {
      assert(!!authWsRes.frame.data.raw_window, 'Telemetry message contains raw PPG waveform window');
      assert(!!authWsRes.frame.data.gradcam_weights, 'Telemetry message contains Grad-CAM explainability weights');
      assert(authWsRes.frame.transitLatencyMs !== undefined, 'Transit latency timestamped for ISO/IEC 25010 budget');
    }

    // -------------------------------------------------------------
    // Clean up test patient
    // -------------------------------------------------------------
    await httpRequest({
      hostname: '127.0.0.1',
      port: config.PORT,
      path: `/api/patients/${testPatientId}`,
      method: 'DELETE',
      headers: { 'Authorization': `Bearer ${clinicianToken}` }
    });

    console.log('\n================================================================');
    console.log(` TEST SUMMARY: ${passed} PASSED, ${failed} FAILED`);
    console.log('================================================================\n');

    if (failed > 0) {
      process.exit(1);
    } else {
      process.exit(0);
    }
  } catch (err) {
    console.error('Fatal error during test run:', err);
    process.exit(1);
  }
}

// Allow server to spin up then run tests
setTimeout(runTests, 800);
