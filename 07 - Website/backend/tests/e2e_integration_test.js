/**
 * End-to-End Integration Benchmark: Python Edge Runner -> TCP Bridge -> Node.js -> WebSocket Client
 * Validates PLAN §0, §1, §2, §3, §4, §5 & ISO/IEC 25010 Performance Efficiency
 */

const { spawn } = require('child_process');
const path = require('path');
const { WebSocket } = require('ws');
const jwt = require('jsonwebtoken');

process.env.JWT_SECRET = process.env.JWT_SECRET || 'test-jwt-secret-minimum-32-chars-key-thesis';
process.env.ADMIN_USERNAME = process.env.ADMIN_USERNAME || 'admin';
process.env.ADMIN_PASSWORD = process.env.ADMIN_PASSWORD || 'ue_thesis_2026';
process.env.ENABLE_SIMULATION = 'true';

const config = require('../src/config');
const db = require('../src/db');
const { server } = require('../src/server');

async function runE2E() {
  console.log('\n================================================================');
  console.log(' STARTING END-TO-END PIPELINE INTEGRATION TEST');
  console.log(' Python Runner (DSP+CNN+GradCAM) -> TCP 5051 -> Node.js -> WS Client');
  console.log('================================================================\n');

  // Generate test clinician token
  const clinicianToken = jwt.sign({
    userId: 'usr-clin-01',
    username: 'clinician',
    role: 'clinician',
    fullName: 'Dr. Maria Santos'
  }, config.JWT_SECRET, { expiresIn: '1h' });

  // Connect authenticated WebSocket client
  const ws = new WebSocket(`ws://127.0.0.1:${config.PORT}/ws/live?token=${clinicianToken}`);

  let framesReceived = 0;
  let hasGradCam = false;
  let latencies = [];

  const wsReady = new Promise((resolve) => {
    ws.on('open', () => {
      console.log('[E2E WS Client] Connected to ws://127.0.0.1:' + config.PORT + '/ws/live');
      resolve();
    });
  });

  ws.on('message', (msgStr) => {
    try {
      const msg = JSON.parse(msgStr.toString());
      if (msg.type === 'ppg_window') {
        framesReceived++;
        const d = msg.data;
        if (d.gradcam_weights && d.gradcam_weights.length > 0) {
          hasGradCam = true;
        }
        if (d.latencies) {
          latencies.push(d.latencies);
        }
        console.log(`[E2E WS Received Frame ${framesReceived}] BPM=${d.bpm} | AF_Prob=${(d.af_probability * 100).toFixed(1)}% | Infer=${d.latencies.inference_ms}ms | BridgeTransit=${msg.transitLatencyMs}ms`);
      }
    } catch (e) {
      console.error('[E2E Error]', e);
    }
  });

  await wsReady;

  // Run Python runner for 8 seconds
  console.log('[E2E] Spawning Python edge inference process...');
  const pythonPath = 'py';
  const runnerScript = path.resolve(__dirname, '../../../03 - ML/edge_inference/runner.py');

  const pyProcess = spawn(pythonPath, [runnerScript, '--duration', '5', '--patient', 'PAT-CAL-001', '--prefill'], {
    cwd: path.resolve(__dirname, '../../../03 - ML')
  });

  pyProcess.stdout.on('data', (data) => {
    process.stdout.write(`  [Python Out] ${data}`);
  });

  pyProcess.stderr.on('data', (data) => {
    process.stderr.write(`  [Python Err] ${data}`);
  });

  await new Promise((resolve) => {
    pyProcess.on('close', (code) => {
      console.log(`[E2E] Python process exited with code ${code}`);
      resolve();
    });
  });

  // Allow last frames to flush
  await new Promise(r => setTimeout(r, 1000));
  ws.close();

  // Validate results
  console.log('\n================================================================');
  console.log(' E2E VERIFICATION RESULTS:');
  console.log(` - Frames Delivered to Browser: ${framesReceived}`);
  console.log(` - Grad-CAM Attention Vectors Attached: ${hasGradCam}`);
  
  if (latencies.length > 0) {
    const avgInfer = latencies.reduce((a, b) => a + b.inference_ms, 0) / latencies.length;
    console.log(` - Average CNN Inference Time: ${avgInfer.toFixed(2)} ms (Budget: <25ms -> ${avgInfer < 25.0 ? 'COMPLIANT' : 'EXCEEDED'})`);
  }

  // Validate SQLite Hash Chain
  const chainAudit = db.validateLedgerChain();
  console.log(` - SQLite Hash Chain Status: ${chainAudit.valid ? 'VALID & UNTAMPERED' : 'BROKEN'}`);
  console.log(` - Total Hash Chain Events: ${chainAudit.count}`);
  console.log('================================================================\n');

  if (framesReceived > 0 && hasGradCam && chainAudit.valid) {
    console.log('SUCCESS: End-to-End Edge to Web Dashboard pipeline verified.');
    process.exit(0);
  } else {
    console.error('FAILURE: E2E Pipeline did not meet all delivery or integrity criteria.');
    process.exit(1);
  }
}

setTimeout(runE2E, 600);
