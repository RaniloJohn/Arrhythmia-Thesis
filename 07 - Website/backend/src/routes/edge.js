/**
 * Edge Gateway Ingestion & Telemetry Routes
 * Thesis: An Internet of Things-Based Framework for Cardiac Arrhythmia Detection
 *         via Photoplethysmography and Machine Learning (3CPE-2A, Univ. of the East)
 */

const express = require('express');
const pubsubRelay = require('../bridge/pubsub_relay');
const { authenticateToken } = require('../middleware/auth');

const router = express.Router();

/**
 * POST /api/edge/ingest-window
 * HTTP ingestion endpoint called by 03 - ML/edge_inference/runner.py as fallback.
 */
router.post('/ingest-window', (req, res) => {
  try {
    const payload = req.body;
    if (!payload || !payload.raw_window) {
      return res.status(400).json({ error: 'Bad Request', message: 'Window payload with raw_window required.' });
    }

    pubsubRelay.handleIncomingWindow(payload, 'http_post');
    return res.status(200).json({ status: 'relayed', receivedAt: Date.now() });
  } catch (err) {
    console.error('[Edge Ingest] Error:', err.message);
    return res.status(500).json({ error: 'Internal Server Error', message: err.message });
  }
});

/**
 * GET /api/edge/bridge/status
 * Telemetry and latency statistics for ISO/IEC 25010 audit.
 */
router.get('/bridge/status', authenticateToken, (req, res) => {
  return res.json({
    status: 'online',
    timestamp: new Date().toISOString(),
    stats: pubsubRelay.getStats()
  });
});

/**
 * POST /api/edge/simulation/toggle
 * Toggle standalone simulation for bench demonstration without hardware.
 */
router.post('/simulation/toggle', authenticateToken, (req, res) => {
  const { enable, patientId } = req.body || {};
  if (enable) {
    pubsubRelay.startSimulation(patientId || 'PAT-CAL-001');
  } else {
    pubsubRelay.stopSimulation();
  }
  return res.json({
    simulationActive: pubsubRelay.simRunning,
    message: pubsubRelay.simRunning ? 'Standalone simulation started' : 'Standalone simulation stopped'
  });
});

module.exports = router;
