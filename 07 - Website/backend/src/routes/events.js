/**
 * Arrhythmia Event History & Cryptographic Verification Routes (PLAN §4)
 * Thesis: An Internet of Things-Based Framework for Cardiac Arrhythmia Detection
 *         via Photoplethysmography and Machine Learning (3CPE-2A, Univ. of the East)
 * 
 * Features:
 * - Diagnostic AF history retrieval with per-patient filtering.
 * - Blockchain sync status inspection (Local Only vs Blockchain Confirmed).
 * - Live cryptographic verification of the SHA-256 backward hash chain.
 */

const express = require('express');
const db = require('../db');
const { authenticateToken } = require('../middleware/auth');

const router = express.Router();

router.use(authenticateToken);

/**
 * GET /api/events
 * Query historical arrhythmia events across all or specific patients.
 */
router.get('/', (req, res) => {
  try {
    const { patientId, afOnly, syncStatus, limit, offset } = req.query;

    const events = db.getEvents({
      patientId: patientId || null,
      afOnly: afOnly === 'true',
      syncStatus: syncStatus !== undefined ? syncStatus : null,
      limit: parseInt(limit || '50', 10),
      offset: parseInt(offset || '0', 10)
    });

    return res.json({
      count: events.length,
      events
    });
  } catch (err) {
    console.error('Error querying events:', err);
    return res.status(500).json({ error: 'Internal Server Error', message: err.message });
  }
});

/**
 * GET /api/events/verify/chain
 * Mathematically validates the SHA-256 backward hash chain on SQLite.
 * Answers RQ4 and ISO/IEC 25010 Security audits.
 */
router.get('/verify/chain', (req, res) => {
  try {
    const auditResult = db.validateLedgerChain();
    return res.json({
      timestamp: new Date().toISOString(),
      ...auditResult
    });
  } catch (err) {
    console.error('Error validating hash chain:', err);
    return res.status(500).json({ error: 'Internal Server Error', message: err.message });
  }
});

/**
 * GET /api/events/:id
 * Retrieve detailed event record including Grad-CAM explainability payload.
 */
router.get('/:id', (req, res) => {
  try {
    const event = db.getEventById(req.params.id);
    if (!event) {
      return res.status(404).json({ error: 'Not Found', message: `Event '${req.params.id}' not found.` });
    }

    let gradcamParsed = null;
    if (event.gradcam_path) {
      try {
        gradcamParsed = JSON.parse(event.gradcam_path);
      } catch (e) {
        gradcamParsed = event.gradcam_path;
      }
    }

    return res.json({
      event: {
        ...event,
        gradcam: gradcamParsed
      }
    });
  } catch (err) {
    console.error('Error fetching event details:', err);
    return res.status(500).json({ error: 'Internal Server Error', message: err.message });
  }
});

module.exports = router;
