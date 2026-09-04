/**
 * Patient Data Management REST Endpoints (PLAN §1 & §3)
 * Thesis: An Internet of Things-Based Framework for Cardiac Arrhythmia Detection
 *         via Photoplethysmography and Machine Learning (3CPE-2A, Univ. of the East)
 * 
 * Rules:
 * - All endpoints protected by clinician authentication.
 * - Written only by this Node.js backend, never by the Python edge process.
 */

const express = require('express');
const db = require('../db');
const { authenticateToken, requireRole } = require('../middleware/auth');

const router = express.Router();

// Apply auth middleware to all patient routes
router.use(authenticateToken);

/**
 * GET /api/patients
 * List all patient profiles with incident summaries.
 */
router.get('/', (req, res) => {
  try {
    const search = req.query.q || '';
    const patients = db.getAllPatients(search);
    return res.json({
      count: patients.length,
      patients
    });
  } catch (err) {
    console.error('Error fetching patients:', err);
    return res.status(500).json({ error: 'Internal Server Error', message: err.message });
  }
});

/**
 * GET /api/patients/:id
 * Retrieve full patient details and recent diagnostic events.
 */
router.get('/:id', (req, res) => {
  try {
    const patient = db.getPatientById(req.params.id);
    if (!patient) {
      return res.status(404).json({ error: 'Not Found', message: `Patient '${req.params.id}' not found.` });
    }

    // Attach last 20 events for quick clinician overview
    const recentEvents = db.getEvents({
      patientId: req.params.id,
      limit: 20
    });

    return res.json({
      patient,
      recentEvents
    });
  } catch (err) {
    console.error('Error fetching patient:', err);
    return res.status(500).json({ error: 'Internal Server Error', message: err.message });
  }
});

/**
 * GET /api/patients/:id/events
 * Retrieve historical events for a specific patient.
 */
router.get('/:id/events', (req, res) => {
  try {
    const patient = db.getPatientById(req.params.id);
    if (!patient) {
      return res.status(404).json({ error: 'Not Found', message: `Patient '${req.params.id}' not found.` });
    }

    const afOnly = req.query.afOnly === 'true';
    const limit = parseInt(req.query.limit || '50', 10);
    const offset = parseInt(req.query.offset || '0', 10);

    const events = db.getEvents({
      patientId: req.params.id,
      afOnly,
      limit,
      offset
    });

    return res.json({
      patientId: req.params.id,
      count: events.length,
      events
    });
  } catch (err) {
    console.error('Error fetching patient events:', err);
    return res.status(500).json({ error: 'Internal Server Error', message: err.message });
  }
});

/**
 * POST /api/patients
 * Register a new primary care patient record.
 */
router.post('/', (req, res) => {
  try {
    const {
      patientId,
      name,
      age,
      gender,
      contactNumber,
      barangay,
      deviceId,
      medicalHistory,
      notes
    } = req.body || {};

    if (!name || !age || !gender || !barangay || !deviceId) {
      return res.status(400).json({
        error: 'Bad Request',
        message: 'Name, age, gender, barangay, and assigned deviceId are required fields.'
      });
    }

    if (patientId) {
      const existing = db.getPatientById(patientId);
      if (existing) {
        return res.status(409).json({
          error: 'Conflict',
          message: `Patient with ID '${patientId}' already exists.`
        });
      }
    }

    const created = db.createPatient({
      patientId,
      name,
      age,
      gender,
      contactNumber,
      barangay,
      deviceId,
      medicalHistory,
      notes
    });

    return res.status(201).json({
      message: 'Patient record created successfully',
      patient: created
    });
  } catch (err) {
    console.error('Error creating patient:', err);
    return res.status(500).json({ error: 'Internal Server Error', message: err.message });
  }
});

/**
 * PUT /api/patients/:id
 * Update an existing patient record.
 */
router.put('/:id', (req, res) => {
  try {
    const existing = db.getPatientById(req.params.id);
    if (!existing) {
      return res.status(404).json({ error: 'Not Found', message: `Patient '${req.params.id}' not found.` });
    }

    const {
      name,
      age,
      gender,
      contactNumber,
      barangay,
      deviceId,
      medicalHistory,
      notes
    } = req.body || {};

    const updated = db.updatePatient(req.params.id, {
      name: name !== undefined ? name : existing.name,
      age: age !== undefined ? age : existing.age,
      gender: gender !== undefined ? gender : existing.gender,
      contactNumber: contactNumber !== undefined ? contactNumber : existing.contact_number,
      barangay: barangay !== undefined ? barangay : existing.barangay,
      deviceId: deviceId !== undefined ? deviceId : existing.device_id,
      medicalHistory: medicalHistory !== undefined ? medicalHistory : existing.medical_history,
      notes: notes !== undefined ? notes : existing.notes
    });

    return res.json({
      message: 'Patient record updated successfully',
      patient: updated
    });
  } catch (err) {
    console.error('Error updating patient:', err);
    return res.status(500).json({ error: 'Internal Server Error', message: err.message });
  }
});

/**
 * DELETE /api/patients/:id
 * Remove a patient record.
 */
router.delete('/:id', (req, res) => {
  try {
    const existing = db.getPatientById(req.params.id);
    if (!existing) {
      return res.status(404).json({ error: 'Not Found', message: `Patient '${req.params.id}' not found.` });
    }

    db.deletePatient(req.params.id);
    return res.json({
      message: `Patient record '${req.params.id}' removed successfully. Related event records unlinked.`
    });
  } catch (err) {
    console.error('Error deleting patient:', err);
    return res.status(500).json({ error: 'Internal Server Error', message: err.message });
  }
});

module.exports = router;
