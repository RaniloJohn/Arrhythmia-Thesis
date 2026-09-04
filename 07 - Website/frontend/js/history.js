/**
 * Arrhythmia Event History & Cryptographic Traceability Controller (PLAN §4 & RQ4)
 * Thesis: An Internet of Things-Based Framework for Cardiac Arrhythmia Detection
 *         via Photoplethysmography and Machine Learning (3CPE-2A, Univ. of the East)
 */

class HistoryManager {
  constructor() {
    this.events = [];
    this.tableBody = document.getElementById('eventsTableBody');
    this.filterPatient = document.getElementById('historyPatientFilter');
    this.filterAfOnly = document.getElementById('historyAfOnlyFilter');
    this.filterSync = document.getElementById('historySyncFilter');
    this.btnVerifyChain = document.getElementById('btnVerifyChain');

    // Audit Modal
    this.auditModal = document.getElementById('auditModal');
    this.auditBody = document.getElementById('auditModalBody');
    this.btnCloseAudit = document.getElementById('btnCloseAuditModal');

    // Event Detail Modal
    this.eventDetailModal = document.getElementById('eventDetailModal');
    this.eventDetailBody = document.getElementById('eventDetailBody');
    this.btnCloseEventDetail = document.getElementById('btnCloseEventDetail');

    this._bindEvents();
  }

  _bindEvents() {
    if (this.filterPatient) {
      this.filterPatient.addEventListener('change', () => this.loadEvents());
    }
    if (this.filterAfOnly) {
      this.filterAfOnly.addEventListener('change', () => this.loadEvents());
    }
    if (this.filterSync) {
      this.filterSync.addEventListener('change', () => this.loadEvents());
    }
    if (this.btnVerifyChain) {
      this.btnVerifyChain.addEventListener('click', () => this.verifyLedgerChain());
    }
    if (this.btnCloseAudit) {
      this.btnCloseAudit.addEventListener('click', () => this.closeAuditModal());
    }
    if (this.btnCloseEventDetail) {
      this.btnCloseEventDetail.addEventListener('click', () => this.closeDetailModal());
    }
  }

  async loadEvents() {
    if (!window.api.getToken()) return;

    const filters = {
      patientId: this.filterPatient ? this.filterPatient.value : null,
      afOnly: this.filterAfOnly ? this.filterAfOnly.checked : false,
      syncStatus: this.filterSync ? this.filterSync.value : '',
      limit: 100
    };

    try {
      const res = await window.api.getEvents(filters);
      this.events = res.events || [];
      this.renderTable();
      this.updatePatientFilterOptions();
    } catch (err) {
      window.showToast(`Failed to load event history: ${err.message}`, 'error');
    }
  }

  updatePatientFilterOptions() {
    if (!this.filterPatient || !window.patientsManager) return;
    const current = this.filterPatient.value;
    const patients = window.patientsManager.patients;

    this.filterPatient.innerHTML = '<option value="">All Patients</option>';
    for (const p of patients) {
      const opt = document.createElement('option');
      opt.value = p.patient_id;
      opt.textContent = `${p.patient_id} — ${p.name}`;
      this.filterPatient.appendChild(opt);
    }
    this.filterPatient.value = current;
  }

  renderTable() {
    if (!this.tableBody) return;
    this.tableBody.innerHTML = '';

    if (this.events.length === 0) {
      this.tableBody.innerHTML = `
        <tr>
          <td colspan="8" style="text-align: center; color: var(--text-muted); padding: 2rem;">
            No arrhythmia events found matching the selected filters.
          </td>
        </tr>
      `;
      return;
    }

    for (const e of this.events) {
      const tr = document.createElement('tr');
      const dateStr = new Date(e.timestamp).toLocaleString();
      const isAf = e.af_detected === 1;
      const isSync = e.sync_status === 1;
      const confPct = (e.confidence * 100).toFixed(1);

      tr.innerHTML = `
        <td><span style="font-size: 0.8rem; color: var(--text-secondary);">${dateStr}</span></td>
        <td>
          <strong>${e.patient_name || e.patient_id || 'Unassigned'}</strong>
          <div style="font-size: 0.75rem; color: var(--text-muted);">${e.patient_id || ''}</div>
        </td>
        <td><span style="font-family: var(--font-mono); font-size: 0.75rem; color: #7dd3fc;">${e.device_id}</span></td>
        <td><strong>${Number(e.bpm).toFixed(1)}</strong> <span style="font-size: 0.75rem; color: var(--text-muted);">BPM</span></td>
        <td>
          <span style="font-weight: 700; color: ${isAf ? 'var(--accent-alert)' : 'var(--accent-success)'};">
            ${isAf ? 'AF DETECTED' : 'NSR'}
          </span>
          <div style="font-size: 0.75rem; color: var(--text-muted);">${confPct}% conf</div>
        </td>
        <td>
          <span class="badge-sync ${isSync ? 'blockchain' : 'local'}">
            ${isSync ? 'Blockchain Confirmed' : 'Local Only'}
          </span>
        </td>
        <td class="hash-cell" title="${e.record_hash}">${e.record_hash}</td>
        <td>
          <button class="btn-secondary" style="padding: 0.25rem 0.5rem; font-size: 0.75rem;" onclick="historyManager.openDetail('${e.event_id}')">
            Inspect
          </button>
        </td>
      `;
      this.tableBody.appendChild(tr);
    }
  }

  async verifyLedgerChain() {
    try {
      this.btnVerifyChain.disabled = true;
      this.btnVerifyChain.innerText = 'Verifying SHA-256 Hashes...';

      const audit = await window.api.verifyChain();
      this.showAuditResult(audit);
    } catch (err) {
      window.showToast(`Verification failed: ${err.message}`, 'error');
    } finally {
      this.btnVerifyChain.disabled = false;
      this.btnVerifyChain.innerHTML = '<svg class="btn-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><polyline points="9 12 11 14 15 10"></polyline></svg> Verify Cryptographic Hash Chain';
    }
  }

  showAuditResult(audit) {
    if (!this.auditModal || !this.auditBody) return;

    const isPass = audit.valid;
    const statusIcon = isPass
      ? '<svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="var(--accent-success)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><polyline points="9 12 11 14 15 10"></polyline></svg>'
      : '<svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="var(--accent-alert)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>';

    this.auditBody.innerHTML = `
      <div style="text-align: center; margin-bottom: 1.5rem;">
        <div style="font-size: 3rem; display: flex; justify-content: center;">${statusIcon}</div>
        <h4 style="font-size: 1.3rem; color: ${isPass ? 'var(--accent-success)' : 'var(--accent-alert)'}; margin-top: 0.5rem;">
          ${isPass ? 'Cryptographic Integrity Verified' : 'Cryptographic Chain Violation Detected'}
        </h4>
        <p style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 0.3rem;">
          ${audit.message}
        </p>
      </div>

      <div style="background-color: var(--bg-secondary); padding: 1rem; border-radius: 8px; font-size: 0.85rem; border: 1px solid var(--border-color);">
        <div class="patient-info-row"><span class="patient-info-label">Standard:</span> <span class="patient-info-val">ISO/IEC 25010 Security / Non-Repudiation</span></div>
        <div class="patient-info-row"><span class="patient-info-label">Hashing Algorithm:</span> <span class="patient-info-val" style="font-family: var(--font-mono);">SHA-256 Backward Chaining</span></div>
        <div class="patient-info-row"><span class="patient-info-label">Total Verified Events:</span> <span class="patient-info-val">${audit.count} records</span></div>
        <div class="patient-info-row"><span class="patient-info-label">Database Mode:</span> <span class="patient-info-val">SQLite WAL (Concurrent Isolation)</span></div>
        <div class="patient-info-row" style="flex-direction: column; margin-top: 0.5rem;">
          <span class="patient-info-label">Latest Cryptographic Record Hash:</span>
          <span class="hash-cell" style="font-size: 0.75rem; word-break: break-all; margin-top: 0.2rem;">${audit.latestHash || 'Genesis State'}</span>
        </div>
      </div>
    `;

    this.auditModal.classList.add('active');
  }

  closeAuditModal() {
    if (this.auditModal) this.auditModal.classList.remove('active');
  }

  async openDetail(eventId) {
    try {
      const res = await window.api.getEvent(eventId);
      const e = res.event;

      let gradcamHtml = '<p style="color: var(--text-muted); font-size: 0.8rem;">No Grad-CAM vector stored.</p>';
      if (e.gradcam && e.gradcam.high_regions) {
        gradcamHtml = `
          <div style="display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.4rem;">
            ${e.gradcam.high_regions.map(r => `
              <span style="background: rgba(193, 59, 59, 0.08); border: 1px solid var(--accent-alert); color: var(--accent-alert); font-size: 0.75rem; padding: 0.2rem 0.5rem; border-radius: 4px;">
                Interval [${r.start_idx}-${r.end_idx}]: Rel=${(r.peak_weight * 100).toFixed(0)}%
              </span>
            `).join('')}
          </div>
        `;
      }

      this.eventDetailBody.innerHTML = `
        <div style="display: flex; flex-direction: column; gap: 0.85rem; font-size: 0.85rem;">
          <div class="patient-info-row"><span class="patient-info-label">Event ID:</span> <span class="patient-info-val" style="font-family: var(--font-mono);">${e.event_id}</span></div>
          <div class="patient-info-row"><span class="patient-info-label">Patient:</span> <span class="patient-info-val">${e.patient_name || e.patient_id} (${e.patient_age}y / ${e.patient_gender})</span></div>
          <div class="patient-info-row"><span class="patient-info-label">Timestamp:</span> <span class="patient-info-val">${new Date(e.timestamp).toLocaleString()}</span></div>
          <div class="patient-info-row"><span class="patient-info-label">Heart Rate:</span> <span class="patient-info-val">${e.bpm.toFixed(1)} BPM</span></div>
          <div class="patient-info-row"><span class="patient-info-label">Diagnosis:</span> <span class="patient-info-val" style="color: ${e.af_detected ? 'var(--accent-alert)' : 'var(--accent-success)'}; font-weight: 700;">${e.af_detected ? 'Atrial Fibrillation (AF)' : 'Normal Sinus Rhythm (NSR)'}</span></div>
          <div class="patient-info-row"><span class="patient-info-label">1D-CNN Confidence:</span> <span class="patient-info-val">${(e.confidence * 100).toFixed(2)}%</span></div>
          <div class="patient-info-row"><span class="patient-info-label">Blockchain Status:</span> <span class="badge-sync ${e.sync_status ? 'blockchain' : 'local'}">${e.sync_status ? 'Blockchain Confirmed' : 'Local Only'}</span></div>
          
          <div style="background-color: var(--bg-secondary); padding: 0.75rem; border-radius: 6px; border: 1px solid var(--border-color);">
            <div style="font-weight: 700; color: var(--accent-primary); margin-bottom: 0.3rem;">1D Grad-CAM High Relevance Regions:</div>
            ${gradcamHtml}
          </div>

          <div style="background-color: var(--bg-secondary); padding: 0.75rem; border-radius: 6px; border: 1px solid var(--border-color);">
            <div style="font-weight: 700; color: var(--text-muted); margin-bottom: 0.3rem;">Cryptographic Hash Chaining Proof:</div>
            <div style="font-size: 0.75rem; color: var(--text-muted);">Previous Hash:</div>
            <div class="hash-cell" style="font-size: 0.7rem; margin-bottom: 0.4rem;">${e.prev_hash}</div>
            <div style="font-size: 0.75rem; color: var(--text-muted);">Record Hash:</div>
            <div class="hash-cell" style="font-size: 0.7rem;">${e.record_hash}</div>
          </div>
        </div>
      `;

      this.eventDetailModal.classList.add('active');
    } catch (err) {
      window.showToast(`Error fetching event: ${err.message}`, 'error');
    }
  }

  closeDetailModal() {
    if (this.eventDetailModal) this.eventDetailModal.classList.remove('active');
  }
}

window.historyManager = new HistoryManager();
