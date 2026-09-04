/**
 * Patient Registry Management Controller (PLAN §1)
 * Thesis: An Internet of Things-Based Framework for Cardiac Arrhythmia Detection
 *         via Photoplethysmography and Machine Learning (3CPE-2A, Univ. of the East)
 */

class PatientsManager {
  constructor() {
    this.patients = [];
    this.tableBody = document.getElementById('patientsTableBody');
    this.searchInput = document.getElementById('patientSearchInput');
    this.btnAddPatient = document.getElementById('btnAddPatient');
    
    // Modals
    this.modal = document.getElementById('patientModal');
    this.form = document.getElementById('patientForm');
    this.modalTitle = document.getElementById('patientModalTitle');
    this.btnCloseModal = document.getElementById('btnClosePatientModal');
    this.btnCancelModal = document.getElementById('btnCancelPatientModal');

    // Inputs
    this.inpId = document.getElementById('pFormId');
    this.inpName = document.getElementById('pFormName');
    this.inpAge = document.getElementById('pFormAge');
    this.inpGender = document.getElementById('pFormGender');
    this.inpContact = document.getElementById('pFormContact');
    this.inpBarangay = document.getElementById('pFormBarangay');
    this.inpDevice = document.getElementById('pFormDevice');
    this.inpHistory = document.getElementById('pFormHistory');
    this.inpNotes = document.getElementById('pFormNotes');

    this.editingId = null;

    this._bindEvents();
  }

  _bindEvents() {
    if (this.searchInput) {
      let debounceTimer;
      this.searchInput.addEventListener('input', (e) => {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => this.loadPatients(e.target.value), 250);
      });
    }

    if (this.btnAddPatient) {
      this.btnAddPatient.addEventListener('click', () => this.openAddModal());
    }

    if (this.btnCloseModal) {
      this.btnCloseModal.addEventListener('click', () => this.closeModal());
    }

    if (this.btnCancelModal) {
      this.btnCancelModal.addEventListener('click', () => this.closeModal());
    }

    if (this.form) {
      this.form.addEventListener('submit', (e) => this.handleSave(e));
    }
  }

  getPatientData(patientId) {
    return this.patients.find(p => p.patient_id === patientId) || null;
  }

  async loadPatients(search = '') {
    if (!window.api.getToken()) return;

    try {
      const res = await window.api.getPatients(search);
      this.patients = res.patients || [];
      this.renderTable();
      this.updateLiveDropdown();
    } catch (err) {
      window.showToast(`Failed to load patients: ${err.message}`, 'error');
    }
  }

  renderTable() {
    if (!this.tableBody) return;
    this.tableBody.innerHTML = '';

    if (this.patients.length === 0) {
      this.tableBody.innerHTML = `
        <tr>
          <td colspan="7" style="text-align: center; color: var(--text-muted); padding: 2rem;">
            No patient records found. Click "+ Register New Patient" to add one.
          </td>
        </tr>
      `;
      return;
    }

    for (const p of this.patients) {
      const tr = document.createElement('tr');
      const afCount = p.af_episodes || 0;
      const lastSeen = p.last_event_time ? new Date(p.last_event_time).toLocaleDateString() : 'Never';

      tr.innerHTML = `
        <td><strong style="color: var(--accent-primary);">${p.patient_id}</strong></td>
        <td>
          <div style="font-weight: 600;">${p.name}</div>
          <div style="font-size: 0.75rem; color: var(--text-muted);">${p.contact_number || 'No contact'}</div>
        </td>
        <td>${p.age} / ${p.gender}</td>
        <td>${p.barangay}</td>
        <td><span style="font-family: var(--font-mono); font-size: 0.8rem; color: var(--accent-primary);">${p.device_id}</span></td>
        <td>
          <span style="font-weight: 700; color: ${afCount > 0 ? 'var(--accent-alert)' : 'var(--accent-success)'};">
            ${afCount} episodes
          </span>
          <div style="font-size: 0.7rem; color: var(--text-muted);">Last: ${lastSeen}</div>
        </td>
        <td>
          <div style="display: flex; gap: 0.4rem;">
            <button class="btn-secondary" style="padding: 0.3rem 0.6rem; font-size: 0.75rem;" onclick="patientsManager.selectForLive('${p.patient_id}')">Monitor</button>
            <button class="btn-secondary" style="padding: 0.3rem 0.6rem; font-size: 0.75rem;" onclick="patientsManager.openEditModal('${p.patient_id}')">Edit</button>
            <button class="btn-danger" style="padding: 0.3rem 0.6rem; font-size: 0.75rem;" onclick="patientsManager.deletePatient('${p.patient_id}')">Delete</button>
          </div>
        </td>
      `;
      this.tableBody.appendChild(tr);
    }
  }

  updateLiveDropdown() {
    const select = document.getElementById('livePatientSelect');
    if (!select) return;

    const currentVal = select.value;
    select.innerHTML = '';

    for (const p of this.patients) {
      const opt = document.createElement('option');
      opt.value = p.patient_id;
      opt.textContent = `${p.patient_id} — ${p.name} (${p.device_id})`;
      select.appendChild(opt);
    }

    if (currentVal && this.patients.some(p => p.patient_id === currentVal)) {
      select.value = currentVal;
    } else if (this.patients.length > 0) {
      select.value = this.patients[0].patient_id;
      if (window.liveStream) {
        window.liveStream.selectedPatientId = this.patients[0].patient_id;
        window.liveStream.updateSelectedPatientInfo();
      }
    }
  }

  selectForLive(patientId) {
    if (window.switchTab) {
      window.switchTab('live-view');
    }
    const select = document.getElementById('livePatientSelect');
    if (select) {
      select.value = patientId;
      select.dispatchEvent(new Event('change'));
    }
  }

  openAddModal() {
    this.editingId = null;
    this.modalTitle.innerText = 'Register New Patient';
    this.form.reset();
    this.inpId.value = `PAT-CAL-${Math.floor(100 + Math.random() * 900)}`;
    this.modal.classList.add('active');
  }

  openEditModal(patientId) {
    const p = this.getPatientData(patientId);
    if (!p) return;

    this.editingId = patientId;
    this.modalTitle.innerText = `Edit Patient: ${patientId}`;
    this.inpId.value = p.patient_id;
    this.inpName.value = p.name;
    this.inpAge.value = p.age;
    this.inpGender.value = p.gender;
    this.inpContact.value = p.contact_number || '';
    this.inpBarangay.value = p.barangay;
    this.inpDevice.value = p.device_id;
    this.inpHistory.value = p.medical_history || '';
    this.inpNotes.value = p.notes || '';

    this.modal.classList.add('active');
  }

  closeModal() {
    this.modal.classList.remove('active');
    this.editingId = null;
  }

  async handleSave(e) {
    e.preventDefault();

    const payload = {
      patientId: this.inpId.value.trim(),
      name: this.inpName.value.trim(),
      age: parseInt(this.inpAge.value, 10),
      gender: this.inpGender.value,
      contactNumber: this.inpContact.value.trim(),
      barangay: this.inpBarangay.value.trim(),
      deviceId: this.inpDevice.value.trim(),
      medicalHistory: this.inpHistory.value.trim(),
      notes: this.inpNotes.value.trim()
    };

    if (!payload.name || !payload.age || !payload.barangay || !payload.deviceId) {
      window.showToast('Please fill all required demographic fields.', 'error');
      return;
    }

    try {
      if (this.editingId) {
        await window.api.updatePatient(this.editingId, payload);
        window.showToast('Patient record updated successfully', 'success');
      } else {
        await window.api.createPatient(payload);
        window.showToast('New patient registered successfully', 'success');
      }

      this.closeModal();
      await this.loadPatients();
    } catch (err) {
      window.showToast(err.message, 'error');
    }
  }

  async deletePatient(patientId) {
    if (!confirm(`Are you sure you want to delete patient record '${patientId}'?`)) {
      return;
    }

    try {
      await window.api.deletePatient(patientId);
      window.showToast('Patient record removed', 'info');
      await this.loadPatients();
    } catch (err) {
      window.showToast(err.message, 'error');
    }
  }
}

window.patientsManager = new PatientsManager();
