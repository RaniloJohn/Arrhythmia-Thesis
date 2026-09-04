/**
 * Clinician Decision Support Application Main Orchestrator
 * Thesis: An Internet of Things-Based Framework for Cardiac Arrhythmia Detection
 *         via Photoplethysmography and Machine Learning (3CPE-2A, Univ. of the East)
 */

document.addEventListener('DOMContentLoaded', async () => {
  // Global Toast Helper
  window.showToast = function (message, type = 'info') {
    const container = document.getElementById('toastContainer');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    let icon = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>';
    if (type === 'success') icon = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="20 6 9 17 4 12"></polyline></svg>';
    if (type === 'error') icon = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>';

    toast.innerHTML = `<span style="display:flex; align-items:center;">${icon}</span> <span>${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transition = 'opacity 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, 4000);
  };

  // Tab Navigation Handling
  const tabs = document.querySelectorAll('.nav-tab');
  const views = document.querySelectorAll('.view-section');

  window.switchTab = function (targetViewId) {
    tabs.forEach(tab => {
      tab.classList.toggle('active', tab.getAttribute('data-view') === targetViewId);
    });
    views.forEach(view => {
      view.classList.toggle('active', view.id === targetViewId);
    });

    // Actions on specific tab activations
    if (targetViewId === 'live-view') {
      if (window.liveStream && window.liveStream.visualizer) {
        window.liveStream.visualizer.resizeCanvases();
        window.liveStream.visualizer.render();
      }
    } else if (targetViewId === 'patients-view') {
      if (window.patientsManager) window.patientsManager.loadPatients();
    } else if (targetViewId === 'history-view') {
      if (window.historyManager) window.historyManager.loadEvents();
    }
  };

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const viewId = tab.getAttribute('data-view');
      window.switchTab(viewId);
    });
  });

  // App Initialization
  const isAuthenticated = await window.authManager.checkInitialAuth();
  if (isAuthenticated) {
    await initializeApp();
  }

  window.addEventListener('auth:success', async () => {
    await initializeApp();
  });

  async function initializeApp() {
    window.liveStream.initVisualizer();
    await window.patientsManager.loadPatients();
    await window.historyManager.loadEvents();
    window.liveStream.connect();
    window.showToast('Secure session established with Edge Gateway', 'success');
  }
});
