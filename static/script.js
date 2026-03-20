/* ── Global state ── */
let currentFile     = null;
let activeGalleryImg = null;

/* ═══════════════════════════════════
   1. HAMBURGER / MOBILE NAV
   ═══════════════════════════════════ */
const hamburger = document.getElementById('hamburger');
const mobileNav = document.getElementById('mobileNav');

hamburger.addEventListener('click', () => {
    hamburger.classList.toggle('open');
    mobileNav.classList.toggle('open');
});

// Close mobile nav on outside click
document.addEventListener('click', (e) => {
    if (!e.target.closest('#site-header')) closeNav();
});

function closeNav() {
    hamburger.classList.remove('open');
    mobileNav.classList.remove('open');
}

/* ═══════════════════════════════════
   2. THEME TOGGLE
   ═══════════════════════════════════ */
const themeToggle = document.getElementById('themeToggle');
themeToggle.addEventListener('click', () => {
    document.body.classList.toggle('light-mode');
    themeToggle.textContent = document.body.classList.contains('light-mode') ? '☀️' : '🌙';
});

/* ═══════════════════════════════════
   3. IMAGE HANDLING (Drop Zone)
   ═══════════════════════════════════ */
const dropZone  = document.getElementById('dropZone');
const fileInput = document.getElementById('imageUpload');
const preview   = document.getElementById('preview');
const previewHint = document.getElementById('previewHint');

/* Click / keyboard to open file picker */
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
});

/* Drag events */
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(evt => {
    dropZone.addEventListener(evt, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.toggle('drag-over', evt === 'dragenter' || evt === 'dragover');
    });
});

dropZone.addEventListener('drop', (e) => {
    const file = e.dataTransfer.files?.[0];
    if (file) handleImageFile(file);
});

fileInput.addEventListener('change', (e) => {
    const file = e.target.files?.[0];
    if (file) handleImageFile(file);
    // Reset so the same file can be re-picked
    fileInput.value = '';
});

function handleImageFile(file) {
    if (!file.type.startsWith('image/')) {
        return showError('⚠️ Please upload a valid image file (JPG, PNG, WEBP, etc.)');
    }
    currentFile = file;

    const reader = new FileReader();
    reader.onload = (ev) => {
        preview.src = ev.target.result;
        preview.classList.add('visible');
        if (previewHint) previewHint.style.display = 'none';
        document.getElementById('result').innerHTML = '';
        dropZone.querySelector('.drop-main').textContent = `✅ ${file.name}`;
        dropZone.querySelector('.drop-sub').textContent  = 'Click to change image';
    };
    reader.readAsDataURL(file);
}

/* ═══════════════════════════════════
   4. DETECTION LOGIC
   ═══════════════════════════════════ */
document.getElementById('detectBtn').addEventListener('click', async () => {
    if (!currentFile) {
        return showError('⚠️ Please upload an image first!');
    }

    const btn       = document.getElementById('detectBtn');
    const resultDiv = document.getElementById('result');
    const formData  = new FormData();
    formData.append('file', currentFile);

    try {
        btn.innerHTML  = '<span class="spinner"></span> Analyzing…';
        btn.disabled   = true;
        resultDiv.innerHTML = '<p class="analyzing-text">🔬 Running model inference…</p>';

        const res = await fetch('/predict', { method: 'POST', body: formData });

        if (!res.ok) {
            const txt = await res.text().catch(() => res.statusText);
            throw new Error(`Server error ${res.status}: ${txt}`);
        }

        const data        = await res.json();
        const isReal      = data.label === 'Real';
        const conf        = Math.round(data.confidence);
        const barColor    = isReal ? '#22c55e' : '#ef4444';

        resultDiv.innerHTML = `
            <div class="result-card ${isReal ? 'result-real' : 'result-fake'}">
                <div class="result-icon">${isReal ? '✅' : '❌'}</div>
                <h3 class="result-label">${data.label}</h3>
                <p class="result-desc">${
                    isReal
                        ? 'This image appears to be <strong>authentic</strong>.'
                        : 'This image shows signs of <strong>AI manipulation</strong>.'
                }</p>
                <div class="confidence-bar-wrap">
                    <label>Model confidence: <strong>${conf}%</strong></label>
                    <div class="confidence-bar-bg">
                        <div class="confidence-bar-fill"
                             style="width:${conf}%; background:${barColor}">
                        </div>
                    </div>
                </div>
            </div>`;

    } catch (err) {
        console.error('API Error:', err);
        showError('❌ Could not reach the Python backend. Make sure the server is running on port 8000.');
    } finally {
        btn.innerHTML = '<i class="fas fa-search"></i> Analyze Image';
        btn.disabled  = false;
    }
});

/* ═══════════════════════════════════
   5. GALLERY DOUBLE-CLICK → DROP ZONE
   ═══════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.sample-card img').forEach(img => {
        img.style.cursor = 'zoom-in';

        img.addEventListener('dblclick', async () => {
            try {
                /* Highlight ring */
                if (activeGalleryImg) activeGalleryImg.classList.remove('gallery-selected');
                img.classList.add('gallery-selected');
                activeGalleryImg = img;

                /* Fetch → File */
                const res      = await fetch(img.src);
                const blob     = await res.blob();
                const filename = img.src.split('/').pop();
                const file     = new File([blob], filename, { type: blob.type || 'image/jpeg' });

                handleImageFile(file);

                /* Smooth-scroll to Detection section */
                document.getElementById('detection')
                    .scrollIntoView({ behavior: 'smooth', block: 'start' });

            } catch (err) {
                console.error('Gallery load error:', err);
                showError('Failed to load gallery image. Try uploading manually.');
            }
        });
    });
});

/* ═══════════════════════════════════
   HELPERS
   ═══════════════════════════════════ */
function showError(msg) {
    document.getElementById('result').innerHTML =
        `<div class="result-card result-error"><p>${msg}</p></div>`;
}