/* =============================================
   NightVision AI — Main JavaScript
   =============================================
   Sections:
   1. Live Clock
   2. File Selection & Drag-Drop
   3. Upload & Processing (fetch + SSE)
   4. Progress UI Updates
   5. Output Display
   6. System Log
   7. Reset
============================================= */

/* ---- 1. Live Clock ---- */
/* Updates the HH:MM:SS clock in the header every second */
function updateClock() {
  const now = new Date();
  document.getElementById("live-clock").textContent = now
    .toTimeString()
    .slice(0, 8);
}
setInterval(updateClock, 1000);
updateClock(); // Run immediately so there's no blank on load

/* ---- 2. File Selection & Drag-Drop ---- */

// Allow user to drag a video file onto the upload zone
const zone = document.getElementById("upload-zone");

zone.addEventListener("dragover", (e) => {
  e.preventDefault(); // Necessary to allow drop
  zone.classList.add("drag-over"); // Highlight the zone
});

zone.addEventListener(
  "dragleave",
  () => zone.classList.remove("drag-over"), // Remove highlight when file leaves
);

zone.addEventListener("drop", (e) => {
  e.preventDefault();
  zone.classList.remove("drag-over");
  const files = e.dataTransfer.files;
  if (files[0]) handleFile(files[0]); // Process first dropped file
});

// Also handle regular click-to-browse selection
document.getElementById("file-input").addEventListener("change", (e) => {
  if (e.target.files[0]) handleFile(e.target.files[0]);
});

// Store the selected file globally so startProcessing() can access it
let selectedFile = null;

/**
 * handleFile — Called when a file is selected (drag or click)
 * Shows filename, file size, and makes the Enhance button visible
 */
function handleFile(file) {
  selectedFile = file;

  // Show the file name and size below the upload zone
  const nameEl = document.getElementById("file-name");
  nameEl.textContent = `▸ ${file.name}  (${(file.size / 1024 / 1024).toFixed(1)} MB)`;
  nameEl.style.display = "block";

  // Show the Enhance & Analyze button
  document.getElementById("process-btn").style.display = "block";

  addLog("ok", `File loaded: ${file.name}`);
}

/* ---- 3. Upload & Processing ---- */

/**
 * startProcessing — Called when user clicks "ENHANCE & ANALYZE"
 *
 * Two things happen simultaneously:
 *   A) SSE connection to /progress — receives live progress updates
 *   B) fetch POST to /upload — sends the video file, waits for result
 */
function startProcessing() {
  if (!selectedFile) return;

  const btn = document.getElementById("process-btn");
  btn.disabled = true; // Prevent double-click
  document.getElementById("progress-section").style.display = "block";
  document.getElementById("output-panel").style.display = "none";

  addLog("ok", "Processing started...");

  // --- A) SSE: Live progress streaming ---
  // Server-Sent Events = the server pushes updates to browser automatically
  // /progress endpoint on the Flask server keeps sending JSON every 0.5 seconds
  const es = new EventSource("/progress");

  es.onmessage = (e) => {
    const data = JSON.parse(e.data); // Each message is a JSON object
    updateProgress(data); // Update the progress bar UI

    // Close the SSE connection once processing finishes
    if (data.status === "done" || data.status === "error") {
      es.close();
    }
  };

  // --- B) fetch POST: Upload the video file ---
  // FormData packages the file for multipart upload
  const fd = new FormData();
  fd.append("video", selectedFile); // "video" matches request.files.get("video") in Flask

  fetch("/upload", { method: "POST", body: fd })
    .then((r) => r.json())
    .then((data) => {
      if (data.error) {
        addLog("alert", `Error: ${data.error}`);
        return;
      }
      addLog("ok", "Processing complete!");
      showOutput(data.video_url, data.stats); // Display the result video + stats
      btn.disabled = false;
    })
    .catch((err) => {
      addLog("alert", `Error: ${err}`);
      btn.disabled = false;
    });
}

/* ---- 4. Progress UI Updates ---- */

/**
 * updateProgress — Updates the progress bar, percentage, status text,
 * and pipeline stage chips based on data received from SSE
 *
 * data.status can be: "starting" | "processing" | "converting" | "done"
 * data.progress: 0–100
 * data.current_frame: current frame number being processed
 */
function updateProgress(data) {
  const pct = data.progress || 0;

  // Update the green progress bar width
  document.getElementById("progress-fill").style.width = pct + "%";
  document.getElementById("progress-pct").textContent = pct + "%";

  // Update the status text above the bar
  document.getElementById("status-text").textContent =
    data.status === "converting"
      ? "CONVERTING TO MP4..."
      : data.status === "done"
        ? "COMPLETE"
        : `PROCESSING FRAME ${data.current_frame || "..."}`;

  // Update the pipeline stage chips (ZERO-DCE, CLAHE, MOTION, RENDER, CONVERT)
  // Each chip turns "active" (cyan) while running, "done" (green) when complete
  if (data.status === "processing") {
    ["stage-dce", "stage-clahe", "stage-motion"].forEach(
      (id) => (document.getElementById(id).className = "stage-chip active"),
    );
  }

  if (data.status === "converting") {
    ["stage-dce", "stage-clahe", "stage-motion"].forEach(
      (id) => (document.getElementById(id).className = "stage-chip done"),
    );
    ["stage-render", "stage-convert"].forEach(
      (id) => (document.getElementById(id).className = "stage-chip active"),
    );
  }

  if (data.status === "done") {
    [
      "stage-dce",
      "stage-clahe",
      "stage-motion",
      "stage-render",
      "stage-convert",
    ].forEach(
      (id) => (document.getElementById(id).className = "stage-chip done"),
    );
  }
}

/* ---- 5. Output Display ---- */

/**
 * showOutput — Called after /upload returns successfully
 * Displays the enhanced video and populates the stats panel
 *
 * videoUrl: path to the processed MP4 (e.g. /static/output/output.mp4)
 * stats: object with brightness_gain, motion_percent, processing_time, etc.
 */
function showOutput(videoUrl, stats) {
  // Set the video player source
  const vid = document.getElementById("output-video");
  vid.src = videoUrl;

  // Set the download button href
  document.getElementById("download-btn").href = videoUrl;

  // Show the output panel
  document.getElementById("output-panel").style.display = "block";

  // Populate stats dashboard
  if (stats) {
    document.getElementById("stat-frames").textContent =
      stats.total_frames_processed || stats.total_frames || "—";

    document.getElementById("stat-motion").textContent =
      (stats.motion_percent || 0) + "%";

    document.getElementById("stat-lum-orig").textContent =
      stats.avg_brightness_orig || "—";

    document.getElementById("stat-lum-enh").textContent =
      stats.avg_brightness_enh || "—";

    const gain = stats.brightness_gain;
    document.getElementById("stat-gain").textContent = gain ? `+${gain}` : "—";

    document.getElementById("stat-time").textContent =
      stats.processing_time || "—";

    // Add results to the system log
    addLog("ok", `Brightness gain: +${stats.brightness_gain || 0} units`);
    addLog("ok", `Motion detected in ${stats.motion_percent || 0}% of frames`);
  }
}

/* ---- 6. System Log ---- */

/**
 * addLog — Adds a new timestamped entry to the system log panel
 * type: "ok" (green) | "alert" (red) | default (gray)
 */
function addLog(type, msg) {
  const log = document.getElementById("alert-log");

  const entry = document.createElement("div");
  entry.className = `log-entry ${type}`;

  const now = new Date().toTimeString().slice(0, 8);
  entry.innerHTML = `<span class="log-time">${now}</span><span class="log-msg">${msg}</span>`;

  log.prepend(entry); // Newest entries appear at the top

  // Keep max 20 log entries to avoid overflow
  if (log.children.length > 20) log.removeChild(log.lastChild);
}

/* ---- 7. Reset ---- */

/**
 * resetAll — Clears everything and returns to initial state
 * Called when user clicks "NEW FILE"
 */
function resetAll() {
  selectedFile = null;

  document.getElementById("file-name").style.display = "none";
  document.getElementById("process-btn").style.display = "none";
  document.getElementById("progress-section").style.display = "none";
  document.getElementById("output-panel").style.display = "none";
  document.getElementById("output-video").src = "";
  document.getElementById("progress-fill").style.width = "0%";

  // Reset all stage chips back to default (gray)
  [
    "stage-dce",
    "stage-clahe",
    "stage-motion",
    "stage-render",
    "stage-convert",
  ].forEach((id) => (document.getElementById(id).className = "stage-chip"));

  addLog("ok", "System reset. Ready for new input.");
}
