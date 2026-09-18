const VIDEO_EXT = /\.(mp4|mov|m4v|avi|mkv|webm|ts|mts|m2ts)$/i;

const fileInput = document.querySelector("#file-input");
const dropZone = document.querySelector("#drop-zone");
const form = document.querySelector("#upload-form");
const sampleBtn = document.querySelector("#sample-btn");
const sampleVideoBtn = document.querySelector("#sample-video-btn");
const calibrateBtn = document.querySelector("#calibrate-btn");
const clearCalibBtn = document.querySelector("#clear-calib-btn");
const detectBtn = document.querySelector("#detect-btn");
const message = document.querySelector("#message");
const canvas = document.querySelector("#view");
const ctx = canvas.getContext("2d");
const empty = document.querySelector("#empty");
const meta = document.querySelector("#meta");
const countEl = document.querySelector("#count");
const timingEl = document.querySelector("#timing");
const stampItem = document.querySelector("#stamp-item");
const stampEl = document.querySelector("#stamp");
const sizeEl = document.querySelector("#size");
const videoOptions = document.querySelector("#video-options");
const intervalSec = document.querySelector("#interval-sec");
const maxFrames = document.querySelector("#max-frames");
const cameraHeightInput = document.querySelector("#camera-height");
const fovHInput = document.querySelector("#fov-h");
const calibStatusEl = document.querySelector("#calib-status");
const progress = document.querySelector("#progress");
const progressBar = document.querySelector("#progress-bar");
const timeline = document.querySelector("#timeline");
const hitStrip = document.querySelector("#hit-strip");
const frameSlider = document.querySelector("#frame-slider");
const prevFrameBtn = document.querySelector("#prev-frame");
const nextFrameBtn = document.querySelector("#next-frame");
const playBtn = document.querySelector("#play-btn");
const personsOnly = document.querySelector("#persons-only");

const CALIB_STORAGE_KEY = "dashadas_calib_v1";

let selectedFile = null;
let videoJob = null;
let viewIndex = 0;
let drawToken = 0;
let playing = false;
let playHandle = 0;
let lastPlayTick = 0;
let calibration = null;

function isVideo(file) {
  return Boolean(file) && (file.type.startsWith("video/") || VIDEO_EXT.test(file.name));
}

function errorDetail(payload, fallback) {
  const detail = payload && payload.detail;
  if (!detail) return fallback;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((item) => item.msg || JSON.stringify(item)).join("; ");
  }
  return fallback;
}

function setMessage(text, isError = false) {
  message.textContent = text;
  message.classList.toggle("error", isError);
}

function setBusy(busy) {
  detectBtn.disabled = busy || !selectedFile;
  sampleBtn.disabled = busy;
  sampleVideoBtn.disabled = busy;
  calibrateBtn.disabled = busy || !selectedFile || !isVideo(selectedFile);
  clearCalibBtn.disabled = busy || !calibration;
}

function cameraHeightM() {
  const n = Number(cameraHeightInput.value);
  return Number.isFinite(n) && n > 0 ? n : 1.4;
}

function fovHDeg() {
  const n = Number(fovHInput.value);
  return Number.isFinite(n) && n > 0 ? n : 120;
}

function loadCalibration() {
  try {
    const raw = sessionStorage.getItem(CALIB_STORAGE_KEY);
    calibration = raw ? JSON.parse(raw) : null;
  } catch {
    calibration = null;
  }
  renderCalibrationStatus();
}

function saveCalibration(payload) {
  calibration = payload;
  if (payload) {
    sessionStorage.setItem(CALIB_STORAGE_KEY, JSON.stringify(payload));
  } else {
    sessionStorage.removeItem(CALIB_STORAGE_KEY);
  }
  renderCalibrationStatus();
  clearCalibBtn.disabled = !calibration;
}

function renderCalibrationStatus() {
  if (!calibration || !calibration.calibrated) {
    calibStatusEl.textContent = "Not calibrated";
    calibStatusEl.classList.remove("is-ready");
    return;
  }
  const vp = calibration.vanishing_point;
  calibStatusEl.textContent = `Calibrated (VP ${vp.x.toFixed(0)}, ${vp.y.toFixed(0)})`;
  calibStatusEl.classList.add("is-ready");
}

function scaledVanishingPoint(imageWidth, imageHeight) {
  if (!calibration || !calibration.calibrated) return null;
  const cw = calibration.image && calibration.image.width;
  const ch = calibration.image && calibration.image.height;
  if (!cw || !ch) return null;
  const aspectCalib = cw / ch;
  const aspectNow = imageWidth / imageHeight;
  if (Math.abs(aspectCalib - aspectNow) > 0.08) return null;
  return {
    x: calibration.vanishing_point.x * (imageWidth / cw),
    y: calibration.vanishing_point.y * (imageHeight / ch),
  };
}

/* Pinhole + flat road: Z ≈ h * fy / (v_contact - v_horizon). */
function estimateDistanceM(bbox, imageWidth, imageHeight) {
  const vp = scaledVanishingPoint(imageWidth, imageHeight);
  if (!vp) return null;
  const h = calibration.camera_height_m || cameraHeightM();
  const fov = ((calibration.fov_h_deg || fovHDeg()) * Math.PI) / 180;
  const fx = imageWidth / 2 / Math.tan(fov / 2);
  const fy = fx;
  const vContact = bbox[3];
  if (vContact <= vp.y + 1) return null;
  const z = (h * fy) / (vContact - vp.y);
  if (!Number.isFinite(z) || z <= 0 || z > 300) return null;
  return z;
}

function detectionLabel(det, imageWidth, imageHeight) {
  const cls = det.label || "object";
  const score = `${(det.score * 100).toFixed(0)}%`;
  const dist = estimateDistanceM(det.bbox, imageWidth, imageHeight);
  if (dist == null) return `${cls} ${score}`;
  return `${cls} ${score}  ${dist.toFixed(1)} m`;
}

function formatTime(sec) {
  const minutes = Math.floor(sec / 60);
  const seconds = (sec % 60).toFixed(1).padStart(4, "0");
  return `${minutes}:${seconds}`;
}

/* Model execute time from the API (inference_ms), not ffmpeg or letterbox. */
function formatFrameMs(ms) {
  const n = Number(ms);
  if (!Number.isFinite(n)) return "—";
  return `${n.toFixed(1)} ms`;
}

function setFrameMeta({ pedestrians, inferenceMs, width, height, timeSec }) {
  empty.classList.add("hidden");
  meta.classList.add("is-shown");
  countEl.textContent = String(pedestrians);
  timingEl.textContent = formatFrameMs(inferenceMs);
  sizeEl.textContent = `${width} x ${height}`;
  if (timeSec == null) {
    stampItem.hidden = true;
    stampEl.textContent = "—";
  } else {
    stampItem.hidden = false;
    stampEl.textContent = formatTime(timeSec);
  }
}

function drawVanishingPoint(image) {
  const vp = scaledVanishingPoint(image.naturalWidth, image.naturalHeight);
  if (!vp) return;
  const size = Math.max(10, Math.round(image.naturalWidth / 80));
  ctx.save();
  ctx.strokeStyle = "#5ad6ff";
  ctx.fillStyle = "#5ad6ff";
  ctx.lineWidth = Math.max(2, Math.round(image.naturalWidth / 500));
  ctx.beginPath();
  ctx.moveTo(vp.x - size, vp.y);
  ctx.lineTo(vp.x + size, vp.y);
  ctx.moveTo(vp.x, vp.y - size);
  ctx.lineTo(vp.x, vp.y + size);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(vp.x, vp.y, Math.max(3, size * 0.28), 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

/* HUD in image pixels so it scales with the canvas, not the CSS layout. */
function drawFrameTime(image, inferenceMs) {
  const text = formatFrameMs(inferenceMs);
  const caption = "FRAME TIME";
  const font = Math.max(18, Math.round(image.naturalWidth / 36));
  const small = Math.max(11, Math.round(font * 0.42));
  const padX = Math.max(10, Math.round(font * 0.5));
  const padY = Math.max(8, Math.round(font * 0.32));
  ctx.save();
  ctx.font = `650 ${font}px sans-serif`;
  ctx.textBaseline = "top";
  const textW = ctx.measureText(text).width;
  ctx.font = `600 ${small}px sans-serif`;
  const captionW = ctx.measureText(caption).width;
  const boxW = Math.max(textW, captionW) + padX * 2;
  const boxH = small + font + padY * 2 + 4;
  const x = Math.max(8, Math.round(image.naturalWidth * 0.015));
  const y = Math.max(8, Math.round(image.naturalHeight * 0.015));
  ctx.fillStyle = "rgba(18, 20, 23, 0.78)";
  ctx.fillRect(x, y, boxW, boxH);
  ctx.fillStyle = "#9aa7b4";
  ctx.fillText(caption, x + padX, y + padY);
  ctx.fillStyle = "#d6ff3f";
  ctx.font = `650 ${font}px sans-serif`;
  ctx.fillText(text, x + padX, y + padY + small + 2);
  ctx.restore();
}

function drawLanes(lanes) {
  if (!lanes || !lanes.polylines || !lanes.polylines.length) return;
  const line = Math.max(2, Math.round(canvas.width / 320));
  ctx.save();
  ctx.strokeStyle = "rgba(90, 214, 255, 0.92)";
  ctx.lineWidth = line;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  for (const poly of lanes.polylines) {
    if (!poly || poly.length < 2) continue;
    ctx.beginPath();
    ctx.moveTo(poly[0][0], poly[0][1]);
    for (let i = 1; i < poly.length; i += 1) {
      ctx.lineTo(poly[i][0], poly[i][1]);
    }
    ctx.stroke();
  }
  ctx.restore();
}

function drawDetections(image, detections, inferenceMs, lanes) {
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  ctx.drawImage(image, 0, 0);
  drawLanes(lanes);
  const line = Math.max(2, Math.round(image.naturalWidth / 400));
  ctx.lineWidth = line;
  ctx.font = `${Math.max(14, Math.round(image.naturalWidth / 50))}px sans-serif`;
  ctx.textBaseline = "top";

  for (const det of detections) {
    const [x1, y1, x2, y2] = det.bbox;
    const width = x2 - x1;
    const height = y2 - y1;
    const label = detectionLabel(det, image.naturalWidth, image.naturalHeight);
    ctx.strokeStyle = "#d6ff3f";
    ctx.strokeRect(x1, y1, width, height);
    const textWidth = ctx.measureText(label).width;
    ctx.fillStyle = "rgba(18, 20, 23, 0.85)";
    ctx.fillRect(x1, Math.max(0, y1 - 22), textWidth + 10, 22);
    ctx.fillStyle = "#d6ff3f";
    ctx.fillText(label, x1 + 5, Math.max(0, y1 - 20));
  }
  drawVanishingPoint(image);
  drawFrameTime(image, inferenceMs);
}

function showStill(imageUrl, payload, options = {}) {
  stopPlayback();
  playBtn.disabled = true;
  timeline.classList.remove("is-shown");
  progress.hidden = true;
  const image = new Image();
  image.onload = () => {
    setFrameMeta({
      pedestrians: payload.detections.length,
      inferenceMs: payload.inference_ms,
      width: payload.image.width,
      height: payload.image.height,
    });
    drawDetections(image, payload.detections, payload.inference_ms, payload.lanes);
    if (options.silent) return;
    if (payload.detections.length === 0) {
      setMessage("No people or vehicles were detected. Try another image.");
    } else {
      const withDist = calibration && calibration.calibrated ? " Distances use the session calibration." : "";
      setMessage(`Detected ${payload.detections.length} object(s).${withDist}`);
    }
  };
  image.onerror = () => setMessage("Failed to display the image.", true);
  image.src = imageUrl;
}

function visibleFrames() {
  if (!videoJob) return [];
  if (personsOnly.checked) {
    return videoJob.frames.filter((frame) => frame.detections.length > 0);
  }
  return videoJob.frames;
}

function currentFrame() {
  const frames = visibleFrames();
  if (!frames.length) return null;
  viewIndex = Math.max(0, Math.min(viewIndex, frames.length - 1));
  return frames[viewIndex];
}

function renderHitStrip() {
  const frames = videoJob ? videoJob.frames : [];
  hitStrip.innerHTML = frames
    .map((frame, index) => {
      const active = currentFrame() && currentFrame().index === frame.index ? " active" : "";
      const on = frame.detections.length ? " on" : "";
      return `<button type="button" class="hit${on}${active}" data-index="${index}" title="${formatTime(frame.time_sec)}"></button>`;
    })
    .join("");
}

function showVideoFrame() {
  const frame = currentFrame();
  const frames = visibleFrames();
  if (!frame) {
    setMessage("No frames with pedestrians. Turn off the filter.");
    return;
  }
  timeline.classList.add("is-shown");
  frameSlider.max = String(Math.max(frames.length - 1, 0));
  frameSlider.value = String(viewIndex);
  renderHitStrip();
  const token = ++drawToken;
  const image = new Image();
  image.onload = () => {
    if (token !== drawToken) return;
    setFrameMeta({
      pedestrians: frame.detections.length,
      inferenceMs: frame.inference_ms,
      width: frame.image.width,
      height: frame.image.height,
      timeSec: frame.time_sec,
    });
    drawDetections(image, frame.detections, frame.inference_ms, frame.lanes);
  };
  image.onerror = () => setMessage("Failed to display the frame.", true);
  image.src = `${frame.url}?t=${Date.now()}`;
  playBtn.disabled = false;
}

function playbackDelayMs() {
  const sec = Number(videoJob && videoJob.interval_sec);
  if (sec > 0) return sec * 1000;
  return 1000 / 30;
}

function stopPlayback() {
  playing = false;
  if (playHandle) {
    cancelAnimationFrame(playHandle);
    playHandle = 0;
  }
  playBtn.textContent = "Play";
  playBtn.classList.remove("is-playing");
}

function setPlaying(on) {
  if (on && !visibleFrames().length) return;
  if (on === playing) return;
  playing = on;
  playBtn.textContent = on ? "Pause" : "Play";
  playBtn.classList.toggle("is-playing", on);
  if (on) {
    lastPlayTick = performance.now();
    playHandle = requestAnimationFrame(playTick);
  } else if (playHandle) {
    cancelAnimationFrame(playHandle);
    playHandle = 0;
  }
}

function playTick(now) {
  if (!playing) return;
  if (now - lastPlayTick >= playbackDelayMs()) {
    lastPlayTick = now;
    const frames = visibleFrames();
    if (!frames.length) {
      setPlaying(false);
      return;
    }
    if (viewIndex >= frames.length - 1) {
      setPlaying(false);
      return;
    }
    viewIndex += 1;
    showVideoFrame();
  }
  playHandle = requestAnimationFrame(playTick);
}

function updateVideoProgress(job) {
  const total = job.total || Number(maxFrames.value) || 1;
  const ratio = job.total ? job.processed / job.total : 0;
  progress.hidden = false;
  progressBar.style.width = `${Math.round(ratio * 100)}%`;
  if (job.status === "running" || job.status === "queued") {
    setMessage(
      `Processing video… ${job.processed} / ${total} frames (${job.person_frames} with detections)`
    );
  }
}

async function pollVideoJob(jobId) {
  stopPlayback();
  while (true) {
    const response = await fetch(`/api/jobs/${jobId}`);
    const job = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(errorDetail(job, "Failed to fetch the job"));
    }
    videoJob = job;
    updateVideoProgress(job);
    if (job.status === "done") {
      progressBar.style.width = "100%";
      const summary = [
        `${job.frames.length} frames processed`,
        `${job.person_frames} with detections`,
        `${job.person_detections} detections`,
      ];
      if (job.video && job.video.duration_sec) {
        summary.unshift(`Duration ${formatTime(job.video.duration_sec)}`);
      }
      setMessage(summary.join(" / "));
      viewIndex = 0;
      showVideoFrame();
      return job;
    }
    if (job.status === "error") {
      throw new Error(job.error || "Video processing failed");
    }
    if (job.frames.length) {
      viewIndex = personsOnly.checked
        ? Math.max(visibleFrames().length - 1, 0)
        : job.frames.length - 1;
      showVideoFrame();
    }
    await new Promise((resolve) => setTimeout(resolve, 400));
  }
}

async function detectImage(file) {
  const body = new FormData();
  body.append("file", file);
  setMessage("Detecting…");
  const response = await fetch("/api/detect", { method: "POST", body });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(errorDetail(payload, "Detection failed"));
  }
  showStill(URL.createObjectURL(file), payload);
}

async function detectVideo(file) {
  const body = new FormData();
  body.append("file", file);
  body.append("interval_sec", intervalSec.value);
  body.append("max_frames", maxFrames.value);
  setMessage("Uploading video…");
  progress.hidden = false;
  progressBar.style.width = "2%";
  const response = await fetch("/api/detect-video", { method: "POST", body });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(errorDetail(payload, "Could not accept the video"));
  }
  await pollVideoJob(payload.id);
}

async function detectSampleImage() {
  setMessage("Detecting the sample image…");
  const response = await fetch("/api/detect-sample", { method: "POST" });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(errorDetail(payload, "Sample detection failed"));
  }
  showStill(payload.sample, payload);
}

async function detectSampleVideo() {
  const body = new FormData();
  body.append("interval_sec", String(1 / 30));
  body.append("max_frames", "300");
  setMessage("Fetching a pedestrian demo clip, then detecting…");
  progress.hidden = false;
  progressBar.style.width = "2%";
  const response = await fetch("/api/detect-video-sample", { method: "POST", body });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(errorDetail(payload, "Could not accept the sample video"));
  }
  await pollVideoJob(payload.id);
  setPlaying(true);
}

function rememberFile(file) {
  selectedFile = file;
  detectBtn.disabled = !file;
  calibrateBtn.disabled = !file || !isVideo(file);
  videoOptions.classList.toggle("is-shown", isVideo(file));
  if (file) {
    setMessage(
      isVideo(file)
        ? `${file.name} selected. Use Calibrate for lanes, or Detect for objects.`
        : `${file.name} selected. Press Detect.`
    );
  }
}

async function calibrateVideo(file) {
  const body = new FormData();
  body.append("file", file);
  body.append("interval_sec", "0.2");
  body.append("max_frames", "90");
  setMessage("Calibrating from highway-style lane markings (no object detect)…");
  progress.hidden = false;
  progressBar.style.width = "15%";
  const response = await fetch("/api/calibrate-video", { method: "POST", body });
  progressBar.style.width = "90%";
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(errorDetail(payload, "Calibration failed"));
  }
  if (!payload.calibrated) {
    saveCalibration(null);
    setMessage(payload.message || "Calibration did not complete.", true);
    progress.hidden = true;
    return;
  }
  saveCalibration({
    calibrated: true,
    vanishing_point: payload.vanishing_point,
    image: payload.image,
    camera_height_m: cameraHeightM(),
    fov_h_deg: fovHDeg(),
    good_frames: payload.good_frames,
    stability: payload.stability,
  });
  if (payload.preview_jpeg_base64) {
    showStill(
      `data:image/jpeg;base64,${payload.preview_jpeg_base64}`,
      {
        detections: [],
        inference_ms: 0,
        image: payload.image,
      },
      { silent: true }
    );
  }
  setMessage(payload.message || "Calibration complete.");
  progress.hidden = true;
}

fileInput.addEventListener("change", () => {
  rememberFile(fileInput.files[0] || null);
});

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragover");
  });
});

dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (!file) return;
  rememberFile(file);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!selectedFile) return;
  setBusy(true);
  try {
    if (isVideo(selectedFile)) {
      await detectVideo(selectedFile);
    } else {
      await detectImage(selectedFile);
    }
  } catch (error) {
    setMessage(error.message, true);
    progress.hidden = true;
  } finally {
    setBusy(false);
  }
});

sampleBtn.addEventListener("click", async () => {
  setBusy(true);
  try {
    await detectSampleImage();
  } catch (error) {
    setMessage(error.message, true);
  } finally {
    setBusy(false);
  }
});

sampleVideoBtn.addEventListener("click", async () => {
  setBusy(true);
  try {
    await detectSampleVideo();
  } catch (error) {
    setMessage(error.message, true);
    progress.hidden = true;
  } finally {
    setBusy(false);
  }
});

calibrateBtn.addEventListener("click", async () => {
  if (!selectedFile || !isVideo(selectedFile)) {
    setMessage("Select a video with clear highway-style lane markings to calibrate.", true);
    return;
  }
  setBusy(true);
  try {
    await calibrateVideo(selectedFile);
  } catch (error) {
    setMessage(error.message, true);
    progress.hidden = true;
  } finally {
    setBusy(false);
  }
});

clearCalibBtn.addEventListener("click", () => {
  saveCalibration(null);
  setMessage("Calibration cleared for this session.");
});

cameraHeightInput.addEventListener("change", () => {
  if (!calibration || !calibration.calibrated) return;
  calibration.camera_height_m = cameraHeightM();
  sessionStorage.setItem(CALIB_STORAGE_KEY, JSON.stringify(calibration));
  if (videoJob) showVideoFrame();
});

fovHInput.addEventListener("change", () => {
  if (!calibration || !calibration.calibrated) return;
  calibration.fov_h_deg = fovHDeg();
  sessionStorage.setItem(CALIB_STORAGE_KEY, JSON.stringify(calibration));
  if (videoJob) showVideoFrame();
});

frameSlider.addEventListener("input", () => {
  viewIndex = Number(frameSlider.value);
  showVideoFrame();
});

playBtn.addEventListener("click", () => {
  setPlaying(!playing);
});

prevFrameBtn.addEventListener("click", () => {
  viewIndex -= 1;
  showVideoFrame();
});

nextFrameBtn.addEventListener("click", () => {
  viewIndex += 1;
  showVideoFrame();
});

personsOnly.addEventListener("change", () => {
  viewIndex = 0;
  showVideoFrame();
});

hitStrip.addEventListener("click", (event) => {
  const button = event.target.closest("[data-index]");
  if (!button || !videoJob) return;
  const absoluteIndex = Number(button.dataset.index);
  const frames = visibleFrames();
  const mapped = frames.findIndex((frame) => frame.index === absoluteIndex);
  if (mapped < 0) return;
  viewIndex = mapped;
  showVideoFrame();
});

document.addEventListener("keydown", (event) => {
  if (!timeline.classList.contains("is-shown")) return;
  if (event.key === " ") {
    event.preventDefault();
    setPlaying(!playing);
    return;
  }
  if (event.key === "ArrowLeft") {
    viewIndex -= 1;
    showVideoFrame();
  }
  if (event.key === "ArrowRight") {
    viewIndex += 1;
    showVideoFrame();
  }
});

const deviceSelect = document.querySelector("#device-select");

function deviceLabel(name, payload) {
  if (name === "cuda") {
    const gpu = payload.gpu_name ? payload.gpu_name.replace(/^NVIDIA GeForce /, "") : "GPU";
    return `cuda (${gpu})`;
  }
  if (name === "qnn") {
    return "qnn (CPU emu)";
  }
  if (name === "yolopv2") {
    const gpu = payload.gpu_name ? payload.gpu_name.replace(/^NVIDIA GeForce /, "") : null;
    return gpu ? `yolopv2 (${gpu})` : "yolopv2";
  }
  return name;
}

function renderDeviceSelect(payload) {
  const available = payload.devices && payload.devices.length ? payload.devices : [payload.device || "cpu"];
  deviceSelect.innerHTML = available
    .map((name) => `<option value="${name}">${deviceLabel(name, payload)}</option>`)
    .join("");
  deviceSelect.value = payload.device || "cpu";
  deviceSelect.disabled = available.length < 2;
  if (payload.model) {
    const modelEl = document.querySelector("#model");
    if (modelEl) modelEl.textContent = payload.model;
  }
  renderDetectClasses(payload);
}

function renderDetectClasses(payload) {
  const el = document.querySelector("#detect-classes");
  if (!el) return;
  const byDevice = payload.detect_classes_by_device || {};
  const current = payload.device || deviceSelect.value || "cpu";
  const list =
    (payload.detect_classes && payload.detect_classes.length && payload.detect_classes) ||
    byDevice[current] ||
    [];
  el.textContent = list.length ? list.join(", ") : "—";
  el.title = list.length ? `Detectable on ${current}: ${list.join(", ")}` : "";
}

async function loadHealth() {
  try {
    const response = await fetch("/api/health");
    const payload = await response.json();
    document.querySelector("#ready").textContent = payload.ready ? "Ready" : "Starting";
    document.querySelector("#model").textContent = payload.model;
    renderDeviceSelect(payload);
  } catch {
    document.querySelector("#ready").textContent = "Offline";
  }
}

deviceSelect.addEventListener("change", async () => {
  const requested = deviceSelect.value;
  deviceSelect.disabled = true;
  setMessage(`Switching device to ${requested}…`);
  try {
    const response = await fetch("/api/device", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ device: requested }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(errorDetail(payload, "Could not switch device"));
    }
    renderDeviceSelect(payload);
    setMessage(`Using ${deviceLabel(payload.device, payload)}.`);
  } catch (error) {
    setMessage(error.message, true);
    await loadHealth();
  }
});

loadCalibration();
loadHealth();
