const VIDEO_EXT = /\.(mp4|mov|m4v|avi|mkv|webm|ts|mts|m2ts)$/i;

const fileInput = document.querySelector("#file-input");
const dropZone = document.querySelector("#drop-zone");
const form = document.querySelector("#upload-form");
const sampleBtn = document.querySelector("#sample-btn");
const sampleVideoBtn = document.querySelector("#sample-video-btn");
const detectBtn = document.querySelector("#detect-btn");
const message = document.querySelector("#message");
const canvas = document.querySelector("#view");
const ctx = canvas.getContext("2d");
const empty = document.querySelector("#empty");
const meta = document.querySelector("#meta");
const countEl = document.querySelector("#count");
const timingEl = document.querySelector("#timing");
const sizeEl = document.querySelector("#size");
const videoOptions = document.querySelector("#video-options");
const intervalSec = document.querySelector("#interval-sec");
const maxFrames = document.querySelector("#max-frames");
const progress = document.querySelector("#progress");
const progressBar = document.querySelector("#progress-bar");
const timeline = document.querySelector("#timeline");
const hitStrip = document.querySelector("#hit-strip");
const frameSlider = document.querySelector("#frame-slider");
const prevFrameBtn = document.querySelector("#prev-frame");
const nextFrameBtn = document.querySelector("#next-frame");
const playBtn = document.querySelector("#play-btn");
const personsOnly = document.querySelector("#persons-only");

let selectedFile = null;
let videoJob = null;
let viewIndex = 0;
let drawToken = 0;
let playing = false;
let playHandle = 0;
let lastPlayTick = 0;

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
}

function formatTime(sec) {
  const minutes = Math.floor(sec / 60);
  const seconds = (sec % 60).toFixed(1).padStart(4, "0");
  return `${minutes}:${seconds}`;
}

function drawDetections(image, detections) {
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  ctx.drawImage(image, 0, 0);
  const line = Math.max(2, Math.round(image.naturalWidth / 400));
  ctx.lineWidth = line;
  ctx.font = `${Math.max(14, Math.round(image.naturalWidth / 50))}px sans-serif`;
  ctx.textBaseline = "top";

  for (const det of detections) {
    const [x1, y1, x2, y2] = det.bbox;
    const width = x2 - x1;
    const height = y2 - y1;
    const label = `person ${(det.score * 100).toFixed(0)}%`;
    ctx.strokeStyle = "#d6ff3f";
    ctx.strokeRect(x1, y1, width, height);
    const textWidth = ctx.measureText(label).width;
    ctx.fillStyle = "rgba(18, 20, 23, 0.85)";
    ctx.fillRect(x1, Math.max(0, y1 - 22), textWidth + 10, 22);
    ctx.fillStyle = "#d6ff3f";
    ctx.fillText(label, x1 + 5, Math.max(0, y1 - 20));
  }
}

function showStill(imageUrl, payload) {
  stopPlayback();
  playBtn.disabled = true;
  timeline.classList.remove("is-shown");
  progress.hidden = true;
  const image = new Image();
  image.onload = () => {
    empty.classList.add("hidden");
    meta.classList.add("is-shown");
    countEl.textContent = `Pedestrians ${payload.detections.length}`;
    timingEl.textContent = `${payload.inference_ms} ms`;
    sizeEl.textContent = `${payload.image.width} x ${payload.image.height}`;
    drawDetections(image, payload.detections);
    if (payload.detections.length === 0) {
      setMessage("No pedestrians were detected. Try another image.");
    } else {
      setMessage(`Detected ${payload.detections.length} pedestrian(s).`);
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
    empty.classList.add("hidden");
    meta.classList.add("is-shown");
    countEl.textContent = `Pedestrians ${frame.detections.length}`;
    timingEl.textContent = `${formatTime(frame.time_sec)} / ${frame.inference_ms} ms`;
    sizeEl.textContent = `${frame.image.width} x ${frame.image.height}`;
    drawDetections(image, frame.detections);
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
    viewIndex = viewIndex >= frames.length - 1 ? 0 : viewIndex + 1;
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
      `Processing video… ${job.processed} / ${total} frames (${job.person_frames} with people)`
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
        `${job.person_frames} with people`,
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
  videoOptions.classList.toggle("is-shown", isVideo(file));
  if (file) {
    setMessage(
      isVideo(file)
        ? `${file.name} selected. Check the interval, then press Detect.`
        : `${file.name} selected. Press Detect.`
    );
  }
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
  return name;
}

function renderDeviceSelect(payload) {
  const available = payload.devices && payload.devices.length ? payload.devices : [payload.device || "cpu"];
  deviceSelect.innerHTML = available
    .map((name) => `<option value="${name}">${deviceLabel(name, payload)}</option>`)
    .join("");
  deviceSelect.value = payload.device || "cpu";
  deviceSelect.disabled = available.length < 2;
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

loadHealth();
