const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const statusEl = document.getElementById("status");
const uploadBtn = document.getElementById("uploadBtn");
const clearBtn = document.getElementById("clearBtn");
const modeSelect = document.getElementById("modeSelect");
const anatomySelect = document.getElementById("anatomySelect");
const windowSelect = document.getElementById("windowSelect");
const prioritySelect = document.getElementById("prioritySelect");
const reviewerInput = document.getElementById("reviewerInput");
const noteInput = document.getElementById("noteInput");
const fileChip = document.getElementById("fileChip");
const previewWrap = document.getElementById("previewWrap");
const previewImg = document.getElementById("previewImg");
const progress = document.getElementById("progress");
const dzTitle = document.getElementById("dzTitle");
const consoleCard = document.getElementById("consoleCard");
const stages = [...document.querySelectorAll(".step")];
const startScpBtn = document.getElementById("startScpBtn");

let selectedFile = null;
let stageTimer = null;
let previewUrl = null;

function mir3d(fn, ...args) {
  if (window.MIR3D && typeof window.MIR3D[fn] === "function") {
    window.MIR3D[fn](...args);
  }
}

function setStatus(msg) {
  statusEl.textContent = msg || "";
}

function resetStages() {
  stages.forEach((s) => s.classList.remove("active", "done"));
}

function runStageAnimation() {
  resetStages();
  const order = ["ingest", "preprocess", "inference", "report", "review"];
  let i = 0;
  clearInterval(stageTimer);
  mir3d("setScanning", true);
  mir3d("setIntensity", 1.6);
  stageTimer = setInterval(() => {
    stages.forEach((s) => s.classList.remove("active"));
    if (i > 0) {
      const prev = document.querySelector(`.step[data-stage="${order[i - 1]}"]`);
      if (prev) prev.classList.add("done");
    }
    if (i < order.length) {
      const cur = document.querySelector(`.step[data-stage="${order[i]}"]`);
      if (cur) cur.classList.add("active");
      mir3d("pulse");
      i += 1;
    } else {
      clearInterval(stageTimer);
    }
  }, 420);
}

function clearSelection() {
  selectedFile = null;
  fileInput.value = "";
  uploadBtn.disabled = true;
  fileChip.classList.remove("show");
  fileChip.textContent = "";
  previewWrap.classList.remove("show");
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  previewImg.removeAttribute("src");
  dzTitle.textContent = "Drop imaging study";
  setStatus("");
  progress.classList.remove("show");
  resetStages();
  mir3d("setScanning", false);
  mir3d("setIntensity", 1);
}

function pickFile(file) {
  if (!file) return;
  selectedFile = file;
  uploadBtn.disabled = false;
  fileChip.classList.add("show");
  fileChip.textContent = `${file.name} · ${(file.size / 1024).toFixed(1)} KB`;
  dzTitle.textContent = "Study loaded";
  setStatus("Ready — run the pipeline.");
  mir3d("pulse");
  mir3d("setIntensity", 1.25);

  if (previewUrl) URL.revokeObjectURL(previewUrl);
  const isImage = /^image\/(png|jpeg|jpg)$/i.test(file.type) || /\.(png|jpe?g)$/i.test(file.name);
  if (isImage) {
    previewUrl = URL.createObjectURL(file);
    previewImg.src = previewUrl;
    previewWrap.classList.add("show");
  } else {
    previewWrap.classList.remove("show");
    previewImg.removeAttribute("src");
  }
}

// Parallax tilt on glass console
if (consoleCard) {
  window.addEventListener(
    "pointermove",
    (e) => {
      const nx = (e.clientX / window.innerWidth - 0.5) * 2;
      const ny = (e.clientY / window.innerHeight - 0.5) * 2;
      consoleCard.style.transform = `perspective(1200px) rotateY(${nx * 4}deg) rotateX(${-ny * 3}deg)`;
    },
    { passive: true }
  );
}

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fileInput.click();
  }
});
fileInput.addEventListener("change", (e) => pickFile(e.target.files[0]));
clearBtn.addEventListener("click", clearSelection);

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag");
    mir3d("setScanning", true);
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag");
    if (evt === "dragleave") mir3d("setScanning", false);
  })
);
dropzone.addEventListener("drop", (e) => pickFile(e.dataTransfer.files[0]));

uploadBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  uploadBtn.disabled = true;
  progress.classList.add("show");
  runStageAnimation();
  setStatus("Running pipeline… then opening Output page");

  const formData = new FormData();
  formData.append("file", selectedFile);
  formData.append("mode", modeSelect.value);
  formData.append("scan_type", anatomySelect ? anatomySelect.value : "auto");
  formData.append("window", windowSelect.value);
  formData.append("priority", prioritySelect.value);
  formData.append("reviewer", reviewerInput.value || "anonymous");
  formData.append("clinical_note", noteInput.value || "");

  try {
    const res = await fetch("/api/scans", { method: "POST", body: formData });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Upload failed" }));
      throw new Error(typeof err.detail === "string" ? err.detail : "Upload failed");
    }
    const report = await res.json();
    stages.forEach((s) => {
      s.classList.remove("active");
      s.classList.add("done");
    });
    mir3d("pulse");
    setStatus("Opening Output…");
    window.location.href = `/report/${report.scan_id}`;
  } catch (err) {
    clearInterval(stageTimer);
    resetStages();
    progress.classList.remove("show");
    mir3d("setScanning", false);
    setStatus(err.message);
    uploadBtn.disabled = false;
  }
});

if (startScpBtn) {
  startScpBtn.addEventListener("click", async () => {
    startScpBtn.disabled = true;
    startScpBtn.textContent = "Starting…";
    try {
      const res = await fetch("/api/pacs/scp/start", { method: "POST" });
      const data = await res.json();
      const statusText = document.getElementById("scpStatusText");
      const received = document.getElementById("scpReceived");
      if (statusText) statusText.textContent = data.running ? "ON" : "OFF";
      if (received) received.textContent = data.received;
      startScpBtn.textContent = data.running ? "PACS live" : "Start PACS";
      mir3d("pulse");
      if (!data.running) startScpBtn.disabled = false;
    } catch (err) {
      startScpBtn.textContent = "Start PACS";
      startScpBtn.disabled = false;
      setStatus(err.message);
    }
  });
}
