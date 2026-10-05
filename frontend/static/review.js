const scanIdFromPath = window.location.pathname.split("/").pop();
const reviewer =
  (window.localStorage.getItem("mir.reviewer") || "radiologist-demo").trim() ||
  "radiologist-demo";

if (window.MIR3D) {
  window.MIR3D.setIntensity(0.9);
  window.MIR3D.pulse();
}

document.querySelectorAll(".review-actions").forEach((group) => {
  const acceptBtn = group.querySelector(".btn-accept");
  const rejectBtn = group.querySelector(".btn-reject");
  const scanId = group.dataset.scanId;
  const findingId = group.dataset.findingId;

  async function save(decision) {
    const res = await fetch(`/api/scans/${scanId}/findings/${findingId}/review`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ decision, reviewer }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Review failed" }));
      alert(err.detail || "Review failed");
      return;
    }
    acceptBtn.classList.toggle("active", decision === "accept");
    rejectBtn.classList.toggle("active", decision === "reject");
    if (window.MIR3D) window.MIR3D.pulse();
  }

  acceptBtn.addEventListener("click", () => save("accept"));
  rejectBtn.addEventListener("click", () => save("reject"));
});

const signoffBtn = document.getElementById("signoffBtn");
if (signoffBtn) {
  signoffBtn.addEventListener("click", async () => {
    const res = await fetch(`/api/scans/${scanIdFromPath}/signoff`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status: "signed", reviewer }),
    });
    if (!res.ok) {
      alert("Sign-off failed");
      return;
    }
    const data = await res.json();
    const el = document.getElementById("signoffStatus");
    if (el) el.textContent = data.status;
    signoffBtn.disabled = true;
    signoffBtn.textContent = "Signed";
    if (window.MIR3D) window.MIR3D.pulse();
  });
}

const overlayImg = document.getElementById("overlayImg");
const originalImg = document.getElementById("originalImg");
const overlayOpacity = document.getElementById("overlayOpacity");
const invertBtn = document.getElementById("invertBtn");
const toggleOverlayBtn = document.getElementById("toggleOverlayBtn");
let inverted = false;
let overlayHidden = false;

if (overlayOpacity && overlayImg) {
  overlayOpacity.addEventListener("input", () => {
    overlayImg.style.opacity = String(Number(overlayOpacity.value) / 100);
  });
}

if (invertBtn) {
  invertBtn.addEventListener("click", () => {
    inverted = !inverted;
    const filter = inverted ? "invert(1) hue-rotate(180deg)" : "none";
    if (originalImg) originalImg.style.filter = filter;
    if (overlayImg) overlayImg.style.filter = filter;
    invertBtn.textContent = inverted ? "Restore" : "Invert";
  });
}

if (toggleOverlayBtn) {
  toggleOverlayBtn.addEventListener("click", () => {
    overlayHidden = !overlayHidden;
    const figure = document.getElementById("overlayFigure");
    if (figure) figure.style.display = overlayHidden ? "none" : "";
    toggleOverlayBtn.textContent = overlayHidden ? "Show overlay" : "Hide overlay";
  });
}

document.querySelectorAll("[data-filter]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const key = btn.getAttribute("data-filter");
    document.querySelectorAll(".finding-row").forEach((row) => {
      const sev = (row.dataset.severity || "").toLowerCase();
      row.style.display = key === "all" || sev.includes(key) ? "" : "none";
    });
  });
});
