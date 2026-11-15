/* ===== Smart Traffic Detection and Monitoring System - front-end logic ===== */
const page = document.body.dataset.page;
const COLORS = {
  Normal: "#22c55e", Moderate: "#facc15", Heavy: "#f97316", Critical: "#ef4444",
  Red: "#ef4444", Yellow: "#facc15", Green: "#22c55e",
};
const PALETTE = ["#38bdf8", "#22c55e", "#f97316", "#facc15", "#ef4444", "#a78bfa", "#2dd4bf", "#f472b6"];

function el(id) { return document.getElementById(id); }
function showLoader(text) { el("loaderText").textContent = text || "Loading..."; el("loader").classList.remove("d-none"); }
function hideLoader() { el("loader").classList.add("d-none"); }
function levelBadge(level) {
  const l = String(level || "").toLowerCase();
  return `<span class="level-badge lv-${l}">${level}</span>`;
}
function lightBadge(light) {
  return `<span class="light-badge lt-${String(light).toLowerCase()}">${light}</span>`;
}
function showMessage(text) {
  const box = el("statsMessage");
  if (!box) return;
  box.textContent = text || "";
  box.classList.toggle("d-none", !text);
}

/* ---------- Charts (dashboard + analytics) ---------- */
if (window.Chart) {
  Chart.defaults.color = "#93a1b5";
  Chart.defaults.borderColor = "#273140";
  Chart.defaults.font.family = '"IBM Plex Sans", "Segoe UI", sans-serif';
}
const charts = {};
function drawChart(id, type, data, colors, label) {
  const canvas = el(id);
  if (!canvas) return;
  const round = type === "doughnut";
  const dataset = { label: label || "", data: data.values, backgroundColor: colors, borderWidth: round ? 0 : 0, borderRadius: round ? 0 : 5 };
  if (charts[id]) {                       // update an existing chart
    charts[id].data.labels = data.labels;
    charts[id].data.datasets[0] = dataset;
    charts[id].update();
    return;
  }
  charts[id] = new Chart(canvas, {
    type, data: { labels: data.labels, datasets: [dataset] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: round, position: "bottom" } },
      scales: round ? {} : { y: { beginAtZero: true } },
    },
  });
}
function paletteFor(labels) { return labels.map((l, i) => COLORS[l] || PALETTE[i % PALETTE.length]); }

/* ---------- Filters ---------- */
let filtersBuilt = false;
const filterDefs = [
  ["location", "Location", "locations"], ["zone", "Zone", "zones"],
  ["level", "Traffic level", "levels"], ["light", "Traffic light", "lights"],
];
function buildFilters(options) {
  const box = el("filterBox");
  if (!box || filtersBuilt) return;
  let html = "";
  filterDefs.forEach(([key, label, optKey]) => {
    html += `<div class="mb-3"><label class="form-label" for="f_${key}">${label}</label>
      <select class="form-select" id="f_${key}" data-filter="${key}"><option value="">All</option>` +
      options[optKey].map(o => `<option value="${o}">${o}</option>`).join("") + `</select></div>`;
  });
  html += `<div class="mb-3"><label class="form-label" for="f_date">Date</label>
    <input type="date" class="form-control" id="f_date" data-filter="date"></div>
    <button type="button" class="btn btn-outline-light btn-sm mb-3" id="resetFilters"><i class="fa-solid fa-rotate-left"></i> Reset filters</button>`;
  box.innerHTML = html;
  box.querySelectorAll("[data-filter]").forEach(f => f.addEventListener("change", loadStats));
  el("resetFilters").addEventListener("click", () => {
    box.querySelectorAll("[data-filter]").forEach(f => (f.value = ""));
    loadStats();
  });
  filtersBuilt = true;
}
function filterQuery() {
  const params = new URLSearchParams();
  document.querySelectorAll("#filterBox [data-filter]").forEach(f => { if (f.value) params.set(f.dataset.filter, f.value); });
  return params.toString();
}

/* ---------- Dashboard + analytics data ---------- */
async function loadStats() {
  showLoader("Loading traffic data...");
  try {
    const res = await fetch("/api/stats?" + filterQuery());
    const data = await res.json();
    if (!res.ok) { showMessage(data.error || "Could not load data."); return; }
    buildFilters(data.options);
    if (data.empty) { showMessage(data.message); return; }
    showMessage("");
    const s = data.summary, c = data.charts;
    const set = (id, v) => { const n = el("s_" + id); if (n) n.innerHTML = v; };
    set("total_vehicles", s.total_vehicles.toLocaleString());
    set("cars", s.cars.toLocaleString());
    set("buses", s.buses.toLocaleString());
    set("trucks", s.trucks.toLocaleString());
    set("motorcycles", s.motorcycles.toLocaleString());
    set("pedestrians", s.pedestrians.toLocaleString());
    set("avg_delay", s.avg_delay);
    set("current_level", levelBadge(s.current_level));
    drawChart("chartLocation", "bar", c.vehicles_by_location, "#38bdf8", "Vehicles");
    drawChart("chartTypes", page === "dashboard" ? "doughnut" : "bar", c.vehicle_types,
              ["#22c55e", "#f97316", "#ef4444", "#facc15"], "Vehicles");
    drawChart("chartLevel", "doughnut", c.level_distribution, paletteFor(c.level_distribution.labels));
    drawChart("chartLight", "doughnut", c.light_distribution, paletteFor(c.light_distribution.labels));
    drawChart("chartDelay", "bar", c.delay_by_location, "#f97316", "Average delay (sec)");
  } catch (err) {
    showMessage("Could not reach the server. Make sure app.py is running.");
  } finally {
    hideLoader();
  }
}

/* ---------- Traffic data table ---------- */
async function loadTable() {
  const errorBox = el("tableError");
  showLoader("Loading records...");
  try {
    const res = await fetch("/api/records");
    const data = await res.json();
    if (!res.ok) { errorBox.textContent = data.error; errorBox.classList.remove("d-none"); return; }
    const rows = data.records;
    el("totalRecords").textContent = rows.length.toLocaleString();

    const fillSelect = (id, key) => {
      const values = [...new Set(rows.map(r => r[key]).filter(Boolean))].sort();
      el(id).insertAdjacentHTML("beforeend", values.map(v => `<option value="${v}">${v}</option>`).join(""));
    };
    fillSelect("fLocation", "Location"); fillSelect("fLevel", "Traffic_Level"); fillSelect("fLight", "Traffic_Light");

    const badge = (fn) => (d, type) => (type === "display" ? fn(d) : d);
    const table = $("#trafficTable").DataTable({
      data: rows, pageLength: 15, lengthMenu: [10, 15, 25, 50, 100], order: [[1, "desc"]],
      columns: [
        { data: "Record_ID" }, { data: "Timestamp" }, { data: "Location" }, { data: "Zone" },
        { data: "Total_Vehicles" }, { data: "Cars" }, { data: "Buses" }, { data: "Trucks" },
        { data: "Motorcycles" }, { data: "Pedestrians" },
        { data: "Traffic_Light", render: badge(lightBadge) },
        { data: "Estimated_Delay_Seconds" },
        { data: "Traffic_Level", render: badge(levelBadge) },
      ],
      language: { search: "Search:", emptyTable: "No traffic records found." },
    });
    const exact = (v) => (v ? "^" + $.fn.dataTable.util.escapeRegex(v) + "$" : "");
    const bind = (id, col) => el(id).addEventListener("change", e => table.column(col).search(exact(e.target.value), true, false).draw());
    bind("fLocation", 2); bind("fLevel", 12); bind("fLight", 10);
  } catch (err) {
    errorBox.textContent = "Could not load records. Make sure app.py is running.";
    errorBox.classList.remove("d-none");
  } finally {
    hideLoader();
  }
}

/* ---------- Detection page ---------- */
function initDetection() {
  const drop = el("dropzone"), input = el("fileInput"), btn = el("detectBtn");
  const errBox = el("uploadError");
  let chosen = null;
  const allowed = ["jpg", "jpeg", "png", "mp4", "avi", "mov"];

  function showError(msg) { errBox.textContent = msg; errBox.classList.toggle("d-none", !msg); }
  function pick(file) {
    showError("");
    if (!file) return;
    const ext = file.name.split(".").pop().toLowerCase();
    if (!allowed.includes(ext)) { showError("Invalid file type. Allowed: JPG, JPEG, PNG, MP4, AVI, MOV."); chosen = null; btn.disabled = true; return; }
    if (file.size === 0) { showError("The selected file is empty."); chosen = null; btn.disabled = true; return; }
    chosen = file;
    el("fileName").textContent = `${file.name} (${(file.size / 1048576).toFixed(1)} MB)`;
    btn.disabled = false;
  }
  drop.addEventListener("click", () => input.click());
  drop.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  input.addEventListener("change", () => pick(input.files[0]));
  ["dragenter", "dragover"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("drag"); }));
  ["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("drag"); }));
  drop.addEventListener("drop", e => pick(e.dataTransfer.files[0]));

  btn.addEventListener("click", () => {
    if (!chosen) return;
    const form = new FormData();
    form.append("file", chosen);
    const xhr = new XMLHttpRequest();
    const bar = el("progressBar");
    el("progressWrap").classList.remove("d-none");
    bar.style.width = "0%";
    btn.disabled = true;
    showError("");
    xhr.upload.onprogress = e => { if (e.lengthComputable) bar.style.width = (e.loaded / e.total * 100) + "%"; };
    xhr.upload.onload = () => showLoader("Detecting vehicles... videos can take a few minutes.");
    xhr.onload = () => {
      hideLoader(); btn.disabled = false;
      let data = {};
      try { data = JSON.parse(xhr.responseText); } catch (e) { data = { error: "Unexpected response from the server." }; }
      if (xhr.status >= 200 && xhr.status < 300 && data.ok) showResult(data);
      else showError(data.error || "Detection failed.");
    };
    xhr.onerror = () => { hideLoader(); btn.disabled = false; showError("Could not reach the server. Make sure app.py is running."); };
    xhr.open("POST", "/upload");
    xhr.send(form);
  });
}

function showResult(d) {
  el("resultEmpty").classList.add("d-none");
  el("resultBox").classList.remove("d-none");
  el("rTotal").textContent = d.counts.total;
  el("rCars").textContent = d.counts.cars;
  el("rMotorcycles").textContent = d.counts.motorcycles;
  el("rBuses").textContent = d.counts.buses;
  el("rTrucks").textContent = d.counts.trucks;
  const lv = el("rLevel");
  lv.textContent = d.level;
  lv.className = "level-badge lv-" + d.level.toLowerCase();

  ["Red", "Yellow", "Green"].forEach(c => el("lamp" + c).classList.remove("on"));
  if (d.light) {
    const name = d.light.charAt(0) + d.light.slice(1).toLowerCase();
    el("lamp" + name).classList.add("on");
    el("rLight").textContent = "Traffic Light: " + d.light;
  } else {
    el("rLight").textContent = "Traffic Light: Not Detected";
  }

  let note = d.note || "";
  const media = el("mediaBox");
  const bust = "?t=" + Date.now();
  if (d.type === "image") {
    media.innerHTML = `<img src="${d.url}${bust}" alt="Processed traffic image with detected vehicles">`;
  } else if (d.playable) {
    media.innerHTML = `<video src="${d.url}${bust}" controls playsinline></video>`;
  } else {
    media.innerHTML = `<p class="mb-0 text-secondary">The processed video is ready, but this browser cannot play its format. Use the download button. (Installing ffmpeg enables in-page playback.)</p>`;
  }
  el("rNote").textContent = note;
  el("downloadLink").href = d.url + "?download=1";

  const rows = el("detRows");
  rows.innerHTML = d.detections.length
    ? d.detections.map(x => `<tr><td>${x.object}</td><td>${(x.confidence * 100).toFixed(0)}%</td><td>${x.box.join(", ")}</td></tr>`).join("")
    : `<tr><td colspan="3" class="text-secondary">No vehicles detected.</td></tr>`;
  el("resultBox").scrollIntoView({ behavior: "smooth", block: "start" });
}

/* ---------- Start the right code for each page ---------- */
document.addEventListener("DOMContentLoaded", () => {
  if (page === "dashboard" || page === "analytics") loadStats();
  if (page === "traffic_data") loadTable();
  if (page === "detection") initDetection();
});
