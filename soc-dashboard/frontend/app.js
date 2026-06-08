const API_BASE_URL = "http://localhost:8000";

const loadSamplesBtn = document.getElementById("loadSamplesBtn");
const sampleSelect = document.getElementById("sampleSelect");
const alertText = document.getElementById("alertText");
const triageBtn = document.getElementById("triageBtn");
const statusText = document.getElementById("status");

const predictedLabel = document.getElementById("predictedLabel");
const containmentPriority = document.getElementById("containmentPriority");
const latency = document.getElementById("latency");
const analystSummary = document.getElementById("analystSummary");
const riskExplanation = document.getElementById("riskExplanation");
const mitreList = document.getElementById("mitreList");
const actionsList = document.getElementById("actionsList");
const rawJson = document.getElementById("rawJson");

let samples = [];


function setStatus(message, isError = false) {
  statusText.textContent = message;
  statusText.style.color = isError ? "#b00020" : "#333";
}


function clearList(element) {
  while (element.firstChild) {
    element.removeChild(element.firstChild);
  }
}


function renderList(element, items) {
  clearList(element);

  if (!items || items.length === 0) {
    const li = document.createElement("li");
    li.textContent = "No items returned.";
    element.appendChild(li);
    return;
  }

  items.forEach(item => {
    const li = document.createElement("li");
    li.textContent = item;
    element.appendChild(li);
  });
}


loadSamplesBtn.addEventListener("click", async () => {
  try {
    setStatus("Loading samples...");

    const response = await fetch(`${API_BASE_URL}/samples?limit=20`);

    if (!response.ok) {
      throw new Error(`Failed to load samples: ${response.status}`);
    }

    const data = await response.json();
    samples = data.samples;

    sampleSelect.innerHTML = `<option value="">Select sample alert</option>`;

    samples.forEach((sample, index) => {
      const option = document.createElement("option");
      option.value = index;
      option.textContent = `Sample ${index} | Label: ${sample.label} | Category Preview`;
      sampleSelect.appendChild(option);
    });

    setStatus(`Loaded ${samples.length} sample alerts.`);
  } catch (error) {
    setStatus(error.message, true);
  }
});


sampleSelect.addEventListener("change", () => {
  const index = sampleSelect.value;

  if (index === "") {
    return;
  }

  const selected = samples[index];
  alertText.value = selected.alert_text;
});


triageBtn.addEventListener("click", async () => {
  const text = alertText.value.trim();

  if (!text) {
    setStatus("Please provide an alert_text value.", true);
    return;
  }

  try {
    setStatus("Running hybrid triage. This may take a few seconds...");
    triageBtn.disabled = true;

    const response = await fetch(`${API_BASE_URL}/triage`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        alert_text: text
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || `API error: ${response.status}`);
    }

    const result = await response.json();

    predictedLabel.textContent = result.ml_predicted_label;
    containmentPriority.textContent = result.containment_priority;
    latency.textContent = `${result.latency_seconds.toFixed(2)} sec`;

    analystSummary.textContent = result.analyst_summary;
    riskExplanation.textContent = result.risk_explanation;

    renderList(mitreList, result.mitre_interpretation);
    renderList(actionsList, result.recommended_actions);

    rawJson.textContent = result.raw_llm_response;

    setStatus("Hybrid triage completed.");
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    triageBtn.disabled = false;
  }
});
