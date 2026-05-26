const API_BASE_URL = "http://localhost:8000";

let gaugeChart = null;
let comparisonChart = null;

// Initialize dashboard on page load
document.addEventListener("DOMContentLoaded", () => {
    initializeCharts();
    fetchMetrics();
    document.getElementById("predictionForm").addEventListener("submit", handlePrediction);
});

function initializeCharts() {
    // Gauge Chart
    const gaugeCtx = document.getElementById("gaugeChart").getContext("2d");
    gaugeChart = new Chart(gaugeCtx, {
        type: "doughnut",
        data: {
            labels: ["CO₂ Level", "Remaining"],
            datasets: [{
                data: [0, 100],
                backgroundColor: ["#00ff88", "#0f1629"],
                borderColor: ["#00ff88", "#1a1f3a"],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: { legend: { display: false } },
            cutout: "75%"
        }
    });

    // Comparison Chart (Now mapping Train MAE vs Test MAE)
    const comparisonCtx = document.getElementById("comparisonChart").getContext("2d");
    comparisonChart = new Chart(comparisonCtx, {
        type: "bar",
        data: {
            labels: ["MAE Error", "RMSE Error"],
            datasets: [
                {
                    label: "Train (80%)",
                    data: [0, 0],
                    backgroundColor: "#00d4ff",
                    borderColor: "#00ff88",
                    borderWidth: 1
                },
                {
                    label: "Test (20%)",
                    data: [0, 0],
                    backgroundColor: "#00ff88",
                    borderColor: "#00d4ff",
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: { legend: { labels: { color: "#e0e0e0" } } },
            scales: {
                y: { beginAtZero: true, ticks: { color: "#a0a0a0" }, grid: { color: "rgba(0, 212, 255, 0.1)" } },
                x: { ticks: { color: "#a0a0a0" }, grid: { display: false } }
            }
        }
    });
}

// Fetch metrics from backend
async function fetchMetrics() {
    try {
        const response = await fetch(`${API_BASE_URL}/metrics`);
        const data = await response.json();

        document.getElementById("algorithm").textContent = data.algorithm || "-";
        document.getElementById("r2Score").textContent = data.test_r2?.toFixed(4) || "-";
        document.getElementById("mae").textContent = data.test_mae?.toFixed(4) || "-";
        document.getElementById("rmse").textContent = data.test_rmse?.toFixed(4) || "-";

        // Update comparison chart
        comparisonChart.data.datasets[0].data = [data.train_mae || 0, data.train_rmse || 0];
        comparisonChart.data.datasets[1].data = [data.test_mae || 0, data.test_rmse || 0];
        comparisonChart.update();
    } catch (error) {
        console.error("Error fetching metrics:", error);
    }
}

// Handle prediction form submission
async function handlePrediction(event) {
    event.preventDefault();

    const fuel_flow = parseFloat(document.getElementById("fuel_flow").value);
    const boiler_load = parseFloat(document.getElementById("boiler_load").value);
    const ambient_temp = parseFloat(document.getElementById("ambient_temp").value);
    const carbon_capture = parseInt(document.getElementById("carbon_capture").value);

    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                fuelflow: fuel_flow,
                boilerload: boiler_load,
                ambient_temp: ambient_temp,
                capture_on: carbon_capture
            })
        });

        const data = await response.json();

        if (data.status === "success") {
            const prediction = data.prediction;

            
            document.getElementById("predictionValue").textContent = prediction.toFixed(4) + "%";

            // Make gauge relative to 50% max limit
            const percentage = Math.min((prediction / 50) * 100, 100);
            gaugeChart.data.datasets[0].data = [percentage, 100 - percentage];
            
            
            if (data.is_high_emission) {
                gaugeChart.data.datasets[0].backgroundColor = ["#ff3333", "#0f1629"];
                gaugeChart.data.datasets[0].borderColor = ["#ff0000", "#1a1f3a"];
            } else {
                gaugeChart.data.datasets[0].backgroundColor = ["#00ff88", "#0f1629"];
                gaugeChart.data.datasets[0].borderColor = ["#00ff88", "#1a1f3a"];
            }
            gaugeChart.update();

            
            const alertBox = document.getElementById("anomalyAlert");
            if (data.is_anomaly) {
                alertBox.classList.remove("hidden");
                alertBox.classList.add("visible");
            } else {
                alertBox.classList.remove("visible");
                alertBox.classList.add("hidden");
            }
        } else {
            console.error("Backend returned an error:", data);
            alert("Backend Error: " + (data.error || "Failed to predict."));
        }
    } catch (error) {
        console.error("Prediction Error:", error);
        alert("Failed to connect to the API. Is FastAPI running?");
    }
}