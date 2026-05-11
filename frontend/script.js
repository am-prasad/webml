const API_BASE_URL = "http://localhost:8000";

let gaugeChart = null;
let comparisonChart = null;

// Initialize dashboard on page load
document.addEventListener("DOMContentLoaded", () => {
    initializeCharts();
    fetchMetrics();
    document.getElementById("predictionForm").addEventListener("submit", handlePrediction);
});

// Initialize Chart.js instances
function initializeCharts() {
    // Gauge Chart
    const gaugeCtx = document.getElementById("gaugeChart").getContext("2d");
    gaugeChart = new Chart(gaugeCtx, {
        type: "doughnut",
        data: {
            labels: ["CO₂ Level", "Remaining"],
            datasets: [{
                data: [0, 100],
                backgroundColor: ["#00d4ff", "#0f1629"],
                borderColor: ["#00ff88", "#1a1f3a"],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: false
                }
            },
            cutout: "75%"
        }
    });

    // Comparison Chart
    const comparisonCtx = document.getElementById("comparisonChart").getContext("2d");
    comparisonChart = new Chart(comparisonCtx, {
        type: "bar",
        data: {
            labels: ["MAE", "RMSE"],
            datasets: [
                {
                    label: "Train (80%)",
                    data: [10.50, 13.20],
                    backgroundColor: "#00d4ff",
                    borderColor: "#00ff88",
                    borderWidth: 1
                },
                {
                    label: "Test (20%)",
                    data: [12.34, 15.67],
                    backgroundColor: "#00ff88",
                    borderColor: "#00d4ff",
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    labels: {
                        color: "#e0e0e0"
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        color: "#a0a0a0"
                    },
                    grid: {
                        color: "rgba(0, 212, 255, 0.1)"
                    }
                },
                x: {
                    ticks: {
                        color: "#a0a0a0"
                    },
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
}

// Fetch metrics from backend
async function fetchMetrics() {
    try {
        const response = await fetch(`${API_BASE_URL}/metrics`);
        const data = await response.json();

        document.getElementById("algorithm").textContent = data.algorithm;
        document.getElementById("r2Score").textContent = data.r2_score.toFixed(4);
        document.getElementById("mae").textContent = data.mae.toFixed(2);
        document.getElementById("rmse").textContent = data.rmse.toFixed(2);

        // Update comparison chart
        comparisonChart.data.datasets[0].data = [data.train_mae, data.train_rmse];
        comparisonChart.data.datasets[1].data = [data.test_mae, data.test_rmse];
        comparisonChart.update();
    } catch (error) {
        console.error("Error fetching metrics:", error);
    }
}

// Handle prediction form submission
async function handlePrediction(event) {

    event.preventDefault();

    const plant_type =
        document.getElementById("plant_type").value;

    const fuel_flow =
        parseFloat(
            document.getElementById("fuel_flow").value
        );

    const boiler_load =
        parseFloat(
            document.getElementById("boiler_load").value
        );

    const ambient_temp =
        parseFloat(
            document.getElementById("ambient_temp").value
        );

    const carbon_capture =
        parseInt(
            document.getElementById("carbon_capture").value
        );

    try {

        const response = await fetch(
            `${API_BASE_URL}/predict`,
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    plant_type,
                    fuel_flow,
                    boiler_load,
                    ambient_temp,
                    carbon_capture
                })
            }
        );

        const data = await response.json();

        if (data.status === "success") {

            const prediction = data.prediction;

            document.getElementById(
                "predictionValue"
            ).textContent = prediction.toFixed(4);

            const percentage =
                Math.min(
                    (prediction / 150) * 100,
                    100
                );

            gaugeChart.data.datasets[0].data = [
                percentage,
                100 - percentage
            ];

            gaugeChart.update();
        }

    } catch (error) {

        console.error(
            "Prediction Error:",
            error
        );
    }
}