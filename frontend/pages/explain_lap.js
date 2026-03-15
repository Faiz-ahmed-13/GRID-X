let shapChart = null;

// ===== INPUT RANGE CLAMPING =====
const inputRanges = {
    air_temp:   { min: 5,  max: 45 },
    track_temp: { min: 5,  max: 60 },
    humidity:   { min: 0,  max: 100 },
    rainfall:   { min: 0,  max: 50 },
    lap_number: { min: 1,  max: 100 },
    stint:      { min: 1,  max: 5 },
    tyre_age:   { min: 1,  max: 50 },
    total_laps: { min: 10, max: 100 }
};

document.addEventListener("DOMContentLoaded", () => {
    Object.entries(inputRanges).forEach(([id, { min, max }]) => {
        const el = document.getElementById(id);
        if (!el) return;
        el.addEventListener("change", () => {
            const val = parseFloat(el.value);
            if (val < min) el.value = min;
            if (val > max) el.value = max;
        });
    });
});

// ===== ENTER KEY NAVIGATION =====
const formFields = [
    "driver", "circuit", "compound",
    "air_temp", "track_temp", "humidity", "rainfall",
    "lap_number", "stint", "tyre_age", "total_laps"
];

document.addEventListener("keydown", function(e) {
    if (e.key !== "Enter") return;
    const active = document.activeElement;
    const currentIndex = formFields.indexOf(active.id);
    if (currentIndex === -1) return;
    e.preventDefault();
    if (currentIndex < formFields.length - 1) {
        document.getElementById(formFields[currentIndex + 1]).focus();
    } else {
        explainLap();
    }
});

// ===== MAIN EXPLAIN FUNCTION =====
function explainLap() {
    document.getElementById("results-section").style.display = "none";
    document.getElementById("error-msg").style.display = "none";
    document.getElementById("loading-msg").style.display = "block";

    function clamp(val, min, max) { return Math.min(max, Math.max(min, val)); }

    const lapNumber = clamp(parseInt(document.getElementById("lap_number").value || "10"), 1, 100);
    const totalLaps = clamp(parseInt(document.getElementById("total_laps").value || "52"), 10, 100);
    const sessionProgress = parseFloat((lapNumber / totalLaps).toFixed(3));

    const values = {
        driver:  document.getElementById("driver").value,
        circuit: document.getElementById("circuit").value,
        compound: document.getElementById("compound").value,
        weather: {
            air_temp:   clamp(parseFloat(document.getElementById("air_temp").value   || "18"), 5,  45),
            track_temp: clamp(parseFloat(document.getElementById("track_temp").value || "25"), 5,  60),
            humidity:   clamp(parseFloat(document.getElementById("humidity").value   || "65"), 0,  100),
            rainfall:   clamp(parseFloat(document.getElementById("rainfall").value   || "0"),  0,  50),
        },
        lap_number:       lapNumber,
        stint:            clamp(parseInt(document.getElementById("stint").value    || "1"),  1, 5),
        tyre_age:         clamp(parseInt(document.getElementById("tyre_age").value || "10"), 1, 50),
        session_progress: sessionProgress,
        total_laps:       totalLaps
    };

    const minDelay = new Promise(resolve => setTimeout(resolve, 1500));

    Promise.all([
        fetch("http://localhost:8000/explain-lap", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(values)
        }).then(response => response.json()),
        minDelay
    ])
    .then(([data]) => {
        console.log(data);
        document.getElementById("loading-msg").style.display = "none";

        if (!data.success) {
            document.getElementById("error-msg").style.display = "block";
            return;
        }

        const sorted = data.explanation.sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value));

        const labels      = sorted.map(d => d.feature.replace(/_/g, " "));
        const values_data = sorted.map(d => d.shap_value);
        const colors      = values_data.map(v => v >= 0 ? "rgba(255, 42, 42, 0.8)" : "rgba(0, 150, 255, 0.8)");
        const borderColors = values_data.map(v => v >= 0 ? "#ff2a2a" : "#0096ff");

        if (shapChart) shapChart.destroy();

        const ctx = document.getElementById("shapChart").getContext("2d");
        shapChart = new Chart(ctx, {
            type: "bar",
            data: {
                labels,
                datasets: [{
                    data: values_data,
                    backgroundColor: colors,
                    borderColor: borderColors,
                    borderWidth: 1,
                    borderRadius: 4,
                }]
            },
            options: {
                indexAxis: "y",
                responsive: true,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: ctx => ` SHAP: ${ctx.raw.toFixed(2)}`
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: "#aaa", font: { family: "Rajdhani" } },
                        grid:  { color: "rgba(255,255,255,0.05)" },
                        title: {
                            display: true,
                            text: "SHAP Value (impact on lap time)",
                            color: "#888",
                            font: { family: "Rajdhani", size: 12 }
                        }
                    },
                    y: {
                        ticks: { color: "#ccc", font: { family: "Rajdhani", size: 13 } },
                        grid:  { color: "rgba(255,255,255,0.05)" }
                    }
                }
            }
        });

        document.getElementById("results-section").style.display = "block";
        document.getElementById("results-section").scrollIntoView({ behavior: "smooth" });
    })
    .catch(error => {
        document.getElementById("loading-msg").style.display = "none";
        document.getElementById("error-msg").style.display = "block";
        console.log("Error:", error);
    });
}