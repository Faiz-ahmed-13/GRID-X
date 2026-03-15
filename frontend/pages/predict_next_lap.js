const defaultLaps = Array.from({ length: 10 }, (_, i) => ({
    DriverNumber: "1",
    LapNumber: i + 1,
    Stint: 1,
    Compound: "SOFT",
    Team: "Red Bull Racing",
    event_name: "Bahrain Grand Prix",
    circuit: "bahrain_circuit",
    year: 2024,
    round: 1,
    AirTemp: 28.0,
    Humidity: 50.0,
    Pressure: 1013.0,
    Rainfall: 0.0,
    TrackTemp: 35.0,
    WindSpeed: 5.0,
    WindDirection: 180.0,
    stint_lap_number: i + 1,
    tyre_age_laps: i + 1,
    session_progress: parseFloat(((i + 1) / 57).toFixed(3)),
    Position: 1,
    position_change: 0,
    AggressionScore: 0.85,
    ConsistencyScore: 0.9,
    BrakingIntensity: 0.75,
    TyrePreservation: 0.8,
    OvertakingAbility: 0.7
}));

const compounds = ["SOFT", "MEDIUM", "HARD"];

// ===== INPUT RANGES =====
const cellRanges = {
    stint:      { min: 1,  max: 5 },
    position:   { min: 1,  max: 20 },
    air_temp:   { min: 5,  max: 45 },
    track_temp: { min: 5,  max: 60 },
    rainfall:   { min: 0,  max: 50,  step: 0.1 },
    tyre_age:   { min: 1,  max: 50 }
};

function clamp(val, min, max) { return Math.min(max, Math.max(min, val)); }

function buildTable() {
    const tbody = document.getElementById("lap-input-body");
    tbody.innerHTML = "";

    defaultLaps.forEach((lap, i) => {
        const row = document.createElement("tr");
        row.innerHTML = `
            <td>${lap.LapNumber}</td>
            <td><input type="number" class="cell-input" id="stint_${i}"      value="${lap.Stint}"          min="1"  max="5"></td>
            <td>
                <select class="cell-select" id="compound_${i}">
                    ${compounds.map(c => `<option value="${c}" ${c === lap.Compound ? "selected" : ""}>${c}</option>`).join("")}
                </select>
            </td>
            <td><input type="number" class="cell-input" id="position_${i}"   value="${lap.Position}"       min="1"  max="20"></td>
            <td><input type="number" class="cell-input" id="air_temp_${i}"   value="${lap.AirTemp}"        min="5"  max="45"></td>
            <td><input type="number" class="cell-input" id="track_temp_${i}" value="${lap.TrackTemp}"      min="5"  max="60"></td>
            <td><input type="number" class="cell-input" id="rainfall_${i}"   value="${lap.Rainfall}"       min="0"  max="50"  step="0.1"></td>
            <td><input type="number" class="cell-input" id="tyre_age_${i}"   value="${lap.tyre_age_laps}"  min="1"  max="50"></td>
        `;
        tbody.appendChild(row);
    });

    // Attach clamp listeners to all cell inputs
    for (let i = 0; i < 10; i++) {
        Object.entries(cellRanges).forEach(([field, { min, max }]) => {
            const el = document.getElementById(`${field}_${i}`);
            if (!el) return;
            el.addEventListener("change", () => {
                const val = parseFloat(el.value);
                if (val < min) el.value = min;
                if (val > max) el.value = max;
            });
        });
    }

    // Build Enter nav order after table is built
    buildEnterNav();
}

// ===== ENTER KEY NAVIGATION =====
// Order: row by row — stint, compound, position, air_temp, track_temp, rainfall, tyre_age
function buildEnterNav() {
    const navOrder = [];
    for (let i = 0; i < 10; i++) {
        navOrder.push(
            `stint_${i}`,
            `compound_${i}`,
            `position_${i}`,
            `air_temp_${i}`,
            `track_temp_${i}`,
            `rainfall_${i}`,
            `tyre_age_${i}`
        );
    }

    document.addEventListener("keydown", function (e) {
        if (e.key !== "Enter") return;
        const active = document.activeElement;
        const idx = navOrder.indexOf(active.id);
        if (idx === -1) return;
        e.preventDefault();
        if (idx < navOrder.length - 1) {
            document.getElementById(navOrder[idx + 1]).focus();
        } else {
            predictNextLap();
        }
    });
}

function collectLaps() {
    return defaultLaps.map((lap, i) => ({
        ...lap,
        Stint:          clamp(parseInt(document.getElementById(`stint_${i}`).value),      1,  5),
        Compound:       document.getElementById(`compound_${i}`).value,
        Position:       clamp(parseInt(document.getElementById(`position_${i}`).value),   1,  20),
        AirTemp:        clamp(parseFloat(document.getElementById(`air_temp_${i}`).value), 5,  45),
        TrackTemp:      clamp(parseFloat(document.getElementById(`track_temp_${i}`).value), 5, 60),
        Rainfall:       clamp(parseFloat(document.getElementById(`rainfall_${i}`).value), 0,  50),
        tyre_age_laps:  clamp(parseInt(document.getElementById(`tyre_age_${i}`).value),   1,  50),
        stint_lap_number: clamp(parseInt(document.getElementById(`tyre_age_${i}`).value), 1,  50),
    }));
}

function predictNextLap() {
    document.getElementById("results-section").style.display = "none";
    document.getElementById("error-msg").style.display = "none";
    document.getElementById("loading-msg").style.display = "block";

    const laps = collectLaps();
    const minDelay = new Promise(resolve => setTimeout(resolve, 1500));

    Promise.all([
        fetch("http://localhost:8000/next-lap", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ laps })
        }).then(response => response.json()),
        minDelay
    ])
    .then(([data]) => {
        document.getElementById("loading-msg").style.display = "none";
        document.getElementById("predicted-lap").textContent = data.predicted_next_lap;
        document.getElementById("results-section").style.display = "block";
        document.getElementById("results-section").scrollIntoView({ behavior: "smooth" });
    })
    .catch(error => {
        document.getElementById("loading-msg").style.display = "none";
        document.getElementById("error-msg").style.display = "block";
        console.log("Error:", error);
    });
}

buildTable();