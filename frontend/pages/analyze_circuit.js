let selectedFile = null;

function handleFile(event) {
    const file = event.target.files[0];
    if (!file) return;

    selectedFile = file;

    const reader = new FileReader();
    reader.onload = function(e) {
        document.getElementById("image-preview").src = e.target.result;
        document.getElementById("preview-container").style.display = "block";
        document.getElementById("analyze-btn").style.display = "inline-block";
        document.getElementById("results-section").style.display = "none";
        document.getElementById("error-msg").style.display = "none";
    };
    reader.readAsDataURL(file);
}

// Drag and drop support
const uploadArea = document.getElementById("upload-area");

uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadArea.classList.add("drag-over");
});

uploadArea.addEventListener("dragleave", () => {
    uploadArea.classList.remove("drag-over");
});

uploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadArea.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) {
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = function(ev) {
            document.getElementById("image-preview").src = ev.target.result;
            document.getElementById("preview-container").style.display = "block";
            document.getElementById("analyze-btn").style.display = "inline-block";
            document.getElementById("results-section").style.display = "none";
        };
        reader.readAsDataURL(file);
    }
});

function formatKey(key) {
    return key.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

function analyzeCircuit() {
    if (!selectedFile) return;

    document.getElementById("results-section").style.display = "none";
    document.getElementById("error-msg").style.display = "none";
    document.getElementById("loading-msg").style.display = "block";
    document.getElementById("analyze-btn").style.display = "none";

    const formData = new FormData();
    formData.append("file", selectedFile);

    const minDelay = new Promise(resolve => setTimeout(resolve, 1500));

    Promise.all([
        fetch("http://localhost:8000/analyze-circuit", {
            method: "POST",
            body: formData
        }).then(response => response.json()),
        minDelay
    ])
    .then(([data]) => {
        console.log(data);
        document.getElementById("loading-msg").style.display = "none";
        document.getElementById("analyze-btn").style.display = "inline-block";

        if (!data.success) {
            document.getElementById("error-msg").style.display = "block";
            return;
        }

        const top = data.predictions[0];
        const confidence = (top.confidence * 100).toFixed(1);

        document.getElementById("top-circuit-name").textContent = top.circuit.replace(/_/g, " ").toUpperCase();
        document.getElementById("top-circuit-confidence").textContent = `Confidence: ${confidence}%`;
        document.getElementById("conf-fill").style.width = `${confidence}%`;

        // Metadata grid
        const metaGrid = document.getElementById("metadata-grid");
        metaGrid.innerHTML = "";
        if (top.metadata && Object.keys(top.metadata).length > 0) {
            Object.entries(top.metadata).forEach(([key, value]) => {
                if (key === "description") return;
                const item = document.createElement("div");
                item.className = "meta-item";
                item.innerHTML = `
                    <p class="meta-key">${formatKey(key)}</p>
                    <p class="meta-value">${value}</p>
                `;
                metaGrid.appendChild(item);
            });

            // Description at the bottom
            if (top.metadata.description) {
                const desc = document.createElement("div");
                desc.className = "meta-description";
                desc.textContent = top.metadata.description;
                metaGrid.appendChild(desc);
            }
        }

        // Other predictions
        const othersContainer = document.getElementById("other-predictions");
        othersContainer.innerHTML = "";
        data.predictions.slice(1).forEach(pred => {
            const conf = (pred.confidence * 100).toFixed(1);
            const card = document.createElement("div");
            card.className = "other-card";
            card.innerHTML = `
                <p class="other-name">${pred.circuit.replace(/_/g, " ").toUpperCase()}</p>
                <p class="other-conf">${conf}%</p>
                <div class="other-bar"><div class="other-fill" style="width:${conf}%"></div></div>
            `;
            othersContainer.appendChild(card);
        });

        document.getElementById("results-section").style.display = "block";
        document.getElementById("results-section").scrollIntoView({ behavior: "smooth" });
    })
    .catch(error => {
        document.getElementById("loading-msg").style.display = "none";
        document.getElementById("analyze-btn").style.display = "inline-block";
        document.getElementById("error-msg").style.display = "block";
        console.log("Error:", error);
    });
}document.addEventListener("keydown", function(e) {
    if (e.key === "Enter" && selectedFile) {
        analyzeCircuit();
    }
});