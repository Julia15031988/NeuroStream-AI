const protocol = window.location.protocol === "https:" ? "wss" : "ws";
const wsUrl = `${protocol}://${window.location.host}/ws/eeg`;

const socket = new WebSocket(wsUrl);

const connectionStatus = document.getElementById("connection-status");

socket.onopen = () => {
    connectionStatus.textContent = "Connected";
    startEEGSimulation();
};

socket.onclose = () => {
    connectionStatus.textContent = "Disconnected";
};

socket.onerror = () => {
    connectionStatus.textContent = "Connection error";
};


// -------------------------
// EEG SIGNAL CHART
// -------------------------

const eegContext = document
    .getElementById("eegChart")
    .getContext("2d");

const eegChart = new Chart(eegContext, {
    type: "line",

    data: {
        labels: [],
        datasets: [
            {
                label: "EEG amplitude",
                data: [],
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.1,
            },
        ],
    },

    options: {
        responsive: true,
        animation: false,

        scales: {
            x: {
                title: {
                    display: true,
                    text: "Sample",
                },
            },

            y: {
                title: {
                    display: true,
                    text: "Amplitude",
                },
            },
        },
    },
});


// -------------------------
// BRAINWAVE POWER CHART
// -------------------------

const powerContext = document
    .getElementById("powerChart")
    .getContext("2d");

const powerChart = new Chart(powerContext, {
    type: "bar",

    data: {
        labels: [
            "Delta",
            "Theta",
            "Alpha",
            "Beta",
        ],

        datasets: [
            {
                label: "Power",
                data: [0, 0, 0, 0],
                borderWidth: 1,
            },
        ],
    },

    options: {
        responsive: true,
        animation: false,

        scales: {
            y: {
                beginAtZero: true,
            },
        },
    },
});


// -------------------------
// TEST EEG GENERATOR
// -------------------------

const samplingRate = 256;
const sampleCount = 256;

let timeOffset = 0;

function generateEEGSamples() {
    const samples = [];

    for (let i = 0; i < sampleCount; i++) {
        const time = (timeOffset + i) / samplingRate;

        // Alpha-dominant synthetic EEG
        const alpha =
            1.0 * Math.sin(2 * Math.PI * 10 * time);

        const theta =
            0.25 * Math.sin(2 * Math.PI * 6 * time);

        const beta =
            0.15 * Math.sin(2 * Math.PI * 20 * time);

        const noise =
            (Math.random() - 0.5) * 0.15;

        samples.push(
            alpha +
            theta +
            beta +
            noise
        );
    }

    timeOffset += sampleCount;

    return samples;
}


// -------------------------
// SEND EEG TO BACKEND
// -------------------------

function sendEEGPacket() {
    if (socket.readyState !== WebSocket.OPEN) {
        return;
    }

    const samples = generateEEGSamples();

    socket.send(
        JSON.stringify({
            channel: "Fp1",
            sampling_rate: samplingRate,
            samples: samples,
        })
    );

    updateEEGChart(samples);
}


function startEEGSimulation() {
    sendEEGPacket();

    setInterval(
        sendEEGPacket,
        1000
    );
}


// -------------------------
// RECEIVE AI RESULT
// -------------------------

socket.onmessage = (event) => {
    const response = JSON.parse(event.data);

    if (response.status !== "processed") {
        console.error(
            "EEG processing error:",
            response
        );

        return;
    }

    updateDashboard(response.result);
};


// -------------------------
// UPDATE EEG CHART
// -------------------------

function updateEEGChart(samples) {
    eegChart.data.labels =
        samples.map(
            (_, index) => index
        );

    eegChart.data.datasets[0].data =
        samples;

    eegChart.update();
}


// -------------------------
// UPDATE DASHBOARD
// -------------------------

function updateDashboard(result) {
    powerChart.data.datasets[0].data = [
        result.delta_power,
        result.theta_power,
        result.alpha_power,
        result.beta_power,
    ];

    powerChart.update();

    document.getElementById(
        "dominant-wave"
    ).textContent =
        result.dominant_wave ?? "-";

    document.getElementById(
        "attention-score"
    ).textContent =
        formatNumber(result.attention_score);

    document.getElementById(
        "relaxation-score"
    ).textContent =
        formatNumber(result.relaxation_score);


    // Machine Learning

    document.getElementById(
        "ml-state"
    ).textContent =
        result.predicted_state ?? "-";

    document.getElementById(
        "ml-confidence"
    ).textContent =
        formatConfidence(
            result.prediction_confidence
        );


    // Deep Learning

    document.getElementById(
        "dl-state"
    ).textContent =
        result.deep_learning_state ?? "-";

    document.getElementById(
        "dl-confidence"
    ).textContent =
        formatConfidence(
            result.deep_learning_confidence
        );


    // CNN

    document.getElementById(
        "cnn-state"
    ).textContent =
        result.cnn_state ?? "-";

    document.getElementById(
        "cnn-confidence"
    ).textContent =
        formatConfidence(
            result.cnn_confidence
        );
}


// -------------------------
// HELPERS
// -------------------------

function formatNumber(value) {
    if (typeof value !== "number") {
        return "-";
    }

    return value.toFixed(2);
}


function formatConfidence(value) {
    if (typeof value !== "number") {
        return "-";
    }

    return `${(value * 100).toFixed(2)}%`;
}
