const express = require("express");
const multer = require("multer");
const { execFile } = require("child_process");
const WebSocket = require("ws");
const fs = require("fs");
const ZoomVideo = require("@zoom/videosdk");

const app = express();
const PORT = 3000;

const wss = new WebSocket.Server({ port: 8080 });
wss.on("connection", (ws) => {
    console.log("WebSocket connection established");
});

// save temp audio files
const upload = multer({ dest: "uploads/" });

// Zoom SDK 
const client = ZoomVideo.createClient();
client.init("en-US", "CDN");

// Process audio using python script
function processAudio(filePath, ws) {
    execFile("python", ["your_python_script.py", filePath], (error, stdout, stderr) => {
        if (error) {
            console.error(`Error: ${error.message}`);
            return;
        }
        console.log(`Prediction: ${stdout}`);
        if (ws && ws.readyState === WebSocket.OPEN) {
             // Send prediction
            ws.send(stdout);
        }

        fs.unlink(filePath, () => {});
    });
}

app.post("/upload", upload.single("audio"), (req, res) => {
    const filePath = req.file.path;
    const ws = wss.clients.values().next().value; 
    processAudio(filePath, ws);
    res.sendStatus(200); 
});


app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});

// connect to zoom 
async function startZoomAudioCapture() {
    await client.join("YOUR_MEETING_SDK_KEY", "YOUR_MEETING_SECRET", "MEETING_ID", "USER_NAME");

    client.on("audio-track-added", (audioTrack) => {
        const stream = audioTrack.getAudioStream();
        let audioBuffer = Buffer.alloc(0);

        // 5 seconds of audio 
        stream.on("data", (chunk) => {
            audioBuffer = Buffer.concat([audioBuffer, chunk]);
            if (audioBuffer.length >= 5 * 16000 * 2) { 
                const filePath = `uploads/${Date.now()}.wav`;
                fs.writeFileSync(filePath, audioBuffer);
                // Get the WebSocket client
                const ws = wss.clients.values().next().value; 
                processAudio(filePath, ws);
                // reset the audio buffer
                audioBuffer = Buffer.alloc(0); 
            }
        });
    });
}

startZoomAudioCapture();
