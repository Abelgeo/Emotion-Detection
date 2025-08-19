const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const emotionSpan = document.getElementById('emotion');
const confidenceSpan = document.getElementById('confidence');
const ctx = canvas.getContext('2d');

navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(err => console.error('Error accessing webcam:', err));

setInterval(() => {
    if (video.videoWidth && video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const frame = canvas.toDataURL('image/jpeg');

        fetch('/detect_emotion', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ frame: frame })
        })
        .then(response => response.json())
        .then(data => {
            if (data.emotion !== 'no_face') {
                emotionSpan.textContent = data.emotion;
                confidenceSpan.textContent = (data.confidence * 100).toFixed(2) + '%';
            } else {
                emotionSpan.textContent = 'No face detected';
                confidenceSpan.textContent = 'N/A';
            }
        })
        .catch(err => console.error('Error:', err));
    }
}, 500);  // Process every 0.5 seconds for real-time feel