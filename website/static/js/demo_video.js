// Real-time video processing and statistics handling
class DemoVideoProcessor {
    constructor() {
        this.socket = io();
        this.videoElement = document.getElementById('camera-feed');
        this.canvas = document.getElementById('detection-overlay');
        this.ctx = this.canvas.getContext('2d');
        this.pedestrianCountElement = document.getElementById('pedestrianCount');
        this.processingFPSElement = document.getElementById('processingFPS');
        this.avgConfidenceElement = document.getElementById('avgConfidence');
        this.trackingIdsElement = document.getElementById('trackingIds');
        this.startButton = document.getElementById('startCamera');
        this.stopButton = document.getElementById('stopCamera');
        this.uploadInput = document.getElementById('videoUpload');
        this.uploadZone = document.getElementById('uploadZone');
        this.processingStatus = document.getElementById('processingStatus');
        this.statusMessage = document.getElementById('statusMessage');
        this.progressContainer = document.getElementById('progressContainer');
        this.progressBar = document.getElementById('progressBar');
        
        this.isProcessing = false;
        this.frameCount = 0;
        this.lastFrameTime = 0;
        this.detections = [];
        
        this.initializeEventListeners();
        this.initializeWebSocket();
        this.setupCanvasResizing();
    }
    
    initializeEventListeners() {
        // Process controls
        this.startButton.textContent = 'Start Process';
        this.startButton.addEventListener('click', () => this.startProcessing());
        this.stopButton.addEventListener('click', () => this.stopProcessing());
        
        // File upload handlers
        this.uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.uploadZone.classList.add('dragover');
        });
        
        this.uploadZone.addEventListener('dragleave', () => {
            this.uploadZone.classList.remove('dragover');
        });
        
        this.uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.uploadZone.classList.remove('dragover');
            
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('video/')) {
                this.handleVideoFile(file);
            } else {
                this.showError('Please drop a valid video file.');
            }
        });
        
        this.uploadInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                this.handleVideoFile(file);
            }
        });
    }
    
    setupCanvasResizing() {
        const resizeCanvas = () => {
            this.canvas.width = this.videoElement.clientWidth;
            this.canvas.height = this.videoElement.clientHeight;
        };
        window.addEventListener('resize', resizeCanvas);
        this.videoElement.addEventListener('loadedmetadata', resizeCanvas);
        resizeCanvas();
    }

    drawDetections(detections) {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        const scaleX = this.canvas.width / this.videoElement.videoWidth;
        const scaleY = this.canvas.height / this.videoElement.videoHeight;
        
        detections.forEach(detection => {
            const [x, y, width, height] = detection.bbox;
            const scaledX = x * scaleX;
            const scaledY = y * scaleY;
            const scaledWidth = width * scaleX;
            const scaledHeight = height * scaleY;
            
            // Draw bounding box
            this.ctx.strokeStyle = `hsl(${detection.track_id * 30 % 360}, 100%, 50%)`;
            this.ctx.lineWidth = 2;
            this.ctx.strokeRect(scaledX, scaledY, scaledWidth, scaledHeight);
            
            // Draw label
            this.ctx.fillStyle = this.ctx.strokeStyle;
            this.ctx.font = '14px Arial';
            const label = `ID: ${detection.track_id} (${(detection.confidence * 100).toFixed(1)}%)`;
            const textWidth = this.ctx.measureText(label).width;
            this.ctx.fillRect(scaledX, scaledY - 20, textWidth + 10, 20);
            this.ctx.fillStyle = '#fff';
            this.ctx.fillText(label, scaledX + 5, scaledY - 5);
        });
    }

    updateTrackingIds(detections) {
        this.trackingIdsElement.innerHTML = '';
        const uniqueIds = new Set(detections.map(d => d.track_id));
        uniqueIds.forEach(id => {
            const detection = detections.find(d => d.track_id === id);
            const confidence = (detection.confidence * 100).toFixed(1);
            const div = document.createElement('div');
            div.className = 'tracking-id';
            div.innerHTML = `ID ${id} <span class="confidence">${confidence}%</span>`;
            this.trackingIdsElement.appendChild(div);
        });
    }

    initializeWebSocket() {
        this.socket.on('connect', () => {
            console.log('Connected to server');
            this.statusMessage.textContent = 'Connected to server';
        });

        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
            this.statusMessage.textContent = 'Disconnected from server';
        });

        // Update existing frame_processed handler
        this.socket.on('frame_processed', (data) => {
            try {
                // Convert base64 to blob
                const imageData = atob(data.frame);
                const arrayBuffer = new ArrayBuffer(imageData.length);
                const uintArray = new Uint8Array(arrayBuffer);
                
                for (let i = 0; i < imageData.length; i++) {
                    uintArray[i] = imageData.charCodeAt(i);
                }
                
                const blob = new Blob([arrayBuffer], { type: 'image/jpeg' });
                const imageUrl = URL.createObjectURL(blob);
                
                // Update video element
                this.videoElement.src = imageUrl;
                this.detections = data.detections || [];
                
                // Draw detections and update metrics
                this.drawDetections(this.detections);
                this.updateMetrics(data.metrics);
                this.updateTrackingIds(this.detections);
                
                // Update progress
                if (data.progress) {
                    this.progressBar.style.width = `${data.progress}%`;
                    this.progressBar.setAttribute('aria-valuenow', data.progress);
                    this.statusMessage.textContent = `Processing: ${data.progress}%`;
                }
            } catch (error) {
                console.error('Error processing frame:', error);
            }
        });

        // Add new handlers for processing status
        this.socket.on('processing_complete', (data) => {
            this.statusMessage.textContent = 'Processing complete!';
            this.progressBar.style.width = '100%';
            this.progressBar.setAttribute('aria-valuenow', 100);
            this.startButton.disabled = false;
            this.stopButton.disabled = true;
            
            if (data.output_video) {
                const videoPlayer = document.createElement('video');
                videoPlayer.src = data.output_video;
                videoPlayer.controls = true;
                videoPlayer.className = 'w-100';
                document.getElementById('output-video-container').appendChild(videoPlayer);
            }
        });
        this.socket.on('processing_error', (data) => {
            this.showError(data.error);
        });
        
        this.socket.on('processing_started', () => {
            this.startButton.disabled = true;
            this.stopButton.disabled = false;
            this.isProcessing = true;
        });
        
        this.socket.on('processing_stopped', () => {
            this.startButton.disabled = false;
            this.stopButton.disabled = true;
            this.isProcessing = false;
        });
    }
    
    async startProcessing() {
        if (!this.currentFile) {
            this.showError('Please upload a video first.');
            return;
        }

        try {
            const formData = new FormData();
            formData.append('video', this.currentFile);
            this.updateUIForProcessing();
            await this.uploadVideo(formData);
            this.socket.emit('start_processing');
        } catch (error) {
            this.showError('Error processing video: ' + error.message);
        }
    }
    
    stopProcessing() {
        this.socket.emit('stop_processing');
        this.startButton.disabled = false;
        this.stopButton.disabled = true;
        this.isProcessing = false;
        this.statusMessage.textContent = 'Processing stopped.';
    }
    
    handleVideoFile(file) {
        // Validate file size (100MB limit)
        const maxSize = 100 * 1024 * 1024;
        if (file.size > maxSize) {
            this.showError('File size exceeds 100MB limit. Please choose a smaller file.');
            return;
        }
        
        // Validate file type
        const allowedTypes = ['video/mp4', 'video/quicktime', 'video/x-msvideo'];
        if (!allowedTypes.includes(file.type)) {
            this.showError('Invalid file type. Please upload MP4, MOV, or AVI files.');
            return;
        }
        
        // Display the video immediately
        const videoUrl = URL.createObjectURL(file);
        this.videoElement.src = videoUrl;
        this.videoElement.controls = true;
        
        // Store the file for processing
        this.currentFile = file;
        
        // Enable start button for processing
        this.startButton.disabled = false;
        
        // Update UI
        this.uploadZone.style.display = 'none';
        this.processingStatus.classList.remove('d-none');
        this.statusMessage.textContent = 'Video loaded. Click "Start Process" to begin detection.';
        this.processingStatus.classList.remove('alert-danger', 'alert-info');
        this.processingStatus.classList.add('alert-success');
    }
    
    updateUIForProcessing() {
        this.processingStatus.classList.remove('alert-success', 'alert-danger');
        this.processingStatus.classList.add('alert-info');
        this.statusMessage.textContent = 'Processing video...';
        this.progressContainer.classList.remove('d-none');
        this.progressBar.style.width = '0%';
        this.progressBar.setAttribute('aria-valuenow', 0);
        this.startButton.disabled = true;
        this.stopButton.disabled = false;
    }
    
    updateMetrics(metrics) {
        this.pedestrianCountElement.textContent = metrics.current_count;
        this.frameCount = metrics.frame_count;
        
        // Update average confidence
        if (this.detections.length > 0) {
            const avgConfidence = this.detections.reduce((sum, d) => sum + d.confidence, 0) / this.detections.length;
            this.avgConfidenceElement.textContent = `${(avgConfidence * 100).toFixed(1)}%`;
        } else {
            this.avgConfidenceElement.textContent = '0%';
        }
    }
    
    showError(message) {
        this.processingStatus.classList.remove('d-none', 'alert-info', 'alert-success');
        this.processingStatus.classList.add('alert-danger');
        this.statusMessage.textContent = message;
    }
    
    // Update upload function
    async uploadVideo(formData) {
        try {
            this.statusMessage.textContent = 'Uploading video...';
            const response = await fetch('/api/process_video', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`Upload failed: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            if (data.success) {
                this.statusMessage.textContent = 'Upload complete. Processing...';
                // Socket connection will handle the rest
            } else {
                throw new Error(data.error || 'Upload failed');
            }
        } catch (error) {
            this.showError(`Error: ${error.message}`);
            this.startButton.disabled = false;
            this.stopButton.disabled = true;
        }
    }
}

// Initialize video processor when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new DemoVideoProcessor();
});