/**
 * Main JavaScript for Web Summarizer app
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    const tooltipList = [...tooltipTriggerList].map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl));
    
    // Add timestamp formatter
    if (typeof Intl !== 'undefined') {
        // Format timestamps in a user-friendly way
        const timestamps = document.querySelectorAll('.timestamp');
        timestamps.forEach(el => {
            const timestamp = parseInt(el.getAttribute('data-timestamp'));
            if (!isNaN(timestamp)) {
                const date = new Date(timestamp * 1000);
                el.textContent = new Intl.DateTimeFormat(navigator.language, {
                    dateStyle: 'medium',
                    timeStyle: 'short'
                }).format(date);
            }
        });
    }
    
    // Function to handle API search (for potential AJAX searching)
    window.apiSearch = async function(query, depth = 3) {
        try {
            const response = await fetch('/api/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: query,
                    depth: depth
                })
            });
            
            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Search error:', error);
            return { error: error.message };
        }
    };
    
    // Handle form submission if we want to make it AJAX in the future
    const searchForm = document.getElementById('search-form');
    if (searchForm) {
        // We're keeping the normal form submission for now
        // but this is where we would add AJAX functionality
        searchForm.addEventListener('submit', function(e) {
            // Add loading state to button
            const button = document.getElementById('search-button');
            if (button) {
                button.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> Processing...';
                button.disabled = true;
            }
            
            // Continue with normal form submission
            return true;
        });
    }
    
    // Add filter functionality on results page if needed
    const filterInput = document.getElementById('filter-sources');
    if (filterInput) {
        filterInput.addEventListener('input', function() {
            const filterValue = this.value.toLowerCase();
            const sourceItems = document.querySelectorAll('.list-group-item-action');
            
            sourceItems.forEach(item => {
                const title = item.querySelector('h5').textContent.toLowerCase();
                const snippet = item.querySelector('.source-snippet').textContent.toLowerCase();
                
                if (title.includes(filterValue) || snippet.includes(filterValue)) {
                    item.style.display = '';
                } else {
                    item.style.display = 'none';
                }
            });
        });
    }

    // Speech Recognition Integration
    initSpeechRecognition();
});

// Add custom filter for date formatting
// This would be handled on the backend in a real app
// For now, we add a simple helper function
function formatDate(timestamp) {
    if (!timestamp) return '';
    
    const date = new Date(timestamp * 1000);
    return date.toLocaleString();
}

/**
 * Initialize speech recognition functionality
 */
function initSpeechRecognition() {
    // Speech recognition elements
    const speechButton = document.getElementById('speech-input-btn');
    const queryInput = document.getElementById('query');
    const speechFeedback = document.getElementById('speech-feedback');
    const speechStatus = document.querySelector('.speech-status');
    const audioBars = document.querySelectorAll('.audio-level-indicator .bar');
    
    // Check if speech recognition is supported
    if (!speechButton || !('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
        if (speechButton) {
            speechButton.style.display = 'none'; // Hide button if not supported
            console.log('Speech recognition not supported in this browser');
        }
        return;
    }
    
    // Initialize speech recognition
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    
    // Configure recognition
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.maxAlternatives = 3;
    recognition.lang = 'en-US'; // Default to English, could be made configurable
    
    let isListening = false;
    let audioContext;
    let analyzer;
    let microphone;
    let animationFrame;
    let microphoneStream = null;
    
    // Event handlers
    speechButton.addEventListener('click', startSpeechRecognition);
    
    // Explicitly request microphone access before starting recognition
    async function startSpeechRecognition() {
        if (isListening) {
            stopSpeechRecognition();
            return;
        }
        
        speechButton.disabled = true;
        speechFeedback.classList.add('active');
        speechStatus.textContent = 'Requesting microphone access...';
        
        try {
            // First explicitly request microphone access
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            microphoneStream = stream;
            
            // Set up audio analysis
            setupAudioAnalysis(stream);
            
            // Then start speech recognition
            recognition.start();
            console.log('Speech recognition started');
            
            speechButton.disabled = false;
        } catch (error) {
            console.error('Error accessing microphone:', error);
            speechStatus.textContent = 'Microphone access denied. Check your browser settings.';
            speechButton.disabled = false;
            
            setTimeout(() => {
                speechFeedback.classList.remove('active');
            }, 3000);
        }
    }
    
    function stopSpeechRecognition() {
        if (isListening) {
            recognition.stop();
            console.log('Speech recognition stopped');
        }
        
        stopAudioAnalysis();
        isListening = false;
    }
    
    recognition.onstart = function() {
        console.log('Recognition started event fired');
        isListening = true;
        speechButton.classList.add('listening');
        speechStatus.textContent = 'Listening... Speak now';
    };
    
    recognition.onresult = function(event) {
        console.log('Recognition result event:', event);
        const result = event.results[0];
        const transcript = result[0].transcript;
        const confidence = result[0].confidence;
        
        console.log('Transcript:', transcript, 'Confidence:', confidence);
        
        // Update input field with the recognized text
        queryInput.value = transcript;
        
        // Set confidence styling
        queryInput.className = 'form-control form-control-lg';
        if (confidence > 0.8) {
            queryInput.classList.add('confidence-high');
        } else if (confidence > 0.5) {
            queryInput.classList.add('confidence-medium');
        } else {
            queryInput.classList.add('confidence-low');
        }
        
        // If final result, update status
        if (result.isFinal) {
            speechStatus.textContent = 'Got it! Click Search or speak again to change.';
        }
    };
    
    recognition.onerror = function(event) {
        console.error('Speech recognition error:', event.error, event);
        isListening = false;
        speechButton.classList.remove('listening', 'processing');
        
        // Show specific error messages
        let errorMessage = '';
        switch (event.error) {
            case 'no-speech':
                errorMessage = 'No speech was detected. Please speak more clearly and try again.';
                break;
            case 'aborted':
                errorMessage = 'Speech input was aborted.';
                break;
            case 'audio-capture':
                errorMessage = 'No microphone was found. Ensure it is plugged in and working.';
                break;
            case 'not-allowed':
                errorMessage = 'Microphone access denied. Check your browser settings.';
                break;
            case 'network':
                errorMessage = 'Network error occurred. Please try again.';
                break;
            case 'service-not-allowed':
                errorMessage = 'Speech service not allowed. Try a different browser.';
                break;
            default:
                errorMessage = `Error: ${event.error}`;
        }
        
        speechStatus.textContent = errorMessage;
        
        // Clean up audio context
        stopAudioAnalysis();
        
        // Hide feedback after a delay
        setTimeout(() => {
            speechFeedback.classList.remove('active');
        }, 5000);
    };
    
    recognition.onend = function() {
        console.log('Recognition ended event fired');
        speechButton.classList.remove('listening');
        speechButton.classList.add('processing');
        
        // If no input was detected
        if (queryInput.value.trim() === '') {
            speechStatus.textContent = 'No speech detected. Try again.';
            speechButton.classList.remove('processing');
            
            // Hide feedback after a delay
            setTimeout(() => {
                speechFeedback.classList.remove('active');
            }, 3000);
        } else {
            speechStatus.textContent = 'Recognition complete. You can edit before searching.';
            
            // Stop showing as "processing" after a short delay
            setTimeout(() => {
                speechButton.classList.remove('processing');
                
                // Hide feedback after additional delay
                setTimeout(() => {
                    speechFeedback.classList.remove('active');
                }, 2000);
            }, 1000);
        }
        
        // Clean up audio context
        stopAudioAnalysis();
        
        isListening = false;
    };
    
    function setupAudioAnalysis(stream) {
        try {
            // Get audio context for visualization
            if (!audioContext) {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
            }
            
            if (audioContext.state === 'suspended') {
                audioContext.resume();
            }
            
            microphone = audioContext.createMediaStreamSource(stream);
            analyzer = audioContext.createAnalyser();
            analyzer.fftSize = 256;
            analyzer.smoothingTimeConstant = 0.8;
            
            microphone.connect(analyzer);
            updateAudioVisualization();
            
            console.log('Audio analysis setup complete');
        } catch (error) {
            console.error('Audio visualization not supported:', error);
        }
    }
    
    function updateAudioVisualization() {
        if (!analyzer) return;
        
        const bufferLength = analyzer.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        
        function draw() {
            if (!isListening) return;
            
            animationFrame = requestAnimationFrame(draw);
            analyzer.getByteFrequencyData(dataArray);
            
            // Calculate average volume
            let sum = 0;
            for (let i = 0; i < bufferLength; i++) {
                sum += dataArray[i];
            }
            const average = sum / bufferLength;
            
            // Update bars
            audioBars.forEach((bar, i) => {
                // Create a wave effect by using different parts of the frequency data
                const index = Math.floor(i * (bufferLength / audioBars.length));
                const value = dataArray[index];
                
                // Scale height based on frequency value (0-255)
                const height = (value / 255) * 25 + 5; // Min 5px, max 30px
                bar.style.height = `${height}px`;
            });
        }
        
        draw();
    }
    
    function stopAudioAnalysis() {
        if (animationFrame) {
            cancelAnimationFrame(animationFrame);
        }
        
        if (microphone) {
            microphone.disconnect();
            microphone = null;
        }
        
        // Stop all tracks on the stream
        if (microphoneStream) {
            microphoneStream.getTracks().forEach(track => track.stop());
            microphoneStream = null;
        }
        
        if (audioContext && audioContext.state !== 'closed') {
            // Reset visualization bars
            audioBars.forEach(bar => {
                bar.style.height = '5px';
            });
        }
    }
}