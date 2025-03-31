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

    // Initialize speech recognition functionality if we're on the index page
    if (document.getElementById('speech-input-btn')) {
        initSpeechRecognition();
    }
    
    // Initialize text-to-speech functionality if we're on the results page
    if (document.getElementById('tts-play-btn')) {
        console.log('Initializing Text-to-Speech feature');
        initTextToSpeech();
    }
    
    // Initialize copy and share functionality on results page
    const copyBtn = document.getElementById('copy-summary');
    if (copyBtn) {
        copyBtn.addEventListener('click', function() {
            const summaryText = document.querySelector('.summary-content').innerText;
            navigator.clipboard.writeText(summaryText)
                .then(() => {
                    this.innerHTML = '<i class="bi bi-clipboard-check"></i> Copied!';
                    setTimeout(() => {
                        this.innerHTML = '<i class="bi bi-clipboard"></i> Copy Summary';
                    }, 2000);
                })
                .catch(err => {
                    console.error('Failed to copy: ', err);
                    alert('Failed to copy summary');
                });
        });
    }
    
    // Share functionality on results page
    const shareBtn = document.getElementById('share-button');
    if (shareBtn) {
        shareBtn.addEventListener('click', function() {
            if (navigator.share) {
                const summaryContent = document.querySelector('.summary-content');
                navigator.share({
                    title: document.title,
                    text: summaryContent ? summaryContent.innerText.substring(0, 100) + '...' : '',
                    url: window.location.href,
                })
                .catch(err => {
                    console.error('Share failed:', err);
                });
            } else {
                alert('Web Share API not supported in your browser');
            }
        });
    }
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

/**
 * Initialize and handle text-to-speech functionality
 */
function initTextToSpeech() {
    console.log('TTS: Initializing text-to-speech functionality');
    
    // Check if Speech Synthesis is supported
    if (!('speechSynthesis' in window)) {
        console.warn('TTS: Text-to-Speech not supported in this browser');
        const speechControls = document.querySelector('.speech-controls');
        if (speechControls) {
            speechControls.style.display = 'none';
        }
        return;
    }

    // Elements
    const playBtn = document.getElementById('tts-play-btn');
    const pauseBtn = document.getElementById('tts-pause-btn');
    const stopBtn = document.getElementById('tts-stop-btn');
    const rateSlider = document.getElementById('tts-rate');
    const rateValue = document.getElementById('tts-rate-value');
    const voiceSelect = document.getElementById('tts-voice');
    const statusElement = document.getElementById('tts-status');
    const summaryContent = document.getElementById('summary-content');

    if (!playBtn || !summaryContent) {
        console.error('TTS: Required elements not found on page');
        return;
    }

    console.log('TTS: Found all required elements');

    // Speech synthesis state
    let utterance = null;
    let isPaused = false;
    let currentSentenceIndex = 0;
    let sentences = [];
    let highlightedElements = [];

    // Populate voices dropdown
    function populateVoiceList() {
        console.log('TTS: Populating voice list');
        
        // Clear existing options (keeping the default)
        while (voiceSelect.options.length > 1) {
            voiceSelect.remove(1);
        }

        // Get available voices
        const voices = speechSynthesis.getVoices();
        console.log(`TTS: Found ${voices.length} voices`);

        // If no voices are available yet, return
        if (voices.length === 0) {
            return;
        }

        // Add each voice as an option
        voices.forEach(voice => {
            const option = document.createElement('option');
            option.textContent = `${voice.name} (${voice.lang})`;
            option.setAttribute('data-lang', voice.lang);
            option.setAttribute('data-voice-uri', voice.voiceURI);
            voiceSelect.appendChild(option);

            // Set English as default if available
            if (voice.lang.startsWith('en-') && voice.default) {
                option.selected = true;
            }
        });
    }

    // Handle the voiceschanged event
    speechSynthesis.onvoiceschanged = function() {
        console.log('TTS: Voices changed event received');
        populateVoiceList();
    };

    // Initial population of voices
    populateVoiceList();

    // Parse the summary content into sentences
    function parseSummaryContent() {
        console.log('TTS: Parsing summary content');
        
        // Get the text content of the summary
        const text = summaryContent.textContent || summaryContent.innerText;
        
        if (!text || text.trim() === '') {
            console.warn('TTS: No text content found in summary');
            return false;
        }
        
        // Split into sentences using regex
        // This regex looks for sentence endings (., !, ?) followed by a space or end of string
        const sentenceRegex = /[^.!?]+[.!?]+(?:\s|$)/g;
        const matches = text.match(sentenceRegex) || [text];
        
        sentences = matches.map(sentence => sentence.trim()).filter(s => s.length > 0);
        
        console.log(`TTS: Parsed ${sentences.length} sentences from summary`);
        return sentences.length > 0;
    }

    // Start speaking the summary
    function speak() {
        console.log('TTS: Starting speech');
        
        if (speechSynthesis.speaking) {
            console.log('TTS: Speech already in progress');
            return;
        }
        
        // Ensure we have parsed the content
        if (sentences.length === 0) {
            const success = parseSummaryContent();
            if (!success) {
                showStatus('No text content to speak');
                return;
            }
        }

        // If we've reached the end, start from beginning
        if (currentSentenceIndex >= sentences.length) {
            currentSentenceIndex = 0;
        }

        // Create a new utterance with the remaining text
        const textToSpeak = sentences.slice(currentSentenceIndex).join(' ');
        console.log(`TTS: Creating utterance with ${textToSpeak.length} characters`);
        
        utterance = new SpeechSynthesisUtterance(textToSpeak);

        // Set selected voice if available
        if (voiceSelect.selectedIndex > 0) {
            const selectedOption = voiceSelect.options[voiceSelect.selectedIndex];
            const voices = speechSynthesis.getVoices();
            for (let i = 0; i < voices.length; i++) {
                if (voices[i].voiceURI === selectedOption.getAttribute('data-voice-uri')) {
                    utterance.voice = voices[i];
                    console.log(`TTS: Selected voice: ${voices[i].name}`);
                    break;
                }
            }
        }

        // Set rate from slider
        utterance.rate = parseFloat(rateSlider.value);
        console.log(`TTS: Speaking at rate: ${utterance.rate}`);

        // Add event handlers
        utterance.onstart = () => {
            console.log('TTS: Speech started');
            updateControlsState(true);
            showStatus('Speaking...');
        };

        utterance.onend = () => {
            console.log('TTS: Speech ended');
            updateControlsState(false);
            showStatus('Finished speaking');
            currentSentenceIndex = 0;
        };

        utterance.onpause = () => {
            console.log('TTS: Speech paused');
            isPaused = true;
            showStatus('Paused');
        };

        utterance.onresume = () => {
            console.log('TTS: Speech resumed');
            isPaused = false;
            showStatus('Speaking...');
        };

        utterance.onboundary = (e) => {
            // This event may not always fire reliably across browsers
            console.log(`TTS: Boundary event: ${e.name} at ${e.charIndex}`);
        };

        utterance.onerror = (e) => {
            console.error('TTS: Speech error:', e);
            updateControlsState(false);
            showStatus(`Error: ${e.error}`);
        };

        // Start speaking
        try {
            speechSynthesis.speak(utterance);
            console.log('TTS: speechSynthesis.speak() called');
        } catch (error) {
            console.error('TTS: Error calling speak():', error);
            showStatus('Error starting speech');
        }
    }

    // Pause or resume speaking
    function togglePause() {
        if (speechSynthesis.speaking) {
            if (isPaused) {
                console.log('TTS: Resuming speech');
                speechSynthesis.resume();
                isPaused = false;
                pauseBtn.innerHTML = '<i class="bi bi-pause-fill"></i> Pause';
                showStatus('Resumed speaking');
            } else {
                console.log('TTS: Pausing speech');
                speechSynthesis.pause();
                isPaused = true;
                pauseBtn.innerHTML = '<i class="bi bi-play-fill"></i> Resume';
                showStatus('Paused');
            }
        }
    }

    // Stop speaking
    function stopSpeaking() {
        if (speechSynthesis.speaking) {
            console.log('TTS: Stopping speech');
            speechSynthesis.cancel();
            updateControlsState(false);
            showStatus('Stopped');
            currentSentenceIndex = 0;
        }
    }

    // Update the state of control buttons
    function updateControlsState(isSpeaking) {
        playBtn.disabled = isSpeaking;
        pauseBtn.disabled = !isSpeaking;
        stopBtn.disabled = !isSpeaking;
    }

    // Show status message
    function showStatus(message) {
        if (message === 'Speaking...' || message === 'Resumed speaking') {
            statusElement.innerHTML = `<span class="speaking-indicator"></span> ${message}`;
        } else {
            statusElement.textContent = message;
        }
        console.log(`TTS: Status: ${message}`);
    }

    // Event listeners
    playBtn.addEventListener('click', () => {
        console.log('TTS: Play button clicked');
        parseSummaryContent(); // Re-parse content in case it changed
        speak();
    });

    pauseBtn.addEventListener('click', () => {
        console.log('TTS: Pause button clicked');
        togglePause();
    });
    
    stopBtn.addEventListener('click', () => {
        console.log('TTS: Stop button clicked');
        stopSpeaking();
    });

    rateSlider.addEventListener('input', () => {
        rateValue.textContent = `${rateSlider.value}x`;
        console.log(`TTS: Rate changed to ${rateSlider.value}x`);
        if (utterance) {
            // To change rate while speaking, we need to restart
            const wasSpeaking = speechSynthesis.speaking && !isPaused;
            if (wasSpeaking) {
                stopSpeaking();
                utterance.rate = parseFloat(rateSlider.value);
                setTimeout(() => speak(), 100); // Small delay for cleanup
            } else {
                utterance.rate = parseFloat(rateSlider.value);
            }
        }
    });

    voiceSelect.addEventListener('change', () => {
        console.log('TTS: Voice selection changed');
        // If currently speaking, restart with new voice
        const wasSpeaking = speechSynthesis.speaking && !isPaused;
        if (wasSpeaking) {
            stopSpeaking();
            setTimeout(() => speak(), 100); // Small delay for cleanup
        }
    });

    // Fix for Chrome bug: speech synthesis stops after ~60 seconds
    // This is a known issue: https://bugs.chromium.org/p/chromium/issues/detail?id=679437
    function chromeSpeechSynthesisFix() {
        if (speechSynthesis.speaking && !isPaused) {
            console.log('TTS: Applying Chrome speech synthesis fix');
            speechSynthesis.pause();
            speechSynthesis.resume();
            setTimeout(chromeSpeechSynthesisFix, 5000);
        }
    }
    
    // Apply fix for Chrome
    if (/Chrome/.test(navigator.userAgent)) {
        console.log('TTS: Chrome detected, adding speech synthesis fix');
        setInterval(chromeSpeechSynthesisFix, 5000);
    }

    // Initialize
    showStatus('Ready to speak');
    parseSummaryContent();

    // Handle page visibility changes
    document.addEventListener('visibilitychange', () => {
        if (document.hidden && speechSynthesis.speaking) {
            // Pause when page is hidden
            if (!isPaused) {
                togglePause();
            }
        }
    });

    // Clean up when navigating away
    window.addEventListener('beforeunload', () => {
        if (speechSynthesis.speaking) {
            speechSynthesis.cancel();
        }
    });
    
    console.log('TTS: Initialization complete');
}