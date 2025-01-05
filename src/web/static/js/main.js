document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('upload-form');
    const psgFile = document.getElementById('psg_file');
    const psgValidation = document.getElementById('psg-validation');
    const loading = document.getElementById('loading');
    const results = document.getElementById('analysis-results');

    // File validation
    psgFile.addEventListener('change', function() {
        const file = this.files[0];
        if (file) {
            if (file.name.endsWith('-PSG.edf')) {
                psgValidation.textContent = '✓ Valid PSG file';
                psgValidation.className = 'validation-message success';
            } else {
                psgValidation.textContent = '✗ PSG file must end with -PSG.edf';
                psgValidation.className = 'validation-message error';
            }
        }
    });

    // Form submission
    form.addEventListener('submit', async function(e) {
        e.preventDefault();

        const formData = new FormData();
        formData.append('psg_file', psgFile.files[0]);

        try {
            loading.style.display = 'block';
            results.style.display = 'none';

            const response = await fetch('/api/analyze', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Analysis failed');
            }

            const data = await response.json();
            displayResults(data);
        } catch (error) {
            console.error('Error:', error);
            alert('Error analyzing sleep recording. Please try again.');
        } finally {
            loading.style.display = 'none';
        }
    });

    function displayResults(data) {
        // Display sleep quality score and interpretation
        const qualityValue = document.getElementById('quality-value');
        const qualityInterpretation = document.getElementById('quality-interpretation');
        qualityValue.textContent = `Score: ${data.sleep_quality.score}%`;
        qualityInterpretation.textContent = data.sleep_quality.interpretation;

        // Display visualizations
        if (data.visualizations) {
            const charts = {
                'distribution-chart': data.visualizations.distribution,
                'hypnogram-chart': data.visualizations.hypnogram,
                'transitions-chart': data.visualizations.transitions,
                'hourly-chart': data.visualizations.hourly_distribution,
                'spectral-chart': data.visualizations.spectral_density
            };

            for (const [id, src] of Object.entries(charts)) {
                const img = document.getElementById(id);
                if (img && src) {
                    img.src = src;
                    img.style.display = 'block';
                }
            }
        }

        // Display recommendations
        const recommendationsList = document.getElementById('recommendations-list');
        recommendationsList.innerHTML = data.recommendations
            .map(rec => `<li>${rec}</li>`)
            .join('');

        // Display recording information
        if (data.recording_info) {
            const info = data.recording_info;
            
            // Format start time
            let startTime = "Not available";
            if (info.start_time) {
                try {
                    startTime = new Date(info.start_time).toLocaleString();
                } catch (e) {
                    console.error('Error formatting start time:', e);
                }
            }
            
            document.getElementById('recording-start').textContent = startTime;
            document.getElementById('total-duration').textContent = formatDuration(info.duration);
            document.getElementById('total-sleep-time').textContent = formatDuration(info.sleep_time);
            document.getElementById('sleep-onset').textContent = formatDuration(info.sleep_onset);
            document.getElementById('wake-time').textContent = formatDuration(info.wake_time);
            document.getElementById('sleep-efficiency').textContent = `${info.efficiency.toFixed(1)}%`;
        }

        results.style.display = 'block';
    }

    function formatDuration(minutes) {
        if (typeof minutes !== 'number' || isNaN(minutes)) {
            return 'N/A';
        }
        const hours = Math.floor(minutes / 60);
        const mins = Math.round(minutes % 60);
        return `${hours}:${mins.toString().padStart(2, '0')}`;
    }
}); 