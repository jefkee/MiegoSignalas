document.getElementById('upload-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const fileInput = document.getElementById('psg_file');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select a file');
        return;
    }
    
    if (!file.name.endsWith('-PSG.edf')) {
        alert('File must end with -PSG.edf');
        return;
    }
    
    // Show loading
    document.getElementById('loading').style.display = 'block';
    document.getElementById('results').style.display = 'none';
    
    // Create form data
    const formData = new FormData();
    formData.append('psg_file', file);
    
    try {
        // Send file to server
        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Error analyzing file');
        }
        
        const results = await response.json();
        
        // Display results
        displayResults(results);
        
    } catch (error) {
        console.error('Error:', error);
        alert(error.message || 'Error analyzing file');
    } finally {
        document.getElementById('loading').style.display = 'none';
    }
});

function displayResults(results) {
    const resultsDiv = document.getElementById('results');
    
    // Display quality score
    const qualityDiv = resultsDiv.querySelector('.quality-score');
    qualityDiv.innerHTML = `
        <h3>Sleep Quality Score: ${results.sleep_quality.score}%</h3>
        <div class="quality-details">
            <p>Wake: ${results.sleep_quality.details.wake_percent.toFixed(1)}%</p>
            <p>N1: ${results.sleep_quality.details.n1_percent.toFixed(1)}%</p>
            <p>N2: ${results.sleep_quality.details.n2_percent.toFixed(1)}%</p>
            <p>N3: ${results.sleep_quality.details.n3_percent.toFixed(1)}%</p>
            <p>REM: ${results.sleep_quality.details.rem_percent.toFixed(1)}%</p>
        </div>
    `;
    
    // Display sleep stages visualization
    const stagesDiv = resultsDiv.querySelector('.sleep-stages');
    stagesDiv.innerHTML = '<h3>Sleep Stages</h3>';
    
    // Create canvas for visualization
    const canvas = document.createElement('canvas');
    stagesDiv.appendChild(canvas);
    
    // Plot sleep stages
    plotSleepStages(canvas, results.stages, results.times);
    
    // Display recommendations
    const recsDiv = resultsDiv.querySelector('.recommendations');
    recsDiv.innerHTML = '<h3>Recommendations</h3><ul>' +
        results.recommendations.map(rec => `<li>${rec}</li>`).join('') +
        '</ul>';
    
    resultsDiv.style.display = 'block';
}

function plotSleepStages(canvas, stages, times) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width = 800;
    const height = canvas.height = 200;
    
    // Clear canvas
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, width, height);
    
    // Set up scales
    const xScale = width / times[times.length - 1];
    const yScale = height / 5;
    
    // Stage colors
    const colors = [
        '#ff9999',  // Wake - light red
        '#99ff99',  // N1 - light green
        '#9999ff',  // N2 - light blue
        '#ffff99',  // N3 - light yellow
        '#ff99ff'   // REM - light purple
    ];
    
    // Plot stages
    ctx.beginPath();
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 2;
    
    for (let i = 0; i < stages.length; i++) {
        const x = times[i] * xScale;
        const y = stages[i] * yScale;
        
        // Draw point
        ctx.fillStyle = colors[stages[i]];
        ctx.fillRect(x, y, 3, 3);
        
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }
    
    ctx.stroke();
    
    // Add labels
    ctx.fillStyle = '#000';
    ctx.font = '12px Arial';
    ctx.textAlign = 'right';
    
    const labels = ['Wake', 'N1', 'N2', 'N3', 'REM'];
    labels.forEach((label, i) => {
        ctx.fillText(label, 40, i * yScale + 15);
    });
} 