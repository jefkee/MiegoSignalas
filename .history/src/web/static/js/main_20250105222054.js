document.getElementById('upload-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const fileInput = document.getElementById('edf-file');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select a file');
        return;
    }
    
    // Show loading
    document.getElementById('loading').style.display = 'block';
    document.getElementById('results').style.display = 'none';
    
    // Create form data
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        // Send file to server
        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });
        
        const results = await response.json();
        
        // Display results
        displayResults(results);
        
    } catch (error) {
        console.error('Error:', error);
        alert('Error analyzing file');
    } finally {
        document.getElementById('loading').style.display = 'none';
    }
});

function displayResults(results) {
    const resultsDiv = document.getElementById('results');
    
    // Display quality score
    resultsDiv.querySelector('.quality-score').textContent = 
        `Sleep Quality Score: ${results.sleep_quality.score}%`;
    
    // Display sleep stages
    const stagesDiv = resultsDiv.querySelector('.sleep-stages');
    stagesDiv.innerHTML = '<h3>Sleep Stages</h3>';
    // Add visualization here
    
    // Display recommendations
    const recsDiv = resultsDiv.querySelector('.recommendations');
    recsDiv.innerHTML = '<h3>Recommendations</h3><ul>' +
        results.recommendations.map(rec => `<li>${rec}</li>`).join('') +
        '</ul>';
    
    resultsDiv.style.display = 'block';
} 