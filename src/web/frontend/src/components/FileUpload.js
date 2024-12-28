import React, { useState } from 'react';
import axios from 'axios';

const FileUpload = () => {
    const [file, setFile] = useState(null);
    const [analysis, setAnalysis] = useState(null);
    const [loading, setLoading] = useState(false);
    
    const handleUpload = async () => {
        if (!file) return;
        
        setLoading(true);
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await axios.post('/api/analyze', formData);
            setAnalysis(response.data);
        } catch (error) {
            console.error('Upload failed:', error);
        } finally {
            setLoading(false);
        }
    };
    
    return (
        <div>
            <input 
                type="file" 
                accept=".edf"
                onChange={(e) => setFile(e.target.files[0])} 
            />
            <button onClick={handleUpload} disabled={!file || loading}>
                Analyze Recording
            </button>
            
            {analysis && (
                <div>
                    <h2>Analysis Results</h2>
                    {/* Display results */}
                </div>
            )}
        </div>
    );
};

export default FileUpload; 