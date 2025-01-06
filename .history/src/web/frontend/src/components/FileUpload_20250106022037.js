import React, { useState } from 'react';
import axios from 'axios';
import './FileUpload.css';

const FileUpload = ({ onAnalysisStart, onAnalysisComplete }) => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [uploadProgress, setUploadProgress] = useState(0);

    const handleFileSelect = (event) => {
        setSelectedFile(event.target.files[0]);
    };

    const handleUpload = async (event) => {
        event.preventDefault();
        
        if (!selectedFile) {
            alert('Please select a file first!');
            return;
        }

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            onAnalysisStart();
            setUploadProgress(0);

            const response = await axios.post('http://localhost:5000/api/analyze', 
                formData,
                {
                    headers: {
                        'Content-Type': 'multipart/form-data',
                    },
                    onUploadProgress: (progressEvent) => {
                        const percentCompleted = Math.round(
                            (progressEvent.loaded * 100) / progressEvent.total
                        );
                        setUploadProgress(percentCompleted);
                    },
                }
            );

            if (response.data) {
                onAnalysisComplete(response.data);
            } else {
                throw new Error('No data received from server');
            }
        } catch (error) {
            console.error('Upload error:', error);
            onAnalysisComplete({ 
                error: error.response?.data?.error || 'Error uploading file' 
            });
        }

        setSelectedFile(null);
        setUploadProgress(0);
    };

    return (
        <div className="file-upload">
            <form onSubmit={handleUpload}>
                <div className="file-input-container">
                    <input
                        type="file"
                        onChange={handleFileSelect}
                        accept=".edf"
                        className="file-input"
                    />
                    <button 
                        type="submit" 
                        disabled={!selectedFile}
                        className="upload-button"
                    >
                        Analyze Sleep Data
                    </button>
                </div>
                {uploadProgress > 0 && uploadProgress < 100 && (
                    <div className="progress-bar">
                        <div 
                            className="progress-bar-fill" 
                            style={{ width: `${uploadProgress}%` }}
                        />
                    </div>
                )}
            </form>
        </div>
    );
};

export default FileUpload; 