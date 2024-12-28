import React from 'react';
import Plot from 'react-plotly.js';

const AnalysisResults = ({ analysis }) => {
    if (!analysis) return null;
    
    return (
        <div className="analysis-results">
            <div className="quality-score">
                <h3>Sleep Quality Score</h3>
                <div className="score">{analysis.sleep_quality.score}%</div>
                <div className="interpretation">
                    {analysis.sleep_quality.interpretation}
                </div>
            </div>
            
            <div className="stage-distribution">
                <h3>Sleep Stages</h3>
                <Plot
                    data={analysis.sleep_stages.visualization}
                    layout={{ width: 720, height: 480 }}
                />
            </div>
            
            <div className="recommendations">
                <h3>Recommendations</h3>
                <ul>
                    {analysis.recommendations.map((rec, index) => (
                        <li key={index}>{rec}</li>
                    ))}
                </ul>
            </div>
        </div>
    );
};

export default AnalysisResults; 