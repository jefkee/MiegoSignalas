import React from 'react';
import Plot from 'react-plotly.js';

const AnalysisResults = ({ analysis }) => {
    if (!analysis) return null;
    
    // Paruošiame duomenis Plotly grafikams
    const stageDistributionData = {
        values: analysis.sleep_stages.distribution_values,
        labels: ['Wake', 'N1', 'N2', 'N3', 'REM'],
        type: 'pie',
        name: 'Sleep Stages'
    };

    const hypnogramData = {
        x: analysis.sleep_stages.time_points,
        y: analysis.sleep_stages.stages,
        type: 'scatter',
        mode: 'lines',
        name: 'Sleep Stages'
    };

    const transitionMatrixData = {
        z: analysis.sleep_stages.transitions,
        x: ['Wake', 'N1', 'N2', 'N3', 'REM'],
        y: ['Wake', 'N1', 'N2', 'N3', 'REM'],
        type: 'heatmap',
        colorscale: 'Viridis'
    };

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
                <h3>Sleep Stage Distribution</h3>
                <Plot
                    data={[stageDistributionData]}
                    layout={{
                        width: 720,
                        height: 480,
                        title: 'Sleep Stage Distribution',
                        showlegend: true
                    }}
                    config={{ responsive: true }}
                />
            </div>

            <div className="hypnogram">
                <h3>Hypnogram</h3>
                <Plot
                    data={[hypnogramData]}
                    layout={{
                        width: 720,
                        height: 480,
                        title: 'Sleep Stages Throughout Night',
                        yaxis: {
                            ticktext: ['Wake', 'N1', 'N2', 'N3', 'REM'],
                            tickvals: [0, 1, 2, 3, 4]
                        },
                        xaxis: { title: 'Time' }
                    }}
                    config={{ responsive: true }}
                />
            </div>

            <div className="transition-matrix">
                <h3>Stage Transitions</h3>
                <Plot
                    data={[transitionMatrixData]}
                    layout={{
                        width: 720,
                        height: 480,
                        title: 'Sleep Stage Transitions',
                        xaxis: { title: 'To Stage' },
                        yaxis: { title: 'From Stage' }
                    }}
                    config={{ responsive: true }}
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