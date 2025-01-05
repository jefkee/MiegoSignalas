import numpy as np

class ReportGenerator:
    def __init__(self):
        self.quality_thresholds = {
            'excellent': 90,
            'good': 75,
            'fair': 60,
            'poor': 0
        }
    
    def generate_recommendations(self, analysis):
        """Generate sleep recommendations based on analysis results"""
        recommendations = []
        
        # Get sleep metrics
        quality_score = analysis.get('quality_score', 0)
        stages = analysis.get('stages', [])
        if not stages:
            return ["Unable to analyze sleep patterns. Please ensure recording quality."]
        
        # Calculate stage percentages
        total_epochs = len(stages)
        stage_counts = np.bincount(stages, minlength=5)
        wake_percent = (stage_counts[0] / total_epochs) * 100
        rem_percent = (stage_counts[4] / total_epochs) * 100
        deep_sleep_percent = (stage_counts[3] / total_epochs) * 100

        # Base recommendations
        recommendations.append("Maintain a consistent sleep schedule")
        recommendations.append("Ensure your bedroom is dark and quiet")

        # Sleep efficiency based recommendations
        sleep_efficiency = float(analysis.get('timing', {}).get('sleep_efficiency', '0').strip('%'))
        if sleep_efficiency < 85:
            recommendations.append("Consider sleep restriction therapy to improve sleep efficiency")
            recommendations.append("Avoid staying in bed when unable to sleep")

        # Sleep onset recommendations
        sleep_onset = analysis.get('timing', {}).get('sleep_onset', '')
        if 'After 00:30' in sleep_onset:  # If sleep onset > 30 minutes
            recommendations.append("Practice relaxation techniques before bedtime")
            recommendations.append("Avoid screens at least 1 hour before bed")
            recommendations.append("Consider cognitive behavioral therapy for insomnia")

        # REM sleep recommendations
        if rem_percent < 20:
            recommendations.append("Avoid alcohol before bedtime as it can suppress REM sleep")
            recommendations.append("Maintain a regular sleep schedule to promote REM sleep")
        elif rem_percent > 30:
            recommendations.append("Excessive REM sleep detected - consider consulting a sleep specialist")

        # Deep sleep recommendations
        if deep_sleep_percent < 15:
            recommendations.append("Exercise regularly, but not too close to bedtime")
            recommendations.append("Consider temperature regulation - slightly cool room temperature can promote deep sleep")
        
        # Wake time recommendations
        if wake_percent > 15:
            recommendations.append("Avoid caffeine in the evening")
            recommendations.append("Consider keeping a sleep diary to identify wake triggers")

        # Quality score based recommendations
        if quality_score < 70:
            recommendations.append("Consider consulting a sleep specialist")
            recommendations.append("Keep a detailed sleep diary for at least two weeks")

        return recommendations
    
    def _generate_summary(self, results):
        return {
            'sleep_quality_score': self._calculate_sleep_quality(results),
            'main_findings': self._identify_main_findings(results),
            'key_metrics': self._extract_key_metrics(results)
        } 
    
    def _calculate_sleep_quality(self, results):
        """Calculate overall sleep quality score"""
        metrics = {
            'sleep_efficiency': self._calculate_sleep_efficiency(results),
            'sleep_latency': self._calculate_sleep_latency(results),
            'awakenings': self._count_awakenings(results),
            'rem_percentage': self._calculate_rem_percentage(results)
        }
        
        # Weighted average of metrics
        weights = {'sleep_efficiency': 0.4, 'sleep_latency': 0.2, 
                  'awakenings': 0.2, 'rem_percentage': 0.2}
        
        score = sum(metrics[key] * weights[key] for key in weights)
        return round(score, 2)
    
    def _identify_main_findings(self, results):
        findings = []
        
        # Check sleep efficiency
        efficiency = self._calculate_sleep_efficiency(results)
        if efficiency < 85:
            findings.append("Sleep efficiency is below recommended levels")
            
        # Check REM sleep
        rem_pct = self._calculate_rem_percentage(results)
        if rem_pct < 20:
            findings.append("REM sleep percentage is lower than normal")
            
        return findings
    
    def _calculate_sleep_efficiency(self, results):
        """Calculate sleep efficiency (time asleep / time in bed)"""
        if not results.get('stages'):
            return 0
            
        total_epochs = len(results['stages'])
        wake_epochs = sum(1 for stage in results['stages'] if stage == 0)  # 0 = Wake
        
        time_in_bed = total_epochs * 30  # 30-second epochs
        time_asleep = (total_epochs - wake_epochs) * 30
        
        return (time_asleep / time_in_bed) * 100 if time_in_bed > 0 else 0
    
    def _calculate_sleep_latency(self, results):
        """Calculate time to fall asleep"""
        stages = results['stages']
        for i, stage in enumerate(stages):
            if stage > 0:  # First non-wake stage
                return i * 30  # Time in seconds
        return len(stages) * 30
    
    def _count_awakenings(self, results):
        """Count number of awakenings during sleep"""
        stages = results['stages']
        awakenings = 0
        for i in range(1, len(stages)):
            if stages[i] == 0 and stages[i-1] != 0:  # Transition to wake
                awakenings += 1
        return awakenings
    
    def _calculate_rem_percentage(self, results):
        """Calculate percentage of REM sleep"""
        stages = results['stages']
        rem_epochs = sum(1 for stage in stages if stage == 4)  # 4 = REM
        return (rem_epochs / len(stages)) * 100 if stages else 0