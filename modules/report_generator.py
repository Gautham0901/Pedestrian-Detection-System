import os
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns

class ReportGenerator:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.report_dir = self.base_dir / 'reports'
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_report(self, job_id, video_stats, detections_data, analytics_data, xai_frames):
        """Generate a comprehensive PDF report with detection results and insights"""
        pdf = FPDF()
        pdf.add_page()
        
        # Title
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Pedestrian Detection Analysis Report', ln=True, align='C')
        pdf.ln(10)
        
        # Video Information
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Video Analysis Summary', ln=True)
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 10, f'Analysis Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', ln=True)
        pdf.cell(0, 10, f'Total Frames: {video_stats["total_frames"]}', ln=True)
        pdf.cell(0, 10, f'Processing FPS: {video_stats["fps"]:.2f}', ln=True)
        pdf.ln(10)
        
        # Detection Statistics
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Detection Statistics', ln=True)
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 10, f'Total Detections: {video_stats["total_detections"]}', ln=True)
        pdf.cell(0, 10, f'Average Confidence: {video_stats["avg_confidence"]:.2f}', ln=True)
        pdf.ln(10)
        
        # XAI Visualizations
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Model Explanation Visualizations', ln=True)
        pdf.set_font('Arial', '', 10)
        
        for i, frame in enumerate(xai_frames):
            frame_path = self.report_dir / f'frame_{job_id}_{i}.jpg'
            cv2.imwrite(str(frame_path), frame)
            pdf.image(str(frame_path), x=10, w=190)
            pdf.cell(0, 10, f'Frame {i+1}: XAI visualization showing model attention regions', ln=True)
            pdf.ln(5)
            os.remove(frame_path)
        
        # Traffic Management Insights
        pdf.add_page()
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Traffic Management Insights', ln=True)
        pdf.set_font('Arial', '', 10)
        
        # Generate and add plots
        self._add_pedestrian_density_plot(pdf, detections_data, job_id)
        self._add_confidence_distribution_plot(pdf, detections_data, job_id)
        
        # Recommendations
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Recommendations', ln=True)
        pdf.set_font('Arial', '', 10)
        
        recommendations = self._generate_recommendations(video_stats, analytics_data)
        for rec in recommendations:
            pdf.multi_cell(0, 10, f'• {rec}')
        
        # Save report
        report_path = self.report_dir / f'report_{job_id}.pdf'
        pdf.output(str(report_path))
        return report_path
    
    def _add_pedestrian_density_plot(self, pdf, detections_data, job_id):
        """Add pedestrian density over time plot"""
        plt.figure(figsize=(10, 6))
        frame_counts = detections_data.groupby('frame').size()
        plt.plot(frame_counts.index, frame_counts.values)
        plt.title('Pedestrian Density Over Time')
        plt.xlabel('Frame Number')
        plt.ylabel('Number of Pedestrians')
        
        plot_path = self.report_dir / f'density_{job_id}.png'
        plt.savefig(plot_path)
        plt.close()
        
        pdf.image(str(plot_path), x=10, w=190)
        pdf.cell(0, 10, 'Pedestrian density variation throughout the video', ln=True)
        pdf.ln(5)
        os.remove(plot_path)
    
    def _add_confidence_distribution_plot(self, pdf, detections_data, job_id):
        """Add detection confidence distribution plot"""
        plt.figure(figsize=(10, 6))
        sns.histplot(data=detections_data, x='confidence', bins=20)
        plt.title('Detection Confidence Distribution')
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        
        plot_path = self.report_dir / f'confidence_{job_id}.png'
        plt.savefig(plot_path)
        plt.close()
        
        pdf.image(str(plot_path), x=10, w=190)
        pdf.cell(0, 10, 'Distribution of detection confidence scores', ln=True)
        pdf.ln(5)
        os.remove(plot_path)
    
    def _generate_recommendations(self, video_stats, analytics_data):
        """Generate traffic management recommendations based on analysis"""
        recommendations = []
        
        # Density-based recommendations
        avg_density = video_stats['total_detections'] / video_stats['total_frames']
        if avg_density > 5:
            recommendations.append('High pedestrian density detected. Consider implementing crowd management measures.')
        
        # Performance-based recommendations
        if video_stats['fps'] < 15:
            recommendations.append('Processing performance could be improved. Consider optimizing detection parameters or upgrading hardware.')
        
        # Confidence-based recommendations
        if video_stats['avg_confidence'] < 0.5:
            recommendations.append('Detection confidence is relatively low. Consider adjusting lighting conditions or camera positioning.')
        
        # Add general recommendations
        recommendations.extend([
            'Regular monitoring of peak hours can help optimize resource allocation.',
            'Consider implementing automated alerts for unusual pedestrian density patterns.',
            'Periodic recalibration of detection models can help maintain optimal performance.'
        ])
        
        return recommendations