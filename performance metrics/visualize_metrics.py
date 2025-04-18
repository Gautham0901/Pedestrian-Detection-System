import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def save_individual_plots(metrics, colors):
    # Inference Time Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(metrics['inference_times'], bins=30, color=colors[0], alpha=0.7, edgecolor='white')
    mean_time = np.mean(metrics['inference_times'])
    plt.axvline(mean_time, color='red', linestyle='--', label=f'Mean: {mean_time:.3f}s')
    plt.title('Inference Time Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig('inference_time_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # FPS Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(metrics['fps'], bins=30, color=colors[1], alpha=0.7, edgecolor='white')
    mean_fps = np.mean(metrics['fps'])
    plt.axvline(mean_fps, color='red', linestyle='--', label=f'Mean: {mean_fps:.2f} FPS')
    plt.title('FPS Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Frames per Second')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig('fps_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Memory Usage
    plt.figure(figsize=(10, 6))
    frame_numbers = range(len(metrics['memory_usage']))
    memory_rolling_avg = np.convolve(metrics['memory_usage'], np.ones(10)/10, mode='valid')
    plt.plot(frame_numbers, metrics['memory_usage'], color=colors[2], alpha=0.3, label='Raw')
    plt.plot(range(len(memory_rolling_avg)), memory_rolling_avg, color=colors[2], linewidth=2, label='Rolling Average')
    plt.title('Memory Usage Over Time', fontsize=12, fontweight='bold')
    plt.xlabel('Frame Number')
    plt.ylabel('Memory Usage (MB)')
    plt.legend()
    plt.savefig('memory_usage_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Detections per Frame
    plt.figure(figsize=(10, 6))
    frame_nums = range(len(metrics['detections_per_frame']))
    detections_rolling_avg = np.convolve(metrics['detections_per_frame'], np.ones(5)/5, mode='valid')
    plt.plot(frame_nums, metrics['detections_per_frame'], color=colors[3], alpha=0.3, label='Raw Detections')
    plt.plot(range(len(detections_rolling_avg)), detections_rolling_avg, color=colors[3], linewidth=2, label='Rolling Average')
    plt.fill_between(frame_nums, metrics['detections_per_frame'], alpha=0.2, color=colors[3])
    plt.title('Detections per Frame', fontsize=12, fontweight='bold')
    plt.xlabel('Frame Number')
    plt.ylabel('Number of Detections')
    plt.legend()
    plt.savefig('detections_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics(metrics_file):
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    metrics = data['detailed_metrics']
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f1c40f']
    
    # Save individual plots
    save_individual_plots(metrics, colors)
    
    # Create combined plot
    plt.style.use('ggplot')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    
    # Inference Time Distribution with enhanced colors
    sns.histplot(metrics['inference_times'], ax=axes[0,0], kde=True, bins=30,
                color=colors[0], edgecolor='white')
    mean_time = np.mean(metrics['inference_times'])
    axes[0,0].axvline(mean_time, color='red', linestyle='--', 
                      label=f'Mean: {mean_time:.3f}s')
    axes[0,0].set_title('Inference Time Distribution', fontsize=12, fontweight='bold')
    axes[0,0].legend()
    
    # FPS Distribution with gradient colors
    sns.histplot(metrics['fps'], ax=axes[0,1], kde=True, bins=30,
                color=colors[1], edgecolor='white')
    mean_fps = np.mean(metrics['fps'])
    axes[0,1].axvline(mean_fps, color='red', linestyle='--', 
                      label=f'Mean: {mean_fps:.2f} FPS')
    axes[0,1].set_title('FPS Distribution', fontsize=12, fontweight='bold')
    axes[0,1].legend()
    
    # Memory Usage with dual-line plot
    frame_numbers = range(len(metrics['memory_usage']))
    memory_rolling_avg = np.convolve(metrics['memory_usage'], 
                                   np.ones(10)/10, mode='valid')
    axes[1,0].plot(frame_numbers, metrics['memory_usage'], 
                   color=colors[2], alpha=0.3, label='Raw')
    axes[1,0].plot(range(len(memory_rolling_avg)), memory_rolling_avg,
                   color=colors[2], label='Rolling Average', linewidth=2)
    axes[1,0].set_title('Memory Usage Over Time', fontsize=12, fontweight='bold')
    axes[1,0].legend()
    
    # Add detections per frame plot with improved visualization
    if 'detections_per_frame' in metrics:
        frame_nums = range(len(metrics['detections_per_frame']))
        # Add rolling average for smoother visualization
        detections_rolling_avg = np.convolve(metrics['detections_per_frame'], 
                                           np.ones(5)/5, mode='valid')
        
        # Plot raw data
        axes[1,1].plot(frame_nums, metrics['detections_per_frame'],
                      color=colors[3], alpha=0.3, label='Raw Detections')
        
        # Plot rolling average
        axes[1,1].plot(range(len(detections_rolling_avg)), detections_rolling_avg,
                      color=colors[3], linewidth=2, label='Rolling Average')
        
        # Add fill between for better visualization
        axes[1,1].fill_between(frame_nums, metrics['detections_per_frame'],
                             alpha=0.2, color=colors[3])
        
        # Set y-axis to start from 0
        axes[1,1].set_ylim(bottom=0)
        axes[1,1].legend()
    
    axes[1,1].set_title('Detections per Frame', fontsize=12, fontweight='bold')
    axes[1,1].set_xlabel('Frame Number')
    axes[1,1].set_ylabel('Number of Detections')
    
    # Add comprehensive metrics summary
    # Update the metrics text positioning and style
    metrics_text = (
        f"Performance Summary:\n"
        f"Total Frames: {metrics['frames_processed']}\n"
        f"Average FPS: {mean_fps:.2f}\n"
        f"Average Inference Time: {mean_time:.3f}s\n"
        f"Peak Memory Usage: {max(metrics['memory_usage']):.1f} MB\n"
        f"Average Memory Usage: {np.mean(metrics['memory_usage']):.1f} MB\n"
        f"Total Detections: {metrics.get('total_detections', 0)}\n"
        f"Avg Detections/Frame: {metrics.get('total_detections', 0)/metrics['frames_processed']:.2f}\n"
        f"Peak Detections in Frame: {max(metrics.get('detections_per_frame', [0]))}\n"
        f"Processing Efficiency: {(1/mean_time)*100:.1f}%"
    )
    
    # Create a text box with better positioning and styling
    plt.figtext(0.02, 0.15, metrics_text, fontsize=10,
                bbox=dict(facecolor='white', 
                         alpha=0.9,
                         edgecolor='#2ecc71',
                         boxstyle='round,pad=1',
                         linewidth=2),
                verticalalignment='bottom',
                horizontalalignment='left')
    
    # Add title to the entire figure
    fig.suptitle('Model Performance Metrics', fontsize=14, fontweight='bold')
    
    # Save with high quality
    plt.savefig('performance_metrics.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

if __name__ == "__main__":
    plot_metrics('metrics_report.json')