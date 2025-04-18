import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use('TkAgg')

class DashboardStyle:
    @staticmethod
    def configure_style():
        style = ttk.Style()
        
        # Configure light theme colors with good contrast
        style.configure('.',
            background='#FFFFFF',
            foreground='#333333',
            fieldbackground='#F5F5F5',
            font=('Segoe UI', 13))
        
        # Configure frame styles
        style.configure('Dashboard.TFrame',
            background='#FFFFFF',
            relief='flat',
            borderwidth=0,
            padding=15)
        
        style.configure('MetricCard.TFrame',
            background='#F5F5F5',
            relief='solid',
            borderwidth=1,
            padding=20,
            bordercolor='#CCCCCC')
        
        # Configure label styles
        style.configure('Dashboard.TLabel',
            background='#FFFFFF',
            foreground='#333333',
            font=('Segoe UI', 13))
        
        style.configure('MetricValue.TLabel',
            background='#F5F5F5',
            foreground='#2E7D32',
            font=('Segoe UI', 32, 'bold'))
        
        style.configure('MetricLabel.TLabel',
            background='#F5F5F5',
            foreground='#333333',
            font=('Segoe UI', 14))
        
        # Configure button styles
        style.configure('Dashboard.TButton',
            background='#E0E0E0',
            foreground='#333333',
            relief='solid',
            borderwidth=1,
            padding=(15, 8),
            font=('Segoe UI', 14),
            bordercolor='#CCCCCC')
        
        style.map('Dashboard.TButton',
            background=[('active', '#CCCCCC'), ('pressed', '#B0B0B0')],
            relief=[('pressed', 'sunken')],
            bordercolor=[('active', '#999999')])

class MetricCard(ttk.Frame):
    def __init__(self, parent, label, value="0", **kwargs):
        super().__init__(parent, style='MetricCard.TFrame', **kwargs)
        
        # Value label
        self.value_label = ttk.Label(self, 
            style='MetricValue.TLabel',
            text=value)
        self.value_label.pack(pady=(10,0))
        
        # Metric label
        self.label = ttk.Label(self,
            style='MetricLabel.TLabel',
            text=label)
        self.label.pack(pady=(0,10))
    
    def update_value(self, value):
        self.value_label.configure(text=value)

class DetectionChart:
    def __init__(self, parent, title="Detection Trend", max_points=100):
        self.max_points = max_points
        
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(3,2), facecolor='#FFFFFF')
        self.ax.set_facecolor('#F5F5F5')
        
        # Style the chart
        self.ax.grid(True, color='#CCCCCC', linestyle='--', alpha=0.9)
        self.ax.tick_params(colors='#333333', labelsize=10)
        self.ax.set_title(title, color='#333333', pad=30, fontsize=18)
        
        for spine in self.ax.spines.values():
            spine.set_color('#CCCCCC')
        
        # Initialize data
        self.x_data = []
        self.y_data = []
        self.line, = self.ax.plot([], [], color='#2E7D32', linewidth=2)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
    
    def update(self, value):
        self.x_data.append(len(self.x_data))
        self.y_data.append(value)
        
        # Limit data points
        if len(self.x_data) > self.max_points:
            self.x_data = self.x_data[-self.max_points:]
            self.y_data = self.y_data[-self.max_points:]
        
        # Update line data
        self.line.set_data(self.x_data, self.y_data)
        
        # Adjust axes limits
        self.ax.relim()
        self.ax.autoscale_view()
        
        # Redraw canvas
        self.canvas.draw()