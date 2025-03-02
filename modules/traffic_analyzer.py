class TrafficAnalyzer:
    def __init__(self):
        self.vehicle_counts = {
            'car': 0,
            'truck': 0,
            'bus': 0,
            'motorcycle': 0,
            'bicycle': 0
        }
        self.traffic_density = 0
        self.violation_counts = {
            'wrong_way': 0,
            'speed': 0,
            'red_light': 0
        }
        
    def analyze_traffic_flow(self, detections, frame):
        """Analyze traffic flow patterns"""
        # Calculate vehicle density
        total_vehicles = len(detections)
        frame_area = frame.shape[0] * frame.shape[1]
        self.traffic_density = total_vehicles / frame_area
        
        # Analyze vehicle trajectories
        vehicle_trajectories = self.track_vehicle_movements(detections)
        
        # Detect traffic violations
        violations = self.detect_violations(vehicle_trajectories)
        
        return {
            'density': self.traffic_density,
            'vehicle_counts': self.vehicle_counts,
            'violations': violations,
            'flow_rate': self.calculate_flow_rate(detections)
        }
        
    def detect_violations(self, trajectories):
        """Detect traffic violations"""
        violations = []
        for trajectory in trajectories:
            # Speed violation detection
            if self.check_speed_violation(trajectory):
                violations.append({
                    'type': 'speed',
                    'vehicle_id': trajectory.id,
                    'speed': trajectory.speed
                })
            
            # Wrong way detection
            if self.check_wrong_way(trajectory):
                violations.append({
                    'type': 'wrong_way',
                    'vehicle_id': trajectory.id
                })
                
        return violations 