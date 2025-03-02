import logging

logger = logging.getLogger(__name__)

class TrafficSignalController:
    def __init__(self):
        self.signals = {
            'signal_1': {
                'phase': 'RED',
                'timing': 30,
                'location': 'Main Street'
            },
            'signal_2': {
                'phase': 'GREEN',
                'timing': 45,
                'location': 'Cross Street'
            }
        }
        self.phases = ['RED', 'GREEN', 'YELLOW']
        self.current_phase = 0
        self.phase_timings = {
            'RED': 30,
            'GREEN': 45,
            'YELLOW': 5
        }
        self.min_green_time = 30
        self.max_green_time = 120
        
    def get_current_status(self):
        """Get current status of all signals"""
        return self.signals
        
    def optimize_signal_timing(self, traffic_data):
        """Optimize traffic signal timings based on real-time data"""
        try:
            vehicle_density = traffic_data['density']
            total_vehicles = traffic_data['total_vehicles']
            
            # Calculate optimal timing based on traffic density
            if vehicle_density > 0.5:  # High traffic
                optimal_timing = min(self.max_green_time, 
                                  int(60 + (vehicle_density * 60)))
            elif vehicle_density > 0.2:  # Medium traffic
                optimal_timing = int(45 + (vehicle_density * 30))
            else:  # Low traffic
                optimal_timing = max(self.min_green_time, 
                                  int(30 + (vehicle_density * 15)))
            
            # Update signal timings
            for signal_id in self.signals:
                if self.signals[signal_id]['phase'] == 'GREEN':
                    self.signals[signal_id]['timing'] = optimal_timing
            
            return {
                'phase': self.current_phase,
                'timing': optimal_timing,
                'density': vehicle_density,
                'total_vehicles': total_vehicles
            }
            
        except Exception as e:
            logger.error(f"Error optimizing signal timing: {e}")
            return None
    
    def update_signal_status(self, emergency_vehicle=False):
        """Update signal status based on current conditions"""
        try:
            for signal_id, signal in self.signals.items():
                # Handle emergency vehicle priority
                if emergency_vehicle:
                    signal['phase'] = 'GREEN'
                    signal['timing'] = 60
                    continue
                
                # Normal signal cycling
                current_phase = signal['phase']
                current_timing = signal['timing']
                
                if current_timing <= 0:
                    # Cycle through phases
                    if current_phase == 'RED':
                        signal['phase'] = 'GREEN'
                        signal['timing'] = self.phase_timings['GREEN']
                    elif current_phase == 'GREEN':
                        signal['phase'] = 'YELLOW'
                        signal['timing'] = self.phase_timings['YELLOW']
                    else:  # YELLOW
                        signal['phase'] = 'RED'
                        signal['timing'] = self.phase_timings['RED']
                else:
                    signal['timing'] -= 1
                    
            return self.signals
            
        except Exception as e:
            logger.error(f"Error updating signal status: {e}")
            return None 