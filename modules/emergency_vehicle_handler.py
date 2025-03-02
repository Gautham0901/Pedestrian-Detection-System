class EmergencyVehicleHandler:
    def __init__(self):
        self.emergency_vehicles = set()
        self.priority_routes = {}
        
    def detect_emergency_vehicle(self, frame, detections):
        """Detect emergency vehicles and provide priority routing"""
        emergency_vehicles = self.identify_emergency_vehicles(detections)
        
        if emergency_vehicles:
            # Calculate optimal route
            priority_route = self.calculate_priority_route(
                emergency_vehicles[0],
                self.get_current_traffic_state()
            )
            
            # Adjust signals for priority passage
            self.adjust_signals_for_emergency(priority_route)
            
        return emergency_vehicles 