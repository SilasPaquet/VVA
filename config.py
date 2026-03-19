"""
Configuration and utility functions for F1 Prediction System
"""
import os
from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class Config:
    """System configuration"""
    
    # Data configuration
    DATA_DIR = 'csv'
    MODELS_DIR = 'models'
    
    # Model configuration
    LOOKBACK_RACES = 5  # Number of races for rolling statistics
    TEST_SIZE = 0.2
    RECENT_YEARS = 10
    RANDOM_STATE = 42
    
    # Training parameters
    POINTS_MODEL_PARAMS = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'random_state': RANDOM_STATE
    }
    
    POSITION_MODEL_PARAMS = {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': RANDOM_STATE
    }
    
    FINISH_MODEL_PARAMS = {
        'n_estimators': 50,
        'max_depth': 10,
        'random_state': RANDOM_STATE
    }
    
    # Simulation parameters
    WEATHER_IMPACT_RANGE = (0.5, 2.0)  # Min and max weather factors
    WEATHER_NEUTRAL = 1.0
    MAX_DRIVERS = 20
    MAX_POSITION = 20
    

    
    # Feature columns (must match training data)
    FEATURE_COLUMNS = [
        'grid', 'driver_age', 'driver_points_rolling', 'driver_finish_rolling',
        'driver_grid_rolling', 'constructor_points_rolling', 'constructor_finish_rolling',
        'avg_circuit_points', 'circuit_grid_avg', 'year', 'round'
    ]
    
    @classmethod
    def ensure_dirs(cls):
        """Ensure all required directories exist"""
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.MODELS_DIR, exist_ok=True)


class WeatherSimulator:
    """Simulate weather conditions"""
    
    # Weather type descriptions
    WEATHER_TYPES = {
        'sunny': {'factor': 1.0, 'description': 'Sunny - Normal conditions'},
        'cloudy': {'factor': 0.95, 'description': 'Cloudy - Slightly cooler'},
        'light_rain': {'factor': 0.7, 'description': 'Light Rain - Challenging conditions'},
        'heavy_rain': {'factor': 0.5, 'description': 'Heavy Rain - Very challenging'},
        'extreme_heat': {'factor': 1.8, 'description': 'Extreme Heat - High tire degradation'},
    }
    
    @staticmethod
    def get_weather_factor(weather_type: str) -> float:
        """Get weather factor from weather type"""
        return WeatherSimulator.WEATHER_TYPES.get(
            weather_type.lower(),
            WeatherSimulator.WEATHER_TYPES['sunny']
        )['factor']
    
    @staticmethod
    def get_random_weather() -> tuple:
        """Get mixed weather conditions (deterministic mean)"""
        return 'mixed', 1.0





class RaceResult:
    """Data class for race results"""
    
    def __init__(self, driver_id: int, driver_name: str, grid_pos: int,
                 final_pos: int, points: int, finished: bool):
        self.driver_id = driver_id
        self.driver_name = driver_name
        self.grid_pos = grid_pos
        self.final_pos = final_pos
        self.points = points
        self.finished = finished
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'driver_id': self.driver_id,
            'driver_name': self.driver_name,
            'grid_position': self.grid_pos,
            'final_position': self.final_pos,
            'points': self.points,
            'finished': self.finished
        }


def validate_inputs(grid: int, drivers_count: int) -> bool:
    """Validate simulation inputs"""
    if grid < 1 or grid > Config.MAX_DRIVERS:
        return False
    if drivers_count < 2 or drivers_count > Config.MAX_DRIVERS:
        return False
    return True


def calculate_consistency(results: list) -> float:
    """Calculate consistency metric from race results"""
    if not results:
        return 0.0
    
    points = [r.get('points', 0) for r in results]
    if len(points) < 2:
        return 1.0
    
    # Calculate coefficient of variation
    mean = np.mean(points)
    if mean == 0:
        return 0.0
    
    std = np.std(points)
    cv = std / mean
    
    # Convert to 0-1 scale where 1 is most consistent
    return max(0, 1 - cv)


def format_time_delta(start_time, finish_time) -> str:
    """Format time difference"""
    delta = finish_time - start_time
    minutes, seconds = divmod(delta.total_seconds(), 60)
    return f"+{int(minutes)}:{seconds:05.2f}"


if __name__ == "__main__":
    Config.ensure_dirs()
    print("Configuration initialized")
    print(f"Data directory: {Config.DATA_DIR}")
    print(f"Models directory: {Config.MODELS_DIR}")
    print(f"Feature columns: {len(Config.FEATURE_COLUMNS)}")
