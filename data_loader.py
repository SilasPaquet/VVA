"""
Data loading and preprocessing module for F1 prediction system
"""
import pandas as pd
import numpy as np
from pathlib import Path


class F1DataLoader:
    """Load and preprocess F1 data from CSV files"""
    
    def __init__(self, data_dir='csv'):
        self.data_dir = Path(data_dir)
        self.races = None
        self.drivers = None
        self.constructors = None
        self.results = None
        self.circuits = None
        self.driver_standings = None
        self.constructor_standings = None
        self.qualifying = None
        self.pit_stops = None
        self.lap_times = None
        self.status = None
        
    def load_all_data(self):
        """Load all CSV files"""
        print("Loading data...")
        self.races = pd.read_csv(self.data_dir / 'races.csv')
        self.drivers = pd.read_csv(self.data_dir / 'drivers.csv')
        self.constructors = pd.read_csv(self.data_dir / 'constructors.csv')
        self.results = pd.read_csv(self.data_dir / 'results.csv')
        self.circuits = pd.read_csv(self.data_dir / 'circuits.csv')
        self.driver_standings = pd.read_csv(self.data_dir / 'driver_standings.csv')
        self.constructor_standings = pd.read_csv(self.data_dir / 'constructor_standings.csv')
        self.qualifying = pd.read_csv(self.data_dir / 'qualifying.csv')
        self.pit_stops = pd.read_csv(self.data_dir / 'pit_stops.csv')
        self.lap_times = pd.read_csv(self.data_dir / 'lap_times.csv')
        self.status = pd.read_csv(self.data_dir / 'status.csv')
        print("Data loaded successfully!")
        
    def get_race_features(self):
        """Create features for race prediction"""
        # Merge results with races, drivers, and constructors
        data = self.results.merge(self.races[['raceId', 'year', 'round', 'circuitId', 'date']], 
                                  on='raceId', how='left')
        data = data.merge(self.drivers[['driverId', 'dob', 'nationality']], 
                          on='driverId', how='left')
        data = data.merge(self.constructors[['constructorId', 'name']], 
                          on='constructorId', how='left')
        data = data.merge(self.circuits[['circuitId', 'country']], 
                          on='circuitId', how='left')
        
        # Convert date to datetime
        data['date'] = pd.to_datetime(data['date'])
        
        # Calculate driver age at race
        data['dob'] = pd.to_datetime(data['dob'])
        data['driver_age'] = (data['date'] - data['dob']).dt.days / 365.25
        
        # Finish status (1 if finished, 0 otherwise)
        data['finished'] = (data['position'].notna()).astype(int)
        
        # Points classification
        data['points_scored'] = data['points']
        
        return data
    
    def get_driver_stats(self, data):
        """Calculate rolling driver statistics"""
        driver_stats = data.groupby('driverId').agg({
            'points_scored': 'mean',
            'finished': 'mean',
            'grid': 'mean',
            'position': lambda x: x.notna().sum()  # races completed
        }).reset_index()
        
        driver_stats.columns = ['driverId', 'avg_points', 'finish_rate', 'avg_grid', 'races_completed']
        return driver_stats
    
    def get_constructor_stats(self, data):
        """Calculate rolling constructor statistics"""
        constructor_stats = data.groupby('constructorId').agg({
            'points_scored': 'mean',
            'finished': 'mean',
            'grid': 'mean'
        }).reset_index()
        
        constructor_stats.columns = ['constructorId', 'avg_points', 'finish_rate', 'avg_grid']
        return constructor_stats
    
    def get_circuit_stats(self, data):
        """Calculate circuit-specific statistics"""
        circuit_stats = data.groupby('circuitId').agg({
            'points_scored': 'mean',
            'grid': 'mean'
        }).reset_index()
        
        circuit_stats.columns = ['circuitId', 'avg_circuit_points', 'circuit_grid_avg']
        return circuit_stats


if __name__ == "__main__":
    loader = F1DataLoader()
    loader.load_all_data()
    print("\nData loaded summary:")
    print(f"Races: {len(loader.races)}")
    print(f"Drivers: {len(loader.drivers)}")
    print(f"Results: {len(loader.results)}")
    print(f"Date range: {loader.races['year'].min()} - {loader.races['year'].max()}")
