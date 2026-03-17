"""
F1 Grand Prix Simulator - Simulate race outcomes based on features and predictions
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple


class GPSimulator:
    """Simulate F1 Grand Prix race results"""
    
    def __init__(self, predictor, loader, engineer):
        self.predictor = predictor
        self.loader = loader
        self.engineer = engineer
        self.race_data = None
        
    def simulate_race(self, circuit_id: int, drivers_info: List[Dict], 
                     weather_factor: float = 1.0, safety_car: bool = False) -> pd.DataFrame:
        """
        Simulate a single race
        
        Args:
            circuit_id: Circuit ID from database
            drivers_info: List of dicts with driver features
            weather_factor: Weather impact (0.5-2.0, where 1.0 is neutral)
            safety_car: Whether safety car intervention occurs
        
        Returns:
            DataFrame with predicted race results
        """
        results = []
        
        for driver_info in drivers_info:
            # Create feature vector
            features = self._create_feature_vector(driver_info, circuit_id, weather_factor)
            
            # Get predictions
            points_pred = self.predictor.predict_points(features)
            position_pred = self.predictor.predict_position(features)
            finish_prob = self.predictor.predict_finish(features)
            
            # Apply weather impact
            if weather_factor < 1.0:  # Wet weather favors some drivers
                finish_prob *= (1 - (weather_factor * 0.3))
                position_pred += int(5 * (1 - weather_factor))  # Worse position in wet
            
            # Apply safety car impact
            if safety_car:
                finish_prob *= 1.1  # Safety car increases finishing probability
            
            # Random component for realism
            random_factor = np.random.normal(1.0, 0.1)
            position_pred = int(max(1, position_pred * random_factor))
            points_pred = max(0, points_pred * random_factor)
            finish_prob = np.clip(finish_prob * random_factor, 0, 1)
            
            # Determine if driver finishes
            finished = np.random.random() < finish_prob
            
            result = {
                'driver_id': driver_info.get('driver_id'),
                'driver_name': driver_info.get('driver_name', 'Unknown'),
                'grid_position': int(driver_info.get('grid', 20)),
                'predicted_position': int(position_pred) if finished else None,
                'predicted_points': int(points_pred) if finished else 0,
                'finish_probability': round(finish_prob, 3),
                'finished': finished,
                'weather_factor': weather_factor,
                'constructor': driver_info.get('constructor', 'Unknown')
            }
            
            results.append(result)
        
        # Sort by predicted position
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(
            'predicted_position', 
            na_position='last'
        ).reset_index(drop=True)
        
        # Assign actual points based on FIA scoring system
        results_df['actual_points'] = results_df.apply(
            lambda row: self._get_points(row['predicted_position']) if row['finished'] else 0,
            axis=1
        )
        
        self.race_data = results_df
        return results_df
    
    def simulate_season(self, races_data: List[Dict], num_simulations: int = 1) -> pd.DataFrame:
        """
        Simulate full season with multiple race simulations
        
        Args:
            races_data: List of dicts with race info (circuit_id, drivers_info, weather)
            num_simulations: Number of simulations to average
        
        Returns:
            DataFrame with driver championship standings
        """
        drivers_points = {}
        
        for _ in range(num_simulations):
            for race in races_data:
                race_results = self.simulate_race(
                    circuit_id=race['circuit_id'],
                    drivers_info=race['drivers_info'],
                    weather_factor=race.get('weather_factor', 1.0),
                    safety_car=race.get('safety_car', False)
                )
                
                for _, row in race_results.iterrows():
                    driver_id = row['driver_id']
                    if driver_id not in drivers_points:
                        drivers_points[driver_id] = {
                            'driver_name': row['driver_name'],
                            'constructor': row['constructor'],
                            'points': 0,
                            'races': 0,
                            'wins': 0
                        }
                    
                    drivers_points[driver_id]['points'] += row['actual_points']
                    drivers_points[driver_id]['races'] += 1
                    if row['predicted_position'] == 1:
                        drivers_points[driver_id]['wins'] += 1
        
        # Average over simulations
        for driver_id in drivers_points:
            drivers_points[driver_id]['points'] /= num_simulations
            drivers_points[driver_id]['races'] /= num_simulations
        
        championship = pd.DataFrame.from_dict(drivers_points, orient='index')
        championship = championship.sort_values('points', ascending=False).reset_index(drop=True)
        championship['position'] = range(1, len(championship) + 1)
        
        return championship
    
    def _create_feature_vector(self, driver_info: Dict, circuit_id: int, 
                              weather_factor: float) -> np.ndarray:
        """Create feature vector for model prediction"""
        # Get engineer data which has all rolling features
        data = self.engineer.data
        
        if data is None or len(data) == 0:
            # Fallback to default values
            features = np.array([
                driver_info.get('grid', 10),
                30,  # driver_age
                5.0,  # driver_points_rolling
                0.7,  # driver_finish
                10.0,  # driver_grid
                3.0,  # constructor_points
                0.6,  # constructor_finish
                5.0,  # circuit_points
                10.0,  # circuit_grid_avg
                2024,  # year
                1  # round
            ])
            return features
        
        # Use recent season for baseline
        current_year = data['year'].max()
        recent_data = data[data['year'] >= (current_year - 2)]
        
        # Get driver baseline stats
        driver_stats = recent_data[
            recent_data['driverId'] == driver_info.get('driver_id', 1)
        ]
        
        if len(driver_stats) > 0:
            driver_points_rolling = driver_stats['driver_points_rolling'].mean()
            driver_finish = driver_stats['finished'].mean()
            driver_grid = driver_stats['grid'].mean()
        else:
            driver_points_rolling = 5.0
            driver_finish = 0.7
            driver_grid = 10.0
        
        # Get constructor stats
        constructor_stats = recent_data[
            recent_data['constructorId'] == driver_info.get('constructor_id', 1)
        ]
        
        if len(constructor_stats) > 0:
            constructor_points = constructor_stats['points_scored'].mean()
            constructor_finish = constructor_stats['finished'].mean()
        else:
            constructor_points = 3.0
            constructor_finish = 0.6
        
        # Get circuit stats
        circuit_stats = recent_data[
            recent_data['circuitId'] == circuit_id
        ]
        
        if len(circuit_stats) > 0:
            circuit_points = circuit_stats['points_scored'].mean()
            circuit_grid = circuit_stats['grid'].mean()
        else:
            circuit_points = 5.0
            circuit_grid = 10.0
        
        # Create features
        features = np.array([
            driver_info.get('grid', 10),  # grid
            30,  # driver_age (placeholder)
            driver_points_rolling,
            driver_finish,
            driver_grid,
            constructor_points,
            constructor_finish,
            circuit_points,
            circuit_grid,
            current_year,
            1  # round
        ])
        
        return features
    
    @staticmethod
    def _get_points(position: int) -> int:
        """Get FIA points for position"""
        points_table = {
            1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
            6: 8, 7: 6, 8: 4, 9: 2, 10: 1
        }
        return points_table.get(int(position), 0)


if __name__ == "__main__":
    # Example usage
    from model_trainer import F1Predictor
    from feature_engineer import FeatureEngineer
    
    loader = F1DataLoader()
    loader.load_all_data()
    
    predictor = F1Predictor()
    predictor.train_models(loader)
    
    engineer = FeatureEngineer(loader)
    engineer.create_training_data()
    
    simulator = GPSimulator(predictor, loader, engineer)
    
    # Simulate a single race
    drivers = [
        {'driver_id': 1, 'driver_name': 'Hamilton', 'grid': 1, 'constructor_id': 1, 'constructor': 'Mercedes'},
        {'driver_id': 2, 'driver_name': 'Verstappen', 'grid': 2, 'constructor_id': 2, 'constructor': 'Red Bull'},
    ]
    
    results = simulator.simulate_race(circuit_id=1, drivers_info=drivers, weather_factor=1.0)
    print(results)
