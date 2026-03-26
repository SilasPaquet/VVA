"""
Feature engineering for F1 prediction model
"""
import pandas as pd
import numpy as np
from data_loader import F1DataLoader


class FeatureEngineer:
    """Create advanced features for ML models"""
    
    def __init__(self, loader):
        self.loader = loader
        self.data = None
        
    def create_training_data(self, lookback_races=5):
        """Create training dataset with rolling features"""
        data = self.loader.get_race_features()
        data = data.sort_values(['driverId', 'date']).reset_index(drop=True)
        
        for driver_id in data['driverId'].unique():
            driver_mask = data['driverId'] == driver_id
            driver_data = data[driver_mask]
            
            data.loc[driver_mask, 'driver_points_rolling'] = \
                driver_data['points_scored'].rolling(window=lookback_races, min_periods=1).mean()
            data.loc[driver_mask, 'driver_finish_rolling'] = \
                driver_data['finished'].rolling(window=lookback_races, min_periods=1).mean()
            data.loc[driver_mask, 'driver_grid_rolling'] = \
                driver_data['grid'].rolling(window=lookback_races, min_periods=1).mean()
        
        for constructor_id in data['constructorId'].unique():
            constructor_mask = data['constructorId'] == constructor_id
            constructor_data = data[constructor_mask]
            
            data.loc[constructor_mask, 'constructor_points_rolling'] = \
                constructor_data['points_scored'].rolling(window=lookback_races, min_periods=1).mean()
            data.loc[constructor_mask, 'constructor_finish_rolling'] = \
                constructor_data['finished'].rolling(window=lookback_races, min_periods=1).mean()
        
        circuit_stats = self.loader.get_circuit_stats(data)
        data = data.merge(circuit_stats, on='circuitId', how='left')

        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = data[numeric_cols].astype(float)
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
        
        self.data = data
        return data
    
    def get_feature_matrix(self, target_col='points_scored'):
        """Get feature matrix for model training"""
        feature_cols = [
            'grid', 'driver_age', 'driver_points_rolling', 'driver_finish_rolling', 
            'driver_grid_rolling', 'constructor_points_rolling', 'constructor_finish_rolling',
            'avg_circuit_points', 'circuit_grid_avg', 'year', 'round'
        ]
        
        X = self.data[feature_cols].fillna(0)
        y = self.data[target_col].fillna(0)
        
        return X, y, feature_cols
    
    def get_position_prediction_data(self):
        """Prepare data for position prediction"""
        feature_cols = [
            'grid', 'driver_age', 'driver_points_rolling', 'driver_finish_rolling',
            'driver_grid_rolling', 'constructor_points_rolling', 'constructor_finish_rolling',
            'avg_circuit_points', 'circuit_grid_avg', 'year', 'round'
        ]
        
        valid_data = self.data[self.data['position'].notna()].copy()
        
        if len(valid_data) == 0:
            return pd.DataFrame(columns=feature_cols), pd.Series([]), feature_cols
        
        X = valid_data[feature_cols].fillna(0)
        y = pd.to_numeric(valid_data['position'], errors='coerce').fillna(1).astype(int)
        return X, y, feature_cols
    
    def preprocess_features(self, X, scaler=None):
        """Normalize features"""
        if scaler is None:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = scaler.transform(X)
        
        return X_scaled, scaler


if __name__ == "__main__":
    loader = F1DataLoader()
    loader.load_all_data()
    
    engineer = FeatureEngineer(loader)
    data = engineer.create_training_data()
    
    print(f"Training data shape: {data.shape}")
    print(f"\nFeatures created:")
    print(data[['driverId', 'constructorId', 'grid', 'driver_points_rolling', 
                'constructor_points_rolling', 'points_scored']].head(10))
