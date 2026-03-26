"""
Machine learning models for F1 race prediction
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import pickle
from data_loader import F1DataLoader
from feature_engineer import FeatureEngineer


class F1Predictor:
    """Machine learning models for F1 predictions"""
    
    def __init__(self):
        self.points_model = None
        self.position_model = None
        self.finish_model = None
        self.scaler = None
        self.feature_cols = None

    def _prepare_prediction_input(self, features):
        """Convert features to a DataFrame aligned with training feature names."""
        if isinstance(features, pd.DataFrame):
            X_input = features.copy()
        else:
            arr = np.asarray(features, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)

            if self.feature_cols is not None and len(self.feature_cols) == arr.shape[1]:
                X_input = pd.DataFrame(arr, columns=self.feature_cols)
            else:
                X_input = pd.DataFrame(arr)

        if self.feature_cols is not None:
            if set(self.feature_cols).issubset(set(X_input.columns)):
                X_input = X_input[self.feature_cols]
            elif X_input.shape[1] == len(self.feature_cols):
                X_input.columns = self.feature_cols
            else:
                raise ValueError(
                    f"Expected {len(self.feature_cols)} features, got {X_input.shape[1]}"
                )

        return X_input
        
    def train_models(self, loader, test_size=0.2, recent_years=10):
        """Train all prediction models"""
        engineer = FeatureEngineer(loader)
        data = engineer.create_training_data()
        
        current_year = data['year'].max()
        recent_data = data[data['year'] >= (current_year - recent_years)]
        
        print("Training points prediction model...")
        X_points, y_points, self.feature_cols = engineer.get_feature_matrix('points_scored')
        X_points = X_points.iloc[len(X_points) - len(recent_data):]
        y_points = y_points.iloc[len(y_points) - len(recent_data):]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_points, y_points, test_size=test_size, random_state=42
        )
        
        self.points_model = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
        )
        self.points_model.fit(X_train, y_train)
        
        points_pred = self.points_model.predict(X_test)
        points_r2 = r2_score(y_test, points_pred)
        points_mae = mean_absolute_error(y_test, points_pred)
        print(f"Points Model - R2: {points_r2:.4f}, MAE: {points_mae:.4f}")
        
        # Train position prediction model
        print("Training position prediction model...")
        X_pos, y_pos, _ = engineer.get_position_prediction_data()
        X_pos = X_pos.iloc[len(X_pos) - len(recent_data):]
        y_pos = y_pos.iloc[len(y_pos) - len(recent_data):]
        
        X_train_pos, X_test_pos, y_train_pos, y_test_pos = train_test_split(
            X_pos, y_pos, test_size=test_size, random_state=42
        )
        
        self.position_model = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42
        )
        self.position_model.fit(X_train_pos, y_train_pos)
        
        pos_pred = self.position_model.predict(X_test_pos)
        pos_r2 = r2_score(y_test_pos, pos_pred)
        pos_mae = mean_absolute_error(y_test_pos, pos_pred)
        print(f"Position Model - R2: {pos_r2:.4f}, MAE: {pos_mae:.4f}")
        
        print("Training finish prediction model...")
        X_finish = X_points.copy()
        y_finish = (recent_data['finished'].iloc[len(recent_data) - len(X_points):].values).astype(int)
        
        X_train_fin, X_test_fin, y_train_fin, y_test_fin = train_test_split(
            X_finish, y_finish, test_size=test_size, random_state=42
        )
        
        from sklearn.ensemble import RandomForestClassifier
        self.finish_model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
        self.finish_model.fit(X_train_fin, y_train_fin)
        
        finish_score = self.finish_model.score(X_test_fin, y_test_fin)
        print(f"Finish Model - Accuracy: {finish_score:.4f}")
        
        return self
    
    def predict_points(self, features):
        """Predict points scored"""
        if self.points_model is None:
            raise ValueError("Model not trained yet. Call train_models() first.")
        X_input = self._prepare_prediction_input(features)
        return self.points_model.predict(X_input)[0]
    
    def predict_position(self, features):
        """Predict finishing position"""
        if self.position_model is None:
            raise ValueError("Model not trained yet. Call train_models() first.")
        X_input = self._prepare_prediction_input(features)
        pred = max(1, int(self.position_model.predict(X_input)[0]))
        return min(pred, 20)
    
    def predict_finish(self, features):
        """Predict probability of finishing race"""
        if self.finish_model is None:
            raise ValueError("Model not trained yet. Call train_models() first.")

        X_input = self._prepare_prediction_input(features)
        proba = self.finish_model.predict_proba(X_input)
        
        if proba.shape[1] == 1:
            return proba[0][0] if self.finish_model.classes_[0] == 1 else 1 - proba[0][0]
        
        return proba[0][1]
    
    def save_models(self, path='models'):
        """Save trained models"""
        import os
        os.makedirs(path, exist_ok=True)
        with open(f'{path}/points_model.pkl', 'wb') as f:
            pickle.dump(self.points_model, f)
        with open(f'{path}/position_model.pkl', 'wb') as f:
            pickle.dump(self.position_model, f)
        with open(f'{path}/finish_model.pkl', 'wb') as f:
            pickle.dump(self.finish_model, f)
        with open(f'{path}/feature_cols.pkl', 'wb') as f:
            pickle.dump(self.feature_cols, f)
        print(f"Models saved to {path}/")
    
    def load_models(self, path='models'):
        """Load trained models"""
        with open(f'{path}/points_model.pkl', 'rb') as f:
            self.points_model = pickle.load(f)
        with open(f'{path}/position_model.pkl', 'rb') as f:
            self.position_model = pickle.load(f)
        with open(f'{path}/finish_model.pkl', 'rb') as f:
            self.finish_model = pickle.load(f)
        with open(f'{path}/feature_cols.pkl', 'rb') as f:
            self.feature_cols = pickle.load(f)
        print(f"Models loaded from {path}/")


if __name__ == "__main__":
    loader = F1DataLoader()
    loader.load_all_data()
    
    predictor = F1Predictor()
    predictor.train_models(loader)
    predictor.save_models()
