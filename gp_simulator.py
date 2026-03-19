"""
F1 Grand Prix Simulator - Simulate race outcomes based on features and predictions
"""
import numpy as np
import pandas as pd
from typing import List, Dict


class GPSimulator:
    """Simulate F1 Grand Prix race results"""

    def __init__(self, predictor, loader, engineer, deterministic: bool = True,
                 volatility_scale: float = 0.08):
        self.predictor = predictor
        self.loader = loader
        self.engineer = engineer
        self.deterministic = bool(deterministic)
        self.volatility_scale = max(0.0, min(float(volatility_scale), 1.0))
        self.race_data = None

        # Data-driven simulation settings inferred from CSVs.
        self.points_table = self._infer_points_table()
        inferred_std = self._infer_randomness_std()
        self.randomness_std = 0.0 if self.deterministic else min(inferred_std, self.volatility_scale)

    def _infer_points_table(self) -> Dict[int, int]:
        """Infer points table from latest available season in CSV results."""
        if self.loader.results is None or self.loader.races is None:
            return {}

        merged = self.loader.results.merge(
            self.loader.races[['raceId', 'year']],
            on='raceId',
            how='left'
        )
        if merged.empty:
            return {}

        merged['positionOrder'] = pd.to_numeric(merged['positionOrder'], errors='coerce')
        merged['points'] = pd.to_numeric(merged['points'], errors='coerce')

        latest_year = merged['year'].max()
        latest = merged[merged['year'] == latest_year].copy()
        if latest.empty:
            latest = merged.copy()

        grouped = (
            latest.dropna(subset=['positionOrder', 'points'])
            .groupby('positionOrder')['points']
            .median()
            .sort_index()
        )

        table: Dict[int, int] = {}
        for pos, pts in grouped.items():
            pos_i = int(pos)
            pts_i = int(round(float(pts)))
            if pos_i >= 1 and pts_i > 0:
                table[pos_i] = pts_i

        return table

    def _infer_randomness_std(self) -> float:
        """Infer simulation volatility from historical grid-to-finish variation."""
        if self.loader.results is None or self.loader.results.empty:
            return 0.0

        frame = self.loader.results[['grid', 'positionOrder']].copy()
        frame['grid'] = pd.to_numeric(frame['grid'], errors='coerce')
        frame['positionOrder'] = pd.to_numeric(frame['positionOrder'], errors='coerce')
        frame = frame.dropna(subset=['grid', 'positionOrder'])
        frame = frame[(frame['grid'] > 0) & (frame['positionOrder'] > 0)]

        if frame.empty:
            return 0.0

        pos_gain = frame['grid'] - frame['positionOrder']
        scale = float(pos_gain.abs().quantile(0.9))
        if pd.isna(scale) or scale == 0:
            return 0.0

        normalized = pos_gain / scale
        std_val = float(normalized.std(ddof=0))
        if pd.isna(std_val) or std_val < 0:
            return 0.0

        return std_val

    def _resolve_reference_date_and_round(self, circuit_id: int, current_year: int):
        """Resolve year/round context from historical races for a selected circuit."""
        if self.loader.races is None or self.loader.races.empty:
            return pd.Timestamp(f"{int(current_year)}-01-01"), 1

        races = self.loader.races[['circuitId', 'date', 'round']].copy()
        races['date'] = pd.to_datetime(races['date'], errors='coerce')
        races['round'] = pd.to_numeric(races['round'], errors='coerce')

        circuit_races = races[races['circuitId'] == circuit_id].dropna(subset=['date'])
        if not circuit_races.empty:
            latest_row = circuit_races.sort_values('date').iloc[-1]
            ref_date = latest_row['date']
            if not pd.isna(latest_row['round']):
                return ref_date, int(latest_row['round'])

            rounds_all = races['round'].dropna()
            if not rounds_all.empty:
                return ref_date, int(rounds_all.median())

            return ref_date, 1

        all_dates = races['date'].dropna()
        ref_date = all_dates.max() if not all_dates.empty else pd.Timestamp(f"{int(current_year)}-01-01")

        rounds = races['round'].dropna()
        round_value = int(rounds.median()) if not rounds.empty else 1
        return ref_date, round_value

    @staticmethod
    def _safe_mean(frame: pd.DataFrame, col_name: str) -> float:
        if col_name not in frame.columns:
            return np.nan
        values = pd.to_numeric(frame[col_name], errors='coerce').dropna()
        if values.empty:
            return np.nan
        return float(values.mean())

    def simulate_race(self, circuit_id: int, drivers_info: List[Dict],
                      weather_factor: float = 1.0, safety_car: bool = False) -> pd.DataFrame:
        """
        Simulate a single race.

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
            # Build model input from historical CSV-derived stats.
            features = self._create_feature_vector(driver_info, circuit_id)

            # Base model predictions.
            points_pred = self.predictor.predict_points(features)
            position_pred = self.predictor.predict_position(features)
            finish_prob = self.predictor.predict_finish(features)

            # Weather is explicitly allowed as external factor.
            if weather_factor < 1.0:
                finish_prob *= (1 - (weather_factor * 0.3))
                position_pred += int(5 * (1 - weather_factor))

            if safety_car:
                finish_prob *= 1.1

            # Deterministic-by-default: same inputs -> same outputs.
            random_factor = 1.0
            if self.randomness_std > 0:
                random_factor = np.random.normal(1.0, self.randomness_std)
                random_factor = max(0.05, float(random_factor))

            position_pred = int(max(1, position_pred * random_factor))
            points_pred = max(0, points_pred * random_factor)
            finish_prob = np.clip(finish_prob * random_factor, 0, 1)

            if self.deterministic:
                finished = finish_prob >= 0.5
            else:
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

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('predicted_position', na_position='last').reset_index(drop=True)

        results_df['actual_points'] = results_df.apply(
            lambda row: self._get_points(row['predicted_position']) if row['finished'] else 0,
            axis=1
        )

        self.race_data = results_df
        return results_df

    def simulate_season(self, races_data: List[Dict], num_simulations: int = 1) -> pd.DataFrame:
        """
        Simulate full season with multiple race simulations.

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

        for driver_id in drivers_points:
            drivers_points[driver_id]['points'] /= num_simulations
            drivers_points[driver_id]['races'] /= num_simulations

        championship = pd.DataFrame.from_dict(drivers_points, orient='index')
        championship = championship.sort_values('points', ascending=False).reset_index(drop=True)
        championship['position'] = range(1, len(championship) + 1)

        return championship

    def _create_feature_vector(self, driver_info: Dict, circuit_id: int) -> np.ndarray:
        """Create model feature vector strictly from CSV-derived historical stats."""
        data = self.engineer.data
        if data is None or len(data) == 0:
            data = self.engineer.create_training_data()

        if data is None or len(data) == 0:
            raise ValueError("No race data available to build prediction features.")

        current_year = int(data['year'].max())
        recent_data = data[data['year'] >= (current_year - 2)]
        if recent_data.empty:
            recent_data = data.copy()

        global_driver_age = self._safe_mean(recent_data, 'driver_age')
        global_driver_points = self._safe_mean(recent_data, 'driver_points_rolling')
        if pd.isna(global_driver_points):
            global_driver_points = self._safe_mean(recent_data, 'points_scored')

        global_driver_finish = self._safe_mean(recent_data, 'driver_finish_rolling')
        if pd.isna(global_driver_finish):
            global_driver_finish = self._safe_mean(recent_data, 'finished')

        global_driver_grid = self._safe_mean(recent_data, 'driver_grid_rolling')
        if pd.isna(global_driver_grid):
            global_driver_grid = self._safe_mean(recent_data, 'grid')

        global_constructor_points = self._safe_mean(recent_data, 'constructor_points_rolling')
        if pd.isna(global_constructor_points):
            global_constructor_points = self._safe_mean(recent_data, 'points_scored')

        global_constructor_finish = self._safe_mean(recent_data, 'constructor_finish_rolling')
        if pd.isna(global_constructor_finish):
            global_constructor_finish = self._safe_mean(recent_data, 'finished')

        global_circuit_points = self._safe_mean(recent_data, 'avg_circuit_points')
        if pd.isna(global_circuit_points):
            global_circuit_points = self._safe_mean(recent_data, 'points_scored')

        global_circuit_grid = self._safe_mean(recent_data, 'circuit_grid_avg')
        if pd.isna(global_circuit_grid):
            global_circuit_grid = self._safe_mean(recent_data, 'grid')

        reference_date, round_value = self._resolve_reference_date_and_round(circuit_id, current_year)

        driver_age = global_driver_age
        if self.loader.drivers is not None and not self.loader.drivers.empty:
            driver_rows = self.loader.drivers[self.loader.drivers['driverId'] == driver_info.get('driver_id')]
            if not driver_rows.empty:
                dob = pd.to_datetime(driver_rows.iloc[0]['dob'], errors='coerce')
                if not pd.isna(dob):
                    driver_age = float((reference_date - dob).days / 365.25)

        driver_stats = recent_data[recent_data['driverId'] == driver_info.get('driver_id')]
        if not driver_stats.empty:
            driver_points_rolling = self._safe_mean(driver_stats, 'driver_points_rolling')
            if pd.isna(driver_points_rolling):
                driver_points_rolling = self._safe_mean(driver_stats, 'points_scored')

            driver_finish = self._safe_mean(driver_stats, 'driver_finish_rolling')
            if pd.isna(driver_finish):
                driver_finish = self._safe_mean(driver_stats, 'finished')

            driver_grid = self._safe_mean(driver_stats, 'driver_grid_rolling')
            if pd.isna(driver_grid):
                driver_grid = self._safe_mean(driver_stats, 'grid')
        else:
            driver_points_rolling = global_driver_points
            driver_finish = global_driver_finish
            driver_grid = global_driver_grid

        constructor_stats = recent_data[recent_data['constructorId'] == driver_info.get('constructor_id')]
        if not constructor_stats.empty:
            constructor_points = self._safe_mean(constructor_stats, 'constructor_points_rolling')
            if pd.isna(constructor_points):
                constructor_points = self._safe_mean(constructor_stats, 'points_scored')

            constructor_finish = self._safe_mean(constructor_stats, 'constructor_finish_rolling')
            if pd.isna(constructor_finish):
                constructor_finish = self._safe_mean(constructor_stats, 'finished')
        else:
            constructor_points = global_constructor_points
            constructor_finish = global_constructor_finish

        circuit_stats = recent_data[recent_data['circuitId'] == circuit_id]
        if not circuit_stats.empty:
            circuit_points = self._safe_mean(circuit_stats, 'avg_circuit_points')
            if pd.isna(circuit_points):
                circuit_points = self._safe_mean(circuit_stats, 'points_scored')

            circuit_grid = self._safe_mean(circuit_stats, 'circuit_grid_avg')
            if pd.isna(circuit_grid):
                circuit_grid = self._safe_mean(circuit_stats, 'grid')
        else:
            circuit_points = global_circuit_points
            circuit_grid = global_circuit_grid

        fallback_round = self._safe_mean(recent_data, 'round')
        if pd.isna(fallback_round):
            fallback_round = 1

        grid_input = driver_info.get('grid')
        if grid_input is None:
            grid_input = global_driver_grid

        year_value = int(reference_date.year) if not pd.isna(reference_date) else current_year
        round_value = int(round_value) if not pd.isna(round_value) else int(fallback_round)

        feature_values = [
            grid_input,
            driver_age,
            driver_points_rolling,
            driver_finish,
            driver_grid,
            constructor_points,
            constructor_finish,
            circuit_points,
            circuit_grid,
            year_value,
            round_value,
        ]

        features = np.nan_to_num(np.array(feature_values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        return features

    def _get_points(self, position: int) -> int:
        """Get points for position using data-inferred scoring table."""
        if pd.isna(position):
            return 0
        return int(self.points_table.get(int(position), 0))


if __name__ == "__main__":
    # Example usage
    from data_loader import F1DataLoader
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
