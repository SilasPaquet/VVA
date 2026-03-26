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
        self.deterministic = True
        self.volatility_scale = 0.0
        self.race_data = None
        self.points_table = self._infer_points_table()
        self.randomness_std = 0.0

    def _infer_points_table(self) -> Dict[int, int]:
        """Infer points table from latest available season in CSV results."""
        if self.loader.results is None or self.loader.races is None:
            return {}

        merged = self._merge_results_and_races()
        if merged.empty:
            return {}

        merged['positionOrder'] = pd.to_numeric(merged['positionOrder'], errors='coerce')
        merged['points'] = pd.to_numeric(merged['points'], errors='coerce')

        latest = self._get_latest_season_data(merged)
        return self._build_points_table(latest)

    def _merge_results_and_races(self) -> pd.DataFrame:
        return self.loader.results.merge(
            self.loader.races[['raceId', 'year']],
            on='raceId',
            how='left'
        )

    def _get_latest_season_data(self, merged_data: pd.DataFrame) -> pd.DataFrame:
        latest_year = merged_data['year'].max()
        latest = merged_data[merged_data['year'] == latest_year].copy()
        return latest if not latest.empty else merged_data.copy()

    def _build_points_table(self, data: pd.DataFrame) -> Dict[int, int]:
        grouped = (
            data.dropna(subset=['positionOrder', 'points'])
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

        frame = self._prepare_grid_position_data()
        if frame.empty:
            return 0.0

        pos_gain = frame['grid'] - frame['positionOrder']
        scale = float(pos_gain.abs().quantile(0.9))
        if pd.isna(scale) or scale == 0:
            return 0.0

        normalized = pos_gain / scale
        std_val = float(normalized.std(ddof=0))
        return std_val if not pd.isna(std_val) and std_val >= 0 else 0.0

    def _prepare_grid_position_data(self) -> pd.DataFrame:
        frame = self.loader.results[['grid', 'positionOrder']].copy()
        frame['grid'] = pd.to_numeric(frame['grid'], errors='coerce')
        frame['positionOrder'] = pd.to_numeric(frame['positionOrder'], errors='coerce')
        frame = frame.dropna(subset=['grid', 'positionOrder'])
        return frame[(frame['grid'] > 0) & (frame['positionOrder'] > 0)]

    def _resolve_reference_date_and_round(self, circuit_id: int, current_year: int):
        """Resolve year/round context from historical races for a selected circuit."""
        if self.loader.races is None or self.loader.races.empty:
            return pd.Timestamp(f"{int(current_year)}-01-01"), 1

        races = self._prepare_races_data()
        circuit_races = races[races['circuitId'] == circuit_id].dropna(subset=['date'])

        if not circuit_races.empty:
            return self._get_circuit_reference(circuit_races, races)

        return self._get_fallback_reference(races, current_year)

    def _prepare_races_data(self) -> pd.DataFrame:
        races = self.loader.races[['circuitId', 'date', 'round']].copy()
        races['date'] = pd.to_datetime(races['date'], errors='coerce')
        races['round'] = pd.to_numeric(races['round'], errors='coerce')
        return races

    def _get_circuit_reference(self, circuit_races: pd.DataFrame, all_races: pd.DataFrame):
        latest_row = circuit_races.sort_values('date').iloc[-1]
        ref_date = latest_row['date']
        if not pd.isna(latest_row['round']):
            return ref_date, int(latest_row['round'])

        rounds_all = all_races['round'].dropna()
        return ref_date, int(rounds_all.median()) if not rounds_all.empty else 1

    def _get_fallback_reference(self, all_races: pd.DataFrame, current_year: int):
        all_dates = all_races['date'].dropna()
        ref_date = all_dates.max() if not all_dates.empty else pd.Timestamp(f"{int(current_year)}-01-01")

        rounds = all_races['round'].dropna()
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
        """
        results = [self._simulate_driver(driver, circuit_id, weather_factor, safety_car) 
                   for driver in drivers_info]

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('predicted_position', na_position='last').reset_index(drop=True)

        results_df['actual_points'] = results_df.apply(
            lambda row: self._get_points(row['predicted_position']) if row['finished'] else 0,
            axis=1
        )

        self.race_data = results_df
        return results_df

    def _simulate_driver(self, driver_info: Dict, circuit_id: int, weather_factor: float, safety_car: bool) -> Dict:
        features = self._create_feature_vector(driver_info, circuit_id)

        points_pred = self.predictor.predict_points(features)
        position_pred = self.predictor.predict_position(features)
        finish_prob = self.predictor.predict_finish(features)

        if weather_factor < 1.0:
            finish_prob *= (1 - (weather_factor * 0.3))
            position_pred += int(5 * (1 - weather_factor))

        if safety_car:
            finish_prob *= 1.1

        position_pred = int(max(1, position_pred))
        points_pred = max(0, points_pred)
        finish_prob = np.clip(finish_prob, 0, 1)

        finished = finish_prob >= 0.5

        return {
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

    def simulate_season(self, races_data: List[Dict], num_simulations: int = 1) -> pd.DataFrame:
        """
        Simulate full season with multiple race simulations.
        """
        drivers_points = {}

        for _ in range(num_simulations):
            self._simulate_single_season(races_data, drivers_points)

        for driver_id in drivers_points:
            drivers_points[driver_id]['points'] /= num_simulations
            drivers_points[driver_id]['races'] /= num_simulations

        championship = pd.DataFrame.from_dict(drivers_points, orient='index')
        championship = championship.sort_values('points', ascending=False).reset_index(drop=True)
        championship['position'] = range(1, len(championship) + 1)

        return championship

    def _simulate_single_season(self, races_data: List[Dict], drivers_points: Dict):
        for race in races_data:
            race_results = self.simulate_race(
                circuit_id=race['circuit_id'],
                drivers_info=race['drivers_info'],
                weather_factor=race.get('weather_factor', 1.0),
                safety_car=race.get('safety_car', False)
            )
            self._update_championship_standings(race_results, drivers_points)

    def _update_championship_standings(self, race_results: pd.DataFrame, drivers_points: Dict):
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

    def _get_recent_data(self) -> pd.DataFrame:
        data = self.engineer.data
        if data is None or len(data) == 0:
            data = self.engineer.create_training_data()
            
        if data is None or len(data) == 0:
            raise ValueError("No race data available to build prediction features.")
            
        current_year = int(data['year'].max())
        recent_data = data[data['year'] >= (current_year - 2)]
        return recent_data if not recent_data.empty else data.copy()

    def _get_global_stats(self, recent_data: pd.DataFrame) -> Dict[str, float]:
        def get_stat(col, fallback):
            val = self._safe_mean(recent_data, col)
            return val if not pd.isna(val) else self._safe_mean(recent_data, fallback)
        
        return {
            'driver_age': self._safe_mean(recent_data, 'driver_age'),
            'driver_points': get_stat('driver_points_rolling', 'points_scored'),
            'driver_finish': get_stat('driver_finish_rolling', 'finished'),
            'driver_grid': get_stat('driver_grid_rolling', 'grid'),
            'constructor_points': get_stat('constructor_points_rolling', 'points_scored'),
            'constructor_finish': get_stat('constructor_finish_rolling', 'finished'),
            'circuit_points': get_stat('avg_circuit_points', 'points_scored'),
            'circuit_grid': get_stat('circuit_grid_avg', 'grid'),
            'round': self._safe_mean(recent_data, 'round')
        }

    def _get_driver_stats(self, recent_data: pd.DataFrame, driver_id: int, global_stats: Dict[str, float]) -> Dict[str, float]:
        stats = recent_data[recent_data['driverId'] == driver_id]
        if stats.empty:
            return {
                'points_rolling': global_stats['driver_points'],
                'finish': global_stats['driver_finish'],
                'grid': global_stats['driver_grid']
            }
        
        def get_stat(col, fallback):
            val = self._safe_mean(stats, col)
            return val if not pd.isna(val) else self._safe_mean(stats, fallback)

        return {
            'points_rolling': get_stat('driver_points_rolling', 'points_scored'),
            'finish': get_stat('driver_finish_rolling', 'finished'),
            'grid': get_stat('driver_grid_rolling', 'grid')
        }

    def _get_constructor_stats(self, recent_data: pd.DataFrame, constructor_id: int, global_stats: Dict[str, float]) -> Dict[str, float]:
        stats = recent_data[recent_data['constructorId'] == constructor_id]
        if stats.empty:
            return {
                'points': global_stats['constructor_points'],
                'finish': global_stats['constructor_finish']
            }
        
        def get_stat(col, fallback):
            val = self._safe_mean(stats, col)
            return val if not pd.isna(val) else self._safe_mean(stats, fallback)

        return {
            'points': get_stat('constructor_points_rolling', 'points_scored'),
            'finish': get_stat('constructor_finish_rolling', 'finished')
        }

    def _get_circuit_stats(self, recent_data: pd.DataFrame, circuit_id: int, global_stats: Dict[str, float]) -> Dict[str, float]:
        stats = recent_data[recent_data['circuitId'] == circuit_id]
        if stats.empty:
            return {
                'points': global_stats['circuit_points'],
                'grid': global_stats['circuit_grid']
            }
        
        def get_stat(col, fallback):
            val = self._safe_mean(stats, col)
            return val if not pd.isna(val) else self._safe_mean(stats, fallback)

        return {
            'points': get_stat('avg_circuit_points', 'points_scored'),
            'grid': get_stat('circuit_grid_avg', 'grid')
        }

    def _calculate_driver_age(self, driver_id: int, reference_date: pd.Timestamp, global_age: float) -> float:
        if self.loader.drivers is None or self.loader.drivers.empty:
            return global_age
        
        driver_rows = self.loader.drivers[self.loader.drivers['driverId'] == driver_id]
        if driver_rows.empty:
            return global_age
            
        dob = pd.to_datetime(driver_rows.iloc[0]['dob'], errors='coerce')
        if pd.isna(dob):
            return global_age
            
        return float((reference_date - dob).days / 365.25)

    def _create_feature_vector(self, driver_info: Dict, circuit_id: int) -> np.ndarray:
        """Create model feature vector strictly from CSV-derived historical stats."""
        recent_data = self._get_recent_data()
        current_year = int(recent_data['year'].max())
        
        global_stats = self._get_global_stats(recent_data)
        
        reference_date, round_value = self._resolve_reference_date_and_round(circuit_id, current_year)
        
        driver_id = driver_info.get('driver_id')
        constructor_id = driver_info.get('constructor_id')
        
        driver_age = self._calculate_driver_age(driver_id, reference_date, global_stats['driver_age'])
        driver_stats = self._get_driver_stats(recent_data, driver_id, global_stats)
        constructor_stats = self._get_constructor_stats(recent_data, constructor_id, global_stats)
        circuit_stats = self._get_circuit_stats(recent_data, circuit_id, global_stats)
        
        fallback_round = global_stats['round'] if not pd.isna(global_stats['round']) else 1
        
        grid_input = driver_info.get('grid', global_stats['driver_grid'])
        if grid_input is None:
            grid_input = global_stats['driver_grid']
        
        year_value = int(reference_date.year) if not pd.isna(reference_date) else current_year
        round_value = int(round_value) if not pd.isna(round_value) else int(fallback_round)
        
        feature_values = [
            grid_input,
            driver_age,
            driver_stats['points_rolling'],
            driver_stats['finish'],
            driver_stats['grid'],
            constructor_stats['points'],
            constructor_stats['finish'],
            circuit_stats['points'],
            circuit_stats['grid'],
            year_value,
            round_value,
        ]

        return np.nan_to_num(np.array(feature_values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

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
