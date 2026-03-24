"""
Data loading and preprocessing module for F1 prediction system
"""
import pandas as pd
import numpy as np
from pathlib import Path


class F1DataLoader:
    """Load and preprocess F1 data from CSV files"""
    
    def __init__(self, data_dir='csv', cleaned_subdir='cleaned'):
        self.data_dir = Path(data_dir)
        self.cleaned_dir = self.data_dir / cleaned_subdir
        self.merged_cache_file = self.cleaned_dir / 'race_features.csv'
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
        self.quality_report = []

        self._dataset_attrs = [
            'races', 'drivers', 'constructors', 'results', 'circuits',
            'driver_standings', 'constructor_standings', 'qualifying',
            'pit_stops', 'lap_times', 'status'
        ]

        self._date_columns = {
            'races': ['date'],
            'drivers': ['dob'],
        }

        # Use pandas nullable integer types so missing values stay as <NA>.
        self._dtype_map = {
            'races': {
                'raceId': 'Int64',
                'year': 'Int64',
                'round': 'Int64',
                'circuitId': 'Int64',
            },
            'drivers': {
                'driverId': 'Int64',
            },
            'constructors': {
                'constructorId': 'Int64',
            },
            'results': {
                'raceId': 'Int64',
                'driverId': 'Int64',
                'constructorId': 'Int64',
                'grid': 'Int64',
                'position': 'Int64',
                'statusId': 'Int64',
            },
            'circuits': {
                'circuitId': 'Int64',
            },
            'driver_standings': {
                'raceId': 'Int64',
                'driverId': 'Int64',
                'position': 'Int64',
            },
            'constructor_standings': {
                'raceId': 'Int64',
                'constructorId': 'Int64',
                'position': 'Int64',
            },
            'qualifying': {
                'raceId': 'Int64',
                'driverId': 'Int64',
                'constructorId': 'Int64',
                'position': 'Int64',
            },
            'pit_stops': {
                'raceId': 'Int64',
                'driverId': 'Int64',
            },
            'lap_times': {
                'raceId': 'Int64',
                'driverId': 'Int64',
                'lap': 'Int64',
                'position': 'Int64',
            },
            'status': {
                'statusId': 'Int64',
            },
        }

        # Float fields that should always be numeric.
        self._float_columns = {
            'results': ['points'],
            'driver_standings': ['points'],
            'constructor_standings': ['points'],
        }

        # Natural keys used to detect duplicate records.
        self._natural_keys = {
            'results': ['raceId', 'driverId'],
            'qualifying': ['raceId', 'driverId'],
            'driver_standings': ['raceId', 'driverId'],
            'constructor_standings': ['raceId', 'constructorId'],
        }
        
    def _read_csv(self, path):
        """Read CSV and normalize known placeholder null values."""
        return pd.read_csv(path, na_values=['\\N'])

    def _coerce_expected_types(self, table_name, df):
        """Apply table-specific type coercion for key model columns."""
        for col in self._dtype_map.get(table_name, {}):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(self._dtype_map[table_name][col])

        for col in self._float_columns.get(table_name, []):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)

        for col in self._date_columns.get(table_name, []):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        return df

    def _record_quality(self, dataset, issue, affected_rows):
        self.quality_report.append({
            'dataset': dataset,
            'issue': issue,
            'affected_rows': int(affected_rows),
        })

    def _apply_duplicate_rules(self, table_name, df):
        """Drop duplicate rows using dataset natural keys."""
        natural_key = self._natural_keys.get(table_name)
        if not natural_key:
            return df

        if not set(natural_key).issubset(df.columns):
            return df

        duplicate_mask = df.duplicated(subset=natural_key, keep='last')
        duplicate_count = int(duplicate_mask.sum())
        if duplicate_count > 0:
            self._record_quality(table_name, f'duplicates_on_{"_".join(natural_key)}', duplicate_count)
            df = df.loc[~duplicate_mask].copy()
        return df

    def _save_cleaned_cache(self):
        """Persist cleaned datasets so future runs can load quickly."""
        self.cleaned_dir.mkdir(parents=True, exist_ok=True)
        for attr in self._dataset_attrs:
            data = getattr(self, attr)
            if data is not None:
                data.to_csv(self.cleaned_dir / f'{attr}.csv', index=False)

        if self.quality_report:
            pd.DataFrame(self.quality_report).to_csv(self.cleaned_dir / 'quality_report.csv', index=False)

    def _save_merged_features_cache(self, data):
        """Persist merged race features used by feature engineering."""
        self.cleaned_dir.mkdir(parents=True, exist_ok=True)
        data.to_csv(self.merged_cache_file, index=False)

    def _load_merged_features_cache(self):
        """Load merged race features from cleaned cache."""
        data = self._read_csv(self.merged_cache_file)
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'], errors='coerce')
        if 'dob' in data.columns:
            data['dob'] = pd.to_datetime(data['dob'], errors='coerce')
        return data

    def _load_cleaned_cache(self):
        """Load already-cleaned datasets from cache."""
        for attr in self._dataset_attrs:
            cached_file = self.cleaned_dir / f'{attr}.csv'
            table = self._read_csv(cached_file)
            table = self._coerce_expected_types(attr, table)
            setattr(self, attr, table)

        quality_file = self.cleaned_dir / 'quality_report.csv'
        if quality_file.exists():
            self.quality_report = pd.read_csv(quality_file).to_dict(orient='records')
        else:
            self.quality_report = []

    def _cleaned_cache_exists(self):
        """Check whether all cleaned datasets are available on disk."""
        return all((self.cleaned_dir / f'{attr}.csv').exists() for attr in self._dataset_attrs)

    def load_all_data(self, use_clean_cache=True, force_rebuild=False):
        """Load all CSV files, clean them, and optionally cache cleaned versions."""
        print('Loading data...')

        if force_rebuild and self.merged_cache_file.exists():
            self.merged_cache_file.unlink()

        if use_clean_cache and not force_rebuild and self._cleaned_cache_exists():
            self._load_cleaned_cache()
            print(f'Data loaded from cleaned cache: {self.cleaned_dir}')
            return

        self.quality_report = []
        for attr in self._dataset_attrs:
            raw_file = self.data_dir / f'{attr}.csv'
            table = self._read_csv(raw_file)
            table = self._coerce_expected_types(attr, table)
            table = self._apply_duplicate_rules(attr, table)
            setattr(self, attr, table)

        if use_clean_cache:
            self._save_cleaned_cache()
            print(f'Cleaned datasets cached to: {self.cleaned_dir}')

        print('Data loaded successfully!')

    def get_dataset_limits(self):
        """Return dataset-driven limits used by simulator controls."""
        max_races_per_season = 23
        max_drivers_per_race = 20

        if self.races is not None and not self.races.empty:
            races_per_season = self.races.groupby('year')['raceId'].nunique()
            if not races_per_season.empty:
                max_races_per_season = int(races_per_season.max())

        if self.results is not None and not self.results.empty:
            drivers_per_race = self.results.groupby('raceId')['driverId'].nunique()
            if not drivers_per_race.empty:
                max_drivers_per_race = int(drivers_per_race.max())

        return {
            'max_races_per_season': max(2, max_races_per_season),
            'max_drivers_per_race': max(2, max_drivers_per_race),
        }
        
    def get_race_features(self, use_merged_cache=True, force_rebuild=False):
        """Create features for race prediction"""
        if use_merged_cache and not force_rebuild and self.merged_cache_file.exists():
            return self._load_merged_features_cache()

        # Merge results with races, drivers, and constructors
        data = self.results.merge(self.races[['raceId', 'year', 'round', 'circuitId', 'date']], 
                                  on='raceId', how='left')
        data = data.merge(self.drivers[['driverId', 'dob', 'nationality']], 
                          on='driverId', how='left')
        data = data.merge(self.constructors[['constructorId', 'name']], 
                          on='constructorId', how='left')
        data = data.merge(self.circuits[['circuitId', 'country']], 
                          on='circuitId', how='left')
        
        data['date'] = pd.to_datetime(data['date'])
        data['dob'] = pd.to_datetime(data['dob'])
        data['driver_age'] = (data['date'] - data['dob']).dt.days / 365.25
        data['finished'] = (data['position'].notna()).astype(int)
        data['points_scored'] = data['points']

        if use_merged_cache:
            self._save_merged_features_cache(data)

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
