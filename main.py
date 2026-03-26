#!/usr/bin/env python3
"""
F1 Race Prediction System - Main Entry Point

This system predicts F1 race results based on historical data using machine learning.
It includes:
- Dashboard for visualization
- Race simulator for single races
- Season simulator for full championship predictions
- Analytics and model performance tracking

Usage:
    python main.py                 # Run the Streamlit dashboard
    python main.py --train-only    # Train models without running dashboard
    python main.py --simulate      # Run simulations
"""

import sys
import argparse
import os
from pathlib import Path

from data_loader import F1DataLoader
from feature_engineer import FeatureEngineer
from model_trainer import F1Predictor
from gp_simulator import GPSimulator


def train_models(force_rebuild_data=False, use_clean_cache=True):
    """Train all prediction models"""
    print("=" * 60)
    print("F1 RACE PREDICTION SYSTEM - Model Training")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading F1 historical data...")
    loader = F1DataLoader()
    loader.load_all_data(
        use_clean_cache=use_clean_cache,
        force_rebuild=force_rebuild_data
    )
    
    print(f"   ✓ Loaded {len(loader.races)} races")
    print(f"   ✓ Loaded {len(loader.drivers)} drivers")
    print(f"   ✓ Loaded {len(loader.results)} race results")
    print(f"   ✓ Date range: {loader.races['year'].min()}-{loader.races['year'].max()}")
    
    # Engineer features
    print("\n2. Engineering features...")
    engineer = FeatureEngineer(loader)
    data = engineer.create_training_data()
    print(f"   ✓ Created training dataset with {len(data)} records")
    print(f"   ✓ Feature dimensions: {len(data.columns)} features")
    
    # Train models
    print("\n3. Training prediction models...")
    predictor = F1Predictor()
    predictor.train_models(loader)
    
    # Save models
    print("\n4. Saving models...")
    predictor.save_models()
    print("   ✓ Models saved to 'models/' directory")
    
    print("\n" + "=" * 60)
    print("✓ Model training complete!")
    print("=" * 60)
    return loader, engineer, predictor


def run_dashboard(force_rebuild_data=False, use_clean_cache=True):
    """Run Streamlit dashboard"""
    import subprocess
    print("Starting Streamlit dashboard...")
    print("Access the app at: http://localhost:8501")
    env = os.environ.copy()
    env['F1_FORCE_REBUILD_DATA'] = '1' if force_rebuild_data else '0'
    env['F1_USE_CLEAN_CACHE'] = '1' if use_clean_cache else '0'
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], env=env)


def run_quick_simulation(force_rebuild_data=False, use_clean_cache=True):
    """Run a quick simulation example"""
    print("\n" + "=" * 60)
    print("F1 RACE SIMULATION - Quick Example")
    print("=" * 60)
    
    loader = F1DataLoader()
    loader.load_all_data(
        use_clean_cache=use_clean_cache,
        force_rebuild=force_rebuild_data
    )
    
    engineer = FeatureEngineer(loader)
    engineer.create_training_data()
    
    predictor = F1Predictor()
    try:
        predictor.load_models()
    except FileNotFoundError:
        print("Models not found. Training new models...")
        predictor.train_models(loader)
        predictor.save_models()
    
    simulator = GPSimulator(predictor, loader, engineer)
    
    print("\nSimulating race at Circuit ID 1 with 10 drivers...")
    drivers = [
        {
            'driver_id': i,
            'driver_name': f'Driver {i}',
            'grid': i + 1,
            'constructor_id': (i % 10) + 1,
            'constructor': f'Team {(i % 10) + 1}'
        }
        for i in range(10)
    ]
    
    results = simulator.simulate_race(
        circuit_id=1,
        drivers_info=drivers,
        weather_factor=1.0,
        safety_car=False
    )
    
    print("\n" + "=" * 60)
    print("RACE RESULTS")
    print("=" * 60)
    print(results[['driver_name', 'grid_position', 'predicted_position', 
                  'predicted_points', 'finish_probability']].to_string(index=False))
    print("=" * 60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="F1 Race Prediction System"
    )
    parser.add_argument(
        '--train-only',
        action='store_true',
        help='Train models without running dashboard'
    )
    parser.add_argument(
        '--simulate',
        action='store_true',
        help='Run a quick simulation example'
    )
    parser.add_argument(
        '--force-rebuild-data',
        action='store_true',
        help='Rebuild cleaned data cache from raw CSV files before running'
    )
    parser.add_argument(
        '--no-clean-cache',
        action='store_true',
        help='Disable cleaned cache usage and read raw CSV files'
    )
    
    args = parser.parse_args()
    
    use_clean_cache = not args.no_clean_cache

    if args.train_only:
        train_models(
            force_rebuild_data=args.force_rebuild_data,
            use_clean_cache=use_clean_cache
        )
    elif args.simulate:
        run_quick_simulation(
            force_rebuild_data=args.force_rebuild_data,
            use_clean_cache=use_clean_cache
        )
    else:
        if not Path('models/points_model.pkl').exists():
            print("Models not found. Training new models...")
            train_models(
                force_rebuild_data=args.force_rebuild_data,
                use_clean_cache=use_clean_cache
            )

        run_dashboard(
            force_rebuild_data=args.force_rebuild_data,
            use_clean_cache=use_clean_cache
        )


if __name__ == '__main__':
    main()
