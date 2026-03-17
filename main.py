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
from pathlib import Path

from data_loader import F1DataLoader
from feature_engineer import FeatureEngineer
from model_trainer import F1Predictor
from gp_simulator import GPSimulator


def train_models():
    """Train all prediction models"""
    print("=" * 60)
    print("F1 RACE PREDICTION SYSTEM - Model Training")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading F1 historical data...")
    loader = F1DataLoader()
    loader.load_all_data()
    
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


def run_dashboard():
    """Run Streamlit dashboard"""
    import subprocess
    print("Starting Streamlit dashboard...")
    print("Access the app at: http://localhost:8501")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])


def run_quick_simulation():
    """Run a quick simulation example"""
    print("\n" + "=" * 60)
    print("F1 RACE SIMULATION - Quick Example")
    print("=" * 60)
    
    loader = F1DataLoader()
    loader.load_all_data()
    
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
    
    # Create sample race
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
    
    args = parser.parse_args()
    
    if args.train_only:
        train_models()
    elif args.simulate:
        run_quick_simulation()
    else:
        # Check if models exist, train if not
        if not Path('models/points_model.pkl').exists():
            print("Models not found. Training new models...")
            train_models()
        
        # Run dashboard
        run_dashboard()


if __name__ == '__main__':
    main()
