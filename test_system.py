#!/usr/bin/env python3
"""
Test script to validate the F1 Prediction System
"""
import sys
import os
from pathlib import Path


def test_imports():
    """Test that all required modules can be imported"""
    print("\n" + "=" * 60)
    print("TESTING MODULE IMPORTS")
    print("=" * 60)
    
    modules = [
        'data_loader',
        'feature_engineer',
        'model_trainer',
        'gp_simulator',
        'config',
        # Skip app.py in tests due to Streamlit context
    ]
    
    all_ok = True
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")
            all_ok = False
    
    return all_ok


def test_data_loading():
    """Test data loading"""
    print("\n" + "=" * 60)
    print("TESTING DATA LOADING")
    print("=" * 60)
    
    try:
        from data_loader import F1DataLoader
        
        loader = F1DataLoader()
        loader.load_all_data()
        
        print(f"✓ Races loaded: {len(loader.races)}")
        print(f"✓ Drivers loaded: {len(loader.drivers)}")
        print(f"✓ Results loaded: {len(loader.results)}")
        print(f"✓ Constructors loaded: {len(loader.constructors)}")
        print(f"✓ Circuits loaded: {len(loader.circuits)}")
        
        return True, loader
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False, None


def test_feature_engineering(loader):
    """Test feature engineering"""
    print("\n" + "=" * 60)
    print("TESTING FEATURE ENGINEERING")
    print("=" * 60)
    
    try:
        from feature_engineer import FeatureEngineer
        
        engineer = FeatureEngineer(loader)
        data = engineer.create_training_data()
        
        print(f"✓ Training data created: {len(data)} records")
        print(f"✓ Features: {len(data.columns)} columns")
        
        X, y, features = engineer.get_feature_matrix()
        print(f"✓ Feature matrix shape: {X.shape}")
        print(f"✓ Target shape: {y.shape}")
        
        return True, engineer
    except Exception as e:
        print(f"✗ Feature engineering failed: {e}")
        return False, None


def test_model_training(loader, engineer):
    """Test model training"""
    print("\n" + "=" * 60)
    print("TESTING MODEL TRAINING")
    print("=" * 60)
    
    try:
        from model_trainer import F1Predictor
        
        predictor = F1Predictor()
        
        # Try to load existing models
        if Path('models/points_model.pkl').exists():
            print("Loading existing models...")
            predictor.load_models()
            print("✓ Models loaded from disk")
        else:
            print("Training new models...")
            predictor.train_models(loader)
            print("✓ Models trained successfully")
            predictor.save_models()
            print("✓ Models saved to disk")
        
        return True, predictor
    except Exception as e:
        print(f"✗ Model training failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_simulation(predictor, loader, engineer):
    """Test race simulation"""
    print("\n" + "=" * 60)
    print("TESTING RACE SIMULATION")
    print("=" * 60)
    
    try:
        from gp_simulator import GPSimulator
        
        simulator = GPSimulator(predictor, loader, engineer)
        
        # Test single race simulation
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
        
        print(f"✓ Single race simulated: {len(results)} drivers")
        print(f"  - Finished: {results['finished'].sum()} drivers")
        print(f"  - Top scorer: {results.iloc[0]['driver_name']} with {results.iloc[0]['predicted_points']} points")
        
        # Test season simulation
        races_data = [
            {
                'circuit_id': (i % 24) + 1,
                'drivers_info': drivers,
                'weather_factor': 1.0
            }
            for i in range(3)
        ]
        
        championship = simulator.simulate_season(races_data, num_simulations=1)
        print(f"✓ Season simulated: {len(championship)} drivers")
        print(f"  - Champion: {championship.iloc[0]['driver_name']} with {championship.iloc[0]['points']} points")
        
        return True
    except Exception as e:
        print(f"✗ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """Test configuration"""
    print("\n" + "=" * 60)
    print("TESTING CONFIGURATION")
    print("=" * 60)
    
    try:
        from config import Config, WeatherSimulator, CircuitCharacteristics
        
        Config.ensure_dirs()
        print(f"✓ Directories ensured")
        print(f"  - Data dir: {Config.DATA_DIR}")
        print(f"  - Models dir: {Config.MODELS_DIR}")
        
        weather = WeatherSimulator.get_weather_factor('light_rain')
        print(f"✓ Weather system working: light_rain factor = {weather}")
        
        circuit = CircuitCharacteristics.get_circuit_info(1)
        print(f"✓ Circuit info: {circuit['name']} ({circuit['country']})")
        
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("F1 PREDICTION SYSTEM - VALIDATION TESTS")
    print("=" * 60)
    
    results = {}
    
    # Test imports
    results['imports'] = test_imports()
    
    if not results['imports']:
        print("\n⚠ Cannot continue - import failed")
        return
    
    # Test configuration
    results['config'] = test_config()
    
    # Test data loading
    ok, loader = test_data_loading()
    results['data_loading'] = ok and loader is not None
    
    if not results['data_loading']:
        print("\n⚠ Cannot continue - data loading failed")
        return
    
    # Test feature engineering
    ok, engineer = test_feature_engineering(loader)
    results['feature_engineering'] = ok and engineer is not None
    
    if not results['feature_engineering']:
        print("\n⚠ Cannot continue - feature engineering failed")
        return
    
    # Test model training
    ok, predictor = test_model_training(loader, engineer)
    results['model_training'] = ok and predictor is not None
    
    if not results['model_training']:
        print("\n⚠ Cannot continue - model training failed")
        return
    
    # Test simulation
    if predictor:
        results['simulation'] = test_simulation(predictor, loader, engineer)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test.replace('_', ' ').title()}: {status}")
    
    if all(results.values()):
        print("\n✓ ALL TESTS PASSED!")
        print("\nYou can now run the system with:")
        print("  python main.py          # Run the dashboard")
        print("  python main.py --simulate  # Run a quick simulation")
    else:
        print("\n✗ SOME TESTS FAILED - Please check the errors above")
        sys.exit(1)


if __name__ == '__main__':
    main()
