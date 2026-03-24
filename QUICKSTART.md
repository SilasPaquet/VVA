# 🏁 F1 Prediction System - Quick Start Guide

## Installation & Setup (2 minutes)

```bash
# Navigate to the project directory
cd /path/to/VVA

# Install dependencies
pip install -r requirements.txt

# Run the test suite to verify installation
python test_system.py
```

## Running the System

### Option 1: Interactive Dashboard (Recommended)
```bash
python main.py
```
Access at: `http://localhost:8501`

**Features:**
- 📊 Dashboard: Historical statistics and analysis
- 🎯 Race Simulator: Predict individual race results
- 🏆 Season Simulator: Predict full championship standings
- 📈 Analytics: Deep dive into performance metrics
- 🤖 Model Performance: View model metrics and accuracy

### Option 2: Quick Simulation
```bash
python main.py --simulate
```
Runs a sample 10-driver race simulation and displays results.

### Option 3: Train Models Only
```bash
python main.py --train-only
```
Trains models and saves them to `models/` directory (takes 2-5 minutes).

## Cleaned Data Cache (Fast Re-runs)

The loader saves cleaned files in `csv/cleaned/` and reuses them by default.

```bash
# Default behavior: use cleaned cache if present
python main.py

# Force rebuild cleaned cache from raw csv/*.csv
python main.py --force-rebuild-data

# Train only + force rebuild
python main.py --train-only --force-rebuild-data

# Ignore cleaned cache for this run
python main.py --no-clean-cache
```

Use `--force-rebuild-data` when raw files changed or cleaning rules were updated.

## Dashboard Pages

### 1. **Dashboard**
- Overview metrics
- Historical race trends
- Top drivers by career points
- Track record analysis

### 2. **Race Simulator**
- Customize driver grid
- Adjust weather conditions
- Add safety car intervention
- Get race predictions
- View position changes from grid to finish

### 3. **Season Simulator**
- Configure number of races
- Set number of drivers
- Run multiple simulations for averaging
- View championship standings
- Analyze win probabilities

### 4. **Analytics**
- Grid position impact on final position
- Finish rates by starting position
- Constructor performance trends
- Circuit-specific statistics

### 5. **Model Performance**
- Model accuracy metrics (R², RMSE, MAE)
- Feature importance rankings
- Prediction vs actual comparison charts
- Model type information

## Key Features

### ML Models Included
1. **Points Prediction** (Gradient Boosting)
   - Predicts points earned by driver
   - R² Score: ~0.63
   
2. **Position Prediction** (Random Forest)
   - Predicts finishing position
   - R² Score: ~0.23
   
3. **Finish Prediction** (Random Forest Classifier)
   - Predicts probability of completing race
   - Accuracy: ~100% (conservative)

### Simulation Capabilities
✓ Single race simulation with weather effects  
✓ Full season championship predictions  
✓ Safety car impact modeling  
✓ FIA points system implementation  
✓ Driver and constructor performance tracking  
✓ Circuit-specific analysis  

### Weather System
- **0.5 (Very Wet)**: Challenging conditions, worse finishes
- **1.0 (Normal)**: Baseline conditions
- **2.0 (Extreme Heat)**: High tire degradation

## Data Overview

The system uses 70+ years of F1 history:
- **1,125 races** analyzed
- **859 different drivers**
- **212 constructors** tracked
- **77 circuits** worldwide
- **26,500+ race results**

## Example Usage

### Simulate a Race

```python
from data_loader import F1DataLoader
from feature_engineer import FeatureEngineer
from model_trainer import F1Predictor
from gp_simulator import GPSimulator

# Initialize
loader = F1DataLoader()
loader.load_all_data()

engineer = FeatureEngineer(loader)
engineer.create_training_data()

predictor = F1Predictor()
predictor.load_models()  # or train_models(loader)

simulator = GPSimulator(predictor, loader, engineer)

# Simulate race
drivers = [
    {'driver_id': 1, 'driver_name': 'Hamilton', 'grid': 1, 
     'constructor_id': 1, 'constructor': 'Mercedes'},
    {'driver_id': 2, 'driver_name': 'Verstappen', 'grid': 2, 
     'constructor_id': 2, 'constructor': 'Red Bull'},
]

results = simulator.simulate_race(
    circuit_id=1,
    drivers_info=drivers,
    weather_factor=0.8,  # Light rain
    safety_car=False
)

print(results[['driver_name', 'predicted_position', 'predicted_points']])
```

### Simulate a Season

```python
races_data = []
for i in range(24):  # 24 race season
    races_data.append({
        'circuit_id': i % 77,
        'drivers_info': drivers,
        'weather_factor': 1.0
    })

championship = simulator.simulate_season(races_data, num_simulations=5)
print(championship[['driver_name', 'points', 'wins']])
```

## Project Structure

```
VVA/
├── main.py                 # Entry point
├── app.py                  # Streamlit dashboard
├── data_loader.py         # Data loading & preprocessing
├── feature_engineer.py    # Feature engineering
├── model_trainer.py       # ML models
├── gp_simulator.py        # Race simulator
├── config.py              # Configuration & utilities
├── test_system.py         # Test suite
├── requirements.txt       # Dependencies
├── README.md              # Full documentation
└── csv/                   # Historical F1 data (12 CSV files)
```

## Performance Notes

- **First run**: Models train in 2-5 minutes
- **Subsequent runs**: Models load from disk (~1 second)
- **Single race simulation**: <1 second
- **Season simulation**: 2-10 seconds (depending on number of races)
- **Dashboard**: Real-time, responsive interface

## Troubleshooting

### "Model files not found"
→ Run `python main.py --train-only` to train models

### "Port 8501 already in use"
→ Kill existing Streamlit process: `pkill -f streamlit`

### "CSV files not found"
→ Ensure `csv/` folder exists with all historical data files

### "Memory issues with large simulations"
→ Reduce number of simulations or races

## Next Steps

1. ✅ Run tests: `python test_system.py`
2. 🚀 Start dashboard: `python main.py`
3. 🎯 Try race simulations
4. 📊 Explore analytics
5. 🏆 Predict championships

## Have Questions?

Check the README.md for detailed documentation on:
- Feature engineering process
- Model training details
- API usage examples
- Advanced configuration

---

**Enjoy predicting F1 race outcomes! 🏁**
