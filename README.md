# 🏁 F1 Race Prediction System

A comprehensive machine learning system for predicting Formula 1 race results based on historical data, featuring weather impact analysis, a GP simulator, and an interactive dashboard.

## Features

### 1. **Intelligent Prediction Models**
- **Points Prediction**: Gradient Boosting model to predict driver points scored
- **Position Prediction**: Random Forest model for finishing position
- **Finish Probability**: Classifier to determine race completion likelihood

### 2. **Interactive Dashboard**
Built with Streamlit featuring:
- **Race Predictions**: Real-time race outcome simulations
- **Season Simulator**: Full championship predictions
- **Historical Analytics**: Trend analysis and statistics
- **Model Performance**: Metrics and feature importance visualization
- **Weather Impact**: Analyze how conditions affect race outcomes

### 3. **GP Simulator**
- Single race simulations with customizable conditions
- Full season championship simulations
- Weather factor modeling (0.5 = Very Wet, 2.0 = Extreme Heat)
- Safety car impact simulation
- FIA points system implementation

### 4. **Advanced Features**
- Rolling statistics (5-race averages)
- Circuit-specific analysis
- Driver age and experience factors
- Constructor performance tracking
- Historical trend analysis

## Installation

### Requirements
- Python 3.8+
- pip or conda

### Setup

```bash
# Clone or navigate to the project directory
cd /path/to/VVA

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Option 1: Run the Interactive Dashboard

```bash
python main.py
```
The dashboard will be available at `http://localhost:8501`

### Option 2: Train Models Only

```bash
python main.py --train-only
```
This trains all prediction models and saves them to the `models/` directory.

### Option 3: Run Quick Simulation

```bash
python main.py --simulate
```
Runs a sample race simulation with 10 drivers.

## Project Structure

```
VVA/
├── main.py                 # Entry point
├── data_loader.py         # F1 data loading and preprocessing
├── feature_engineer.py    # Feature engineering and dataset creation
├── model_trainer.py       # Model training (Points, Position, Finish)
├── gp_simulator.py        # Race and season simulation engine
├── app.py                 # Streamlit dashboard
├── requirements.txt       # Python dependencies
├── models/               # Trained models (generated after training)
│   ├── points_model.pkl
│   ├── position_model.pkl
│   ├── finish_model.pkl
│   └── feature_cols.pkl
└── csv/                  # F1 historical data
    ├── races.csv
    ├── drivers.csv
    ├── results.csv
    ├── constructors.csv
    ├── circuits.csv
    ├── driver_standings.csv
    ├── constructor_standings.csv
    ├── qualifying.csv
    ├── pit_stops.csv
    ├── lap_times.csv
    ├── status.csv
    └── ...
```

## Module Documentation

### data_loader.py
**Class: `F1DataLoader`**
- `load_all_data()`: Load all CSV files
- `get_race_features()`: Create merged dataset with race features
- `get_driver_stats()`: Aggregate driver statistics
- `get_constructor_stats()`: Aggregate constructor statistics
- `get_circuit_stats()`: Calculate circuit-specific metrics

### feature_engineer.py
**Class: `FeatureEngineer`**
- `create_training_data(lookback_races=5)`: Generate features from historical data
- `get_feature_matrix()`: Extract features for model training
- `get_position_prediction_data()`: Prepare position prediction dataset
- `preprocess_features()`: Normalize feature values

### model_trainer.py
**Class: `F1Predictor`**
- `train_models()`: Train all three models
- `predict_points()`: Predict points for a driver
- `predict_position()`: Predict finishing position
- `predict_finish()`: Get finish probability
- `save_models()`: Save trained models to disk
- `load_models()`: Load pre-trained models

### gp_simulator.py
**Class: `GPSimulator`**
- `simulate_race()`: Simulate single race with weather and conditions
- `simulate_season()`: Simulate full championship
- `_create_feature_vector()`: Generate features for prediction
- `_get_points()`: Apply FIA points system

### app.py
Streamlit dashboard with 5 pages:
1. **Dashboard**: Overview and historical statistics
2. **Race Simulator**: Single race simulations
3. **Season Simulator**: Full championship predictions
4. **Analytics**: Detailed performance analysis
5. **Model Performance**: Model metrics and validation

## Features & Features Engineering

### Input Features
- Grid Position
- Driver Age
- Driver Rolling Average Points (5-race)
- Driver Finish Rate
- Driver Average Grid Position
- Constructor Rolling Average Points
- Constructor Finish Rate
- Circuit Average Points
- Circuit Average Grid Position
- Season Year
- Race Round

### Target Variables
- **Points**: Points scored (0-25)
- **Position**: Finishing position (1-20)
- **Finish**: Binary finish indicator (0/1)

## Machine Learning Models

### 1. Points Prediction Model
- **Type**: Gradient Boosting Regressor
- **Algorithm**: GBM with 100 estimators
- **Output**: Predicted points (0-25)
- **Typical R² Score**: 0.65-0.75

### 2. Position Prediction Model
- **Type**: Random Forest Regressor
- **Algorithm**: RF with 100 trees, max depth 10
- **Output**: Predicted finishing position (1-20)
- **Typical R² Score**: 0.40-0.50

### 3. Finish Prediction Model
- **Type**: Random Forest Classifier
- **Algorithm**: RF with 50 trees
- **Output**: Probability of finishing (0-1)
- **Typical Accuracy**: 0.75-0.85

## Weather Impact System

The simulator includes a weather factor (0.5 to 2.0):
- **0.5 (Very Wet)**: Reduces finishing probability, worsens position
- **1.0 (Normal)**: Baseline conditions
- **2.0 (Extreme Heat)**: May reduce performance variability

## FIA Points System

Championship points awarded:
- P1: 25 points
- P2: 18 points
- P3: 15 points
- P4: 12 points
- P5: 10 points
- P6-10: 8, 6, 4, 2, 1 points
- P11+: 0 points

## Example: Single Race Simulation

```python
from data_loader import F1DataLoader
from feature_engineer import FeatureEngineer
from model_trainer import F1Predictor
from gp_simulator import GPSimulator

# Load and train
loader = F1DataLoader()
loader.load_all_data()

engineer = FeatureEngineer(loader)
engineer.create_training_data()

predictor = F1Predictor()
predictor.train_models(loader)

# Simulate race
simulator = GPSimulator(predictor, loader, engineer)

drivers = [
    {'driver_id': 1, 'driver_name': 'Hamilton', 'grid': 1, 
     'constructor_id': 1, 'constructor': 'Mercedes'},
    {'driver_id': 2, 'driver_name': 'Verstappen', 'grid': 2, 
     'constructor_id': 2, 'constructor': 'Red Bull'},
]

results = simulator.simulate_race(
    circuit_id=1,
    drivers_info=drivers,
    weather_factor=1.0,
    safety_car=False
)

print(results)
```

## Example: Season Simulation

```python
races_data = [
    {
        'circuit_id': i,
        'drivers_info': drivers,
        'weather_factor': 1.0
    }
    for i in range(1, 24)
]

championship = simulator.simulate_season(races_data, num_simulations=5)
print(championship[['driver_name', 'points', 'wins']])
```

## Dashboard Walkthrough

### Dashboard Page
- Overview metrics (total races, drivers, constructors)
- Historical race statistics
- Top drivers by career points
- Races per season trends

### Race Simulator Page
- Circuit selection
- Driver grid configuration
- Weather and safety car settings
- Race result predictions
- Performance visualization

### Season Simulator Page
- Multi-race championship simulation
- Driver standing predictions
- Win probability calculations
- Performance visualization

### Analytics Page
- Grid position impact analysis
- Finish rate by starting position
- Constructor performance trends
- Historical statistics

### Model Performance Page
- Model type information
- Prediction accuracy metrics
- Feature importance rankings
- Prediction vs actual comparisons

## Performance Tips

1. **Training Time**: Initial model training takes ~2-5 minutes depending on hardware
2. **Caching**: Streamlit caches model loading (decorated with `@st.cache_resource`)
3. **Large Datasets**: The system efficiently processes 70+ years of F1 history
4. **Real-time Updates**: Simulations run in <1 second per race

## Future Enhancements

- [ ] Weather data integration from external APIs
- [ ] Pit stop strategy optimization
- [ ] Tire degradation modeling
- [ ] Real-time race updates
- [ ] Qualifying prediction model
- [ ] Driver head-to-head comparison
- [ ] Historical race replay analysis
- [ ] Custom calendar creator
- [ ] Mobile app version

## Data Sources

Historical F1 data from:
- Official Formula 1 website
- Ergast Formula 1 Developer API
- Historical race databases

## License

This project is for educational purposes.

## Author

Created for the F1 Prediction System project.

---

**Enjoy predicting F1 race outcomes! 🏁**
