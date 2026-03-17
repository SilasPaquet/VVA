"""
Streamlit Dashboard for F1 Race Prediction System
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

from data_loader import F1DataLoader
from feature_engineer import FeatureEngineer
from model_trainer import F1Predictor
from gp_simulator import GPSimulator


# Page configuration
st.set_page_config(page_title="F1 Race Predictor", layout="wide")

# Sidebar navigation
with st.sidebar:
    st.title("🏁 F1 Predictor")
    page = st.radio("Navigation", [
        "Dashboard",
        "Race Simulator",
        "Season Simulator",
        "Analytics",
        "Model Performance"
    ])

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'loader' not in st.session_state:
    st.session_state.loader = None
if 'engineer' not in st.session_state:
    st.session_state.engineer = None


@st.cache_resource
def load_and_train_models():
    """Load data and train models"""
    with st.spinner("Loading and training models..."):
        loader = F1DataLoader()
        loader.load_all_data()
        
        engineer = FeatureEngineer(loader)
        engineer.create_training_data()
        
        predictor = F1Predictor()
        if os.path.exists('models/points_model.pkl'):
            predictor.load_models()
        else:
            predictor.train_models(loader)
            predictor.save_models()
        
        return loader, engineer, predictor


def get_driver_team_options(loader):
    """Build selectable driver/team options and latest known team mapping."""
    drivers_df = loader.drivers[['driverId', 'forename', 'surname']].copy()
    drivers_df['driver_name'] = (drivers_df['forename'] + ' ' + drivers_df['surname']).str.strip()
    drivers_df = drivers_df.sort_values('driver_name').reset_index(drop=True)

    constructors_df = loader.constructors[['constructorId', 'name']].copy()
    constructors_df = constructors_df.drop_duplicates(subset=['constructorId'])
    constructors_df = constructors_df.sort_values('name').reset_index(drop=True)

    latest_results = loader.results.merge(loader.races[['raceId', 'date']], on='raceId', how='left')
    latest_results['date'] = pd.to_datetime(latest_results['date'], errors='coerce')
    latest_results = latest_results.sort_values('date')
    latest_results = latest_results.dropna(subset=['driverId', 'constructorId'])
    latest_results = latest_results.drop_duplicates(subset=['driverId'], keep='last')

    latest_team_by_driver = {
        int(row['driverId']): int(row['constructorId'])
        for _, row in latest_results[['driverId', 'constructorId']].iterrows()
    }

    return drivers_df, constructors_df, latest_team_by_driver


# Load models
loader, engineer, predictor = load_and_train_models()
st.session_state.loader = loader
st.session_state.engineer = engineer
st.session_state.predictor = predictor


# Page content
if page == "Dashboard":
    st.title("🏁 F1 Race Predictions Dashboard")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Races", len(loader.races))
    with col2:
        st.metric("Total Drivers", len(loader.drivers))
    with col3:
        st.metric("Total Constructors", len(loader.constructors))
    
    st.divider()
    
    # Historical data visualization
    st.subheader("📊 Historical Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Races per year
        races_per_year = loader.races.groupby('year').size()
        fig = px.line(
            x=races_per_year.index,
            y=races_per_year.values,
            labels={'x': 'Year', 'y': 'Number of Races'},
            title='Races per Season'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Drivers per year
        drivers_per_year = loader.results.merge(
            loader.races[['raceId', 'year']], 
            on='raceId'
        ).groupby('year')['driverId'].nunique()
        
        fig = px.line(
            x=drivers_per_year.index,
            y=drivers_per_year.values,
            labels={'x': 'Year', 'y': 'Number of Drivers'},
            title='Active Drivers per Season'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Top drivers
    st.subheader("🏆 Top Drivers (Historical)")
    top_drivers = loader.results.groupby('driverId').agg({
        'points': 'sum'
    }).nlargest(10, 'points').reset_index()
    
    top_drivers = top_drivers.merge(
        loader.drivers[['driverId', 'forename', 'surname']],
        on='driverId'
    )
    top_drivers['name'] = top_drivers['forename'] + ' ' + top_drivers['surname']
    
    fig = px.bar(
        top_drivers,
        x='points',
        y='name',
        orientation='h',
        title='Top 10 Drivers by Career Points',
        labels={'points': 'Total Points', 'name': 'Driver'}
    )
    st.plotly_chart(fig, use_container_width=True)


elif page == "Race Simulator":
    st.title("🎯 Single Race Simulator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Race Configuration")
        
        # Select circuit
        circuits = loader.circuits.sort_values('name')
        selected_circuit = st.selectbox(
            "Select Circuit",
            circuits['name'],
            key='circuit'
        )
        circuit_id = circuits[circuits['name'] == selected_circuit]['circuitId'].values[0]
        
        # Driver grid
        st.subheader("Grid Position")
        num_drivers = st.slider("Number of drivers", 2, 20, 10)
    
    with col2:
        st.subheader("Weather & Conditions")
        
        weather_factor = st.slider(
            "Weather Impact",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            help="0.5=Very Wet, 1.0=Normal, 2.0=Extreme Heat"
        )
        
        safety_car = st.checkbox("Safety Car Intervention")
    
    # Driver setup
    st.subheader("Drivers in Race")
    st.caption("Pick specific drivers and teams from your historical dataset.")

    driver_options_df, constructor_options_df, latest_team_by_driver = get_driver_team_options(loader)

    driver_labels = driver_options_df['driver_name'].tolist()
    driver_ids = driver_options_df['driverId'].astype(int).tolist()
    driver_label_to_id = dict(zip(driver_labels, driver_ids))

    constructor_names = constructor_options_df['name'].tolist()
    constructor_ids = constructor_options_df['constructorId'].astype(int).tolist()
    constructor_name_to_id = dict(zip(constructor_names, constructor_ids))

    drivers_info = []
    for i in range(num_drivers):
        col1, col2, col3 = st.columns(3)

        default_driver_index = i if i < len(driver_labels) else 0
        with col1:
            selected_driver_label = st.selectbox(
                f"Driver {i+1}",
                driver_labels,
                index=default_driver_index,
                key=f"driver_select_{i}"
            )
        selected_driver_id = driver_label_to_id[selected_driver_label]

        with col2:
            grid = st.number_input(
                f"Grid Position {i+1}",
                min_value=1,
                max_value=20,
                value=i + 1,
                key=f"grid_input_{i}"
            )

        default_constructor_id = latest_team_by_driver.get(selected_driver_id, constructor_ids[0])
        try:
            default_constructor_index = constructor_ids.index(default_constructor_id)
        except ValueError:
            default_constructor_index = 0

        with col3:
            selected_team_name = st.selectbox(
                f"Team {i+1}",
                constructor_names,
                index=default_constructor_index,
                key=f"team_select_{i}"
            )
        selected_constructor_id = constructor_name_to_id[selected_team_name]

        drivers_info.append({
            'driver_id': selected_driver_id,
            'driver_name': selected_driver_label,
            'grid': int(grid),
            'constructor_id': selected_constructor_id,
            'constructor': selected_team_name
        })

    selected_driver_ids = [entry['driver_id'] for entry in drivers_info]
    selected_grids = [entry['grid'] for entry in drivers_info]
    has_duplicate_drivers = len(set(selected_driver_ids)) != len(selected_driver_ids)
    has_duplicate_grids = len(set(selected_grids)) != len(selected_grids)

    if has_duplicate_drivers:
        st.warning("You selected the same driver more than once.")
    if has_duplicate_grids:
        st.warning("Two or more drivers share the same grid position.")

    if st.button("🏁 Simulate Race"):
        if has_duplicate_drivers:
            st.error("Please select unique drivers before running the simulation.")
        elif has_duplicate_grids:
            st.error("Please assign unique grid positions before running the simulation.")
        else:
            simulator = GPSimulator(predictor, loader, engineer)
            results = simulator.simulate_race(
                circuit_id=circuit_id,
                drivers_info=drivers_info,
                weather_factor=weather_factor,
                safety_car=safety_car
            )

            st.success("Race Simulation Complete!")
            st.dataframe(
                results[['driver_name', 'grid_position', 'predicted_position',
                        'predicted_points', 'actual_points', 'finish_probability']],
                use_container_width=True
            )

            # Results visualization
            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(
                    results[results['finished']],
                    x='driver_name',
                    y='predicted_points',
                    title='Predicted Points',
                    labels={'driver_name': 'Driver', 'predicted_points': 'Points'}
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.scatter(
                    results,
                    x='grid_position',
                    y='predicted_position',
                    hover_data=['driver_name'],
                    title='Grid vs Predicted Position',
                    labels={'grid_position': 'Grid Position', 'predicted_position': 'Predicted Position'}
                )
                st.plotly_chart(fig, use_container_width=True)


elif page == "Season Simulator":
    st.title("🏆 Season Simulator")
    
    st.info("This feature allows you to simulate an entire F1 season with multiple races.")

    col1, col2, col3 = st.columns(3)

    with col1:
        num_races = st.slider("Number of Races", 2, 23, 10)
    with col2:
        num_simulations = st.slider("Simulations", 1, 10, 1)
    with col3:
        num_season_drivers = st.slider("Number of drivers in season", 5, 20, 10)

    weather_mode = st.selectbox(
        "Weather Mode",
        ["Mixed (random)", "Dry (stable)", "Wet (stable)"],
        index=0
    )

    st.subheader("Season Driver Roster")
    st.caption("Select real drivers and constructors from the historical dataset.")

    driver_options_df, constructor_options_df, latest_team_by_driver = get_driver_team_options(loader)

    driver_labels = driver_options_df['driver_name'].tolist()
    driver_ids = driver_options_df['driverId'].astype(int).tolist()
    driver_label_to_id = dict(zip(driver_labels, driver_ids))

    constructor_names = constructor_options_df['name'].tolist()
    constructor_ids = constructor_options_df['constructorId'].astype(int).tolist()
    constructor_name_to_id = dict(zip(constructor_names, constructor_ids))

    season_roster = []
    for i in range(num_season_drivers):
        roster_col1, roster_col2 = st.columns(2)

        default_driver_index = i if i < len(driver_labels) else 0
        with roster_col1:
            selected_driver_label = st.selectbox(
                f"Driver {i+1}",
                driver_labels,
                index=default_driver_index,
                key=f"season_driver_select_{i}"
            )
        selected_driver_id = driver_label_to_id[selected_driver_label]

        default_constructor_id = latest_team_by_driver.get(selected_driver_id, constructor_ids[0])
        try:
            default_constructor_index = constructor_ids.index(default_constructor_id)
        except ValueError:
            default_constructor_index = 0

        with roster_col2:
            selected_team_name = st.selectbox(
                f"Team {i+1}",
                constructor_names,
                index=default_constructor_index,
                key=f"season_team_select_{i}"
            )
        selected_constructor_id = constructor_name_to_id[selected_team_name]

        season_roster.append({
            'driver_id': selected_driver_id,
            'driver_name': selected_driver_label,
            'constructor_id': selected_constructor_id,
            'constructor': selected_team_name
        })

    selected_driver_ids = [entry['driver_id'] for entry in season_roster]
    has_duplicate_drivers = len(set(selected_driver_ids)) != len(selected_driver_ids)
    if has_duplicate_drivers:
        st.warning("You selected the same driver more than once in the season roster.")

    if st.button("🏁 Simulate Season"):
        if has_duplicate_drivers:
            st.error("Please select unique drivers before running the season simulation.")
        else:
            with st.spinner("Simulating season..."):
                races_data = []
                circuit_ids = loader.circuits['circuitId'].dropna().astype(int).tolist()
                if not circuit_ids:
                    circuit_ids = [1]

                for race_num in range(num_races):
                    grid_positions = np.random.permutation(np.arange(1, num_season_drivers + 1))

                    drivers_for_race = []
                    for idx, base_driver in enumerate(season_roster):
                        race_driver = base_driver.copy()
                        race_driver['grid'] = int(grid_positions[idx])
                        drivers_for_race.append(race_driver)

                    if weather_mode == "Dry (stable)":
                        weather_factor = 1.0
                    elif weather_mode == "Wet (stable)":
                        weather_factor = 0.8
                    else:
                        weather_factor = float(np.random.uniform(0.8, 1.2))

                    races_data.append({
                        'circuit_id': circuit_ids[race_num % len(circuit_ids)],
                        'drivers_info': drivers_for_race,
                        'weather_factor': weather_factor
                    })

                simulator = GPSimulator(predictor, loader, engineer)
                championship = simulator.simulate_season(races_data, num_simulations)

                st.success("Season Simulation Complete!")
                st.dataframe(
                    championship[['position', 'driver_name', 'constructor', 'points', 'wins']],
                    use_container_width=True
                )

                # Visualization
                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:
                    fig = px.bar(
                        championship.head(10),
                        x='driver_name',
                        y='points',
                        title='Top 10 Championship Points',
                        labels={'driver_name': 'Driver', 'points': 'Points'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with chart_col2:
                    fig = px.scatter(
                        championship,
                        x='wins',
                        y='points',
                        hover_data=['driver_name'],
                        title='Wins vs Championship Points',
                        size='races'
                    )
                    st.plotly_chart(fig, use_container_width=True)


elif page == "Analytics":
    st.title("📈 Detailed Analytics")
    
    # Get race features data
    race_data = engineer.create_training_data()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Grid Position Impact")
        grid_impact = race_data.groupby('grid')['points_scored'].mean()
        fig = px.line(
            x=grid_impact.index,
            y=grid_impact.values,
            labels={'x': 'Grid Position', 'y': 'Avg Points'},
            title='Average Points by Grid Position'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Finish Rate by Grid Position")
        finish_rate = race_data.groupby('grid')['finished'].mean()
        fig = px.bar(
            x=finish_rate.index,
            y=finish_rate.values,
            labels={'x': 'Grid Position', 'y': 'Finish Rate'},
            title='Race Completion Rate by Grid Position'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Constructor analysis
    st.subheader("Constructor Performance Over Time")
    constructor_years = race_data.groupby(['year', 'name'])['points_scored'].mean().reset_index()
    
    fig = px.line(
        constructor_years[constructor_years['name'].isin(constructor_years['name'].value_counts().head(5).index)],
        x='year',
        y='points_scored',
        color='name',
        title='Top 5 Constructors Average Points Over Time',
        labels={'points_scored': 'Avg Points', 'name': 'Constructor'}
    )
    st.plotly_chart(fig, use_container_width=True)


elif page == "Model Performance":
    st.title("🤖 Model Performance Metrics")
    
    st.subheader("Model Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Points Model Type", "Gradient Boosting")
    with col2:
        st.metric("Position Model Type", "Random Forest")
    with col3:
        st.metric("Finish Model Type", "Random Forest Classifier")
    
    st.divider()
    
    # Get predictions on test data
    X_points, y_points, feature_cols = engineer.get_feature_matrix('points_scored')
    
    if len(X_points) > 0:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_points, y_points, test_size=0.2, random_state=42
        )
        
        y_pred = predictor.points_model.predict(X_test)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            mse = mean_squared_error(y_test, y_pred)
            st.metric("Mean Squared Error", f"{mse:.4f}")
        with col2:
            r2 = r2_score(y_test, y_pred)
            st.metric("R² Score", f"{r2:.4f}")
        with col3:
            rmse = np.sqrt(mse)
            st.metric("RMSE", f"{rmse:.4f}")
        
        st.divider()
        
        # Feature importance
        st.subheader("Feature Importance (Points Model)")
        
        if hasattr(predictor.points_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': predictor.points_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig = px.bar(
                importance_df,
                x='importance',
                y='feature',
                orientation='h',
                title='Feature Importance for Points Prediction'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Predictions vs Actual
        st.subheader("Predictions vs Actual")
        
        comparison_df = pd.DataFrame({
            'Actual': y_test.values,
            'Predicted': y_pred,
            'Error': y_test.values - y_pred
        }).head(100)
        
        fig = px.scatter(
            comparison_df,
            x='Actual',
            y='Predicted',
            title='Predicted vs Actual Points',
            labels={'Actual': 'Actual Points', 'Predicted': 'Predicted Points'}
        )
        # Add diagonal line
        fig.add_shape(
            type="line",
            x0=0, y0=0,
            x1=comparison_df['Actual'].max(),
            y1=comparison_df['Actual'].max(),
            line=dict(dash="dash")
        )
        st.plotly_chart(fig, use_container_width=True)


st.divider()
st.markdown("---")
st.markdown("🏁 **F1 Race Prediction System** | Powered by Python, Streamlit & Scikit-learn")
