"""
Streamlit Dashboard for F1 Race Prediction System
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import os

# Set custom F1 dark theme for Plotly
pio.templates.default = "plotly_dark"
custom_template = pio.templates["plotly_dark"]
custom_template.layout.paper_bgcolor = "rgba(0,0,0,0)"
custom_template.layout.plot_bgcolor = "rgba(0,0,0,0)"
custom_template.layout.font.family = "Inter, sans-serif"
custom_template.layout.title.font.family = "Teko, sans-serif"
custom_template.layout.title.font.size = 24
pio.templates.default = custom_template

from data_loader import F1DataLoader
from feature_engineer import FeatureEngineer
from model_trainer import F1Predictor
from gp_simulator import GPSimulator
from config import WeatherSimulator


# Page configuration must be the first Streamlit command
st.set_page_config(page_title="F1 Weather Lab", layout="wide", initial_sidebar_state="collapsed")

# Load Custom CSS
try:
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# Custom Header Display
st.markdown("""
<div class="f1-header-container">
    <div class="f1-title-box">
        <h1 class="f1-main-title">F1 WEATHER LAB</h1>
        <div class="f1-sub-title">RACE PROJECTION SYSTEM</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Top navigation
pages = ["COURSE", "QUALIFS", "SAISON", "ÉCURIES", "ANALYTICS"]
page = st.radio("Navigation", pages, horizontal=True, label_visibility="collapsed")

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


def assign_deterministic_grid_positions(loader, drivers_info, race_index=0):
    """Assign deterministic grid positions from historical average grid per driver."""
    if not drivers_info:
        return []

    indexed_drivers = list(enumerate(drivers_info))

    historical_grids = loader.results[['driverId', 'grid']].copy()
    historical_grids['driverId'] = pd.to_numeric(historical_grids['driverId'], errors='coerce')
    historical_grids['grid'] = pd.to_numeric(historical_grids['grid'], errors='coerce')
    historical_grids = historical_grids.dropna(subset=['driverId', 'grid'])
    historical_grids = historical_grids[historical_grids['grid'] > 0]

    if historical_grids.empty:
        ranked_drivers = sorted(
            indexed_drivers,
            key=lambda item: (
                str(item[1].get('driver_name', '')),
                int(item[1].get('driver_id', 0)),
                int(item[0])
            )
        )
    else:
        avg_grid_by_driver = historical_grids.groupby('driverId')['grid'].mean().to_dict()
        ranked_drivers = sorted(
            indexed_drivers,
            key=lambda item: (
                float(avg_grid_by_driver.get(int(item[1]['driver_id']), 1_000.0)),
                str(item[1].get('driver_name', '')),
                int(item[1]['driver_id']),
                int(item[0])
            )
        )

    # Rotate deterministically across races to avoid a frozen grid during season simulations.
    shift = race_index % len(ranked_drivers)
    rotated_drivers = ranked_drivers[shift:] + ranked_drivers[:shift]

    grid_by_index = {
        int(original_index): idx + 1
        for idx, (original_index, _) in enumerate(rotated_drivers)
    }

    drivers_with_grid = []
    for original_index, driver in indexed_drivers:
        driver_with_grid = driver.copy()
        driver_with_grid['grid'] = int(grid_by_index[original_index])
        drivers_with_grid.append(driver_with_grid)

    return drivers_with_grid


# Load models
loader, engineer, predictor = load_and_train_models()
st.session_state.loader = loader
st.session_state.engineer = engineer
st.session_state.predictor = predictor
dataset_limits = loader.get_dataset_limits()
max_race_drivers_limit = max(2, int(dataset_limits.get('max_drivers_per_race', 20)))
max_season_races_limit = max(2, int(dataset_limits.get('max_races_per_season', 23)))


# Page content
if page == "QUALIFS":
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


elif page == "COURSE":
    st.title("🎯 Single Race Simulator")

    driver_options_df, constructor_options_df, latest_team_by_driver = get_driver_team_options(loader)
    if driver_options_df.empty or constructor_options_df.empty:
        st.error("Driver or constructor data is missing from CSV files.")
        st.stop()

    max_available_drivers = max(2, min(len(driver_options_df), max_race_drivers_limit))
    
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
        default_num_drivers = min(10, max_available_drivers)
        num_drivers = st.slider("Number of drivers", 2, max_available_drivers, default_num_drivers)
    
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
                max_value=num_drivers,
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
            simulator = GPSimulator(predictor, loader, engineer, deterministic=True)
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


elif page == "SAISON":
    st.title("🏆 Season Simulator")
    
    st.info("This feature allows you to simulate an entire F1 season with multiple races.")

    driver_options_df, constructor_options_df, latest_team_by_driver = get_driver_team_options(loader)
    if driver_options_df.empty or constructor_options_df.empty:
        st.error("Driver or constructor data is missing from CSV files.")
        st.stop()

    driver_labels = driver_options_df['driver_name'].tolist()
    max_season_drivers = max(2, min(len(driver_labels), max_race_drivers_limit))
    min_season_drivers = 5 if max_season_drivers >= 5 else 2

    col1, col2, col3 = st.columns(3)

    with col1:
        default_num_races = min(10, max_season_races_limit)
        num_races = st.slider("Number of Races", 2, max_season_races_limit, default_num_races)
    with col2:
        num_simulations = st.slider("Simulations", 1, 10, 1)
    with col3:
        default_num_season_drivers = min(10, max_season_drivers)
        num_season_drivers = st.slider(
            "Number of drivers in season",
            min_season_drivers,
            max_season_drivers,
            default_num_season_drivers
        )

    weather_mode = st.selectbox(
        "Weather Mode",
        ["Mixed (random)", "Dry (stable)", "Wet (stable)"],
        index=0
    )

    st.subheader("Season Driver Roster")
    st.caption("Select real drivers and constructors from the historical dataset.")

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
                    drivers_for_race = assign_deterministic_grid_positions(
                        loader,
                        season_roster,
                        race_index=race_num
                    )

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

                simulator = GPSimulator(predictor, loader, engineer, deterministic=True)
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


elif page == "ANALYTICS":
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


elif page == "ÉCURIES":
    st.markdown("<h2 style='text-align: left; margin-bottom: 2rem; color: #94a3b8; font-size: 1.2rem; letter-spacing: 2px;'>CHOISIR LES DEUX ÉCURIES</h2>", unsafe_allow_html=True)

    constructor_df = loader.constructors[['constructorId', 'name']].drop_duplicates(subset=['constructorId']).copy()
    enriched_results = loader.results.merge(constructor_df, on='constructorId', how='left')
    enriched_results = enriched_results.merge(
        loader.drivers[['driverId', 'forename', 'surname']],
        on='driverId',
        how='left'
    )
    enriched_results = enriched_results.merge(loader.races[['raceId', 'date']], on='raceId', how='left')

    enriched_results['driver_name'] = (
        enriched_results['forename'].fillna('') + ' ' + enriched_results['surname'].fillna('')
    ).str.strip()
    enriched_results['date'] = pd.to_datetime(enriched_results['date'], errors='coerce')
    enriched_results['points'] = pd.to_numeric(enriched_results['points'], errors='coerce').fillna(0)
    enriched_results['grid'] = pd.to_numeric(enriched_results['grid'], errors='coerce')
    enriched_results['position_num'] = pd.to_numeric(enriched_results['position'], errors='coerce')
    enriched_results['positionOrder'] = pd.to_numeric(enriched_results['positionOrder'], errors='coerce')
    enriched_results['finished'] = enriched_results['position_num'].notna().astype(int)
    enriched_results['podium'] = (enriched_results['positionOrder'] <= 3).fillna(False).astype(int)

    constructor_stats = enriched_results.groupby(['constructorId', 'name']).agg(
        avg_points=('points', 'mean'),
        finish_rate=('finished', 'mean'),
        podium_rate=('podium', 'mean'),
        avg_grid=('grid', lambda s: s[s > 0].mean()),
        points_std=('points', 'std')
    ).reset_index()
    constructor_stats = constructor_stats.dropna(subset=['name']).fillna(0)

    if constructor_stats.empty:
        st.error("No constructor data found in CSV files.")
    else:
        def normalize_score(series, value, invert=False):
            values = pd.to_numeric(series, errors='coerce').dropna()
            if values.empty:
                return 0.0
            min_val = float(values.min())
            max_val = float(values.max())
            if max_val == min_val:
                return 50.0
            score = ((float(value) - min_val) / (max_val - min_val)) * 100.0
            if invert:
                score = 100.0 - score
            return float(np.clip(score, 0, 100))

        constructor_stats['points_strength'] = constructor_stats['avg_points'].apply(
            lambda v: normalize_score(constructor_stats['avg_points'], v)
        )
        constructor_stats['reliability_strength'] = (constructor_stats['finish_rate'] * 100.0).clip(0, 100)
        constructor_stats['podium_strength'] = (constructor_stats['podium_rate'] * 100.0).clip(0, 100)
        constructor_stats['quali_strength'] = constructor_stats['avg_grid'].apply(
            lambda v: normalize_score(constructor_stats['avg_grid'], v, invert=True)
        )
        constructor_stats['consistency_strength'] = constructor_stats['points_std'].apply(
            lambda v: normalize_score(constructor_stats['points_std'], v, invert=True)
        )

        constructor_names = constructor_stats['name'].sort_values().tolist()

        colA, colB = st.columns(2)
        with colA:
            st.markdown("<div class='f1-subtitle' style='color:#64748b; margin-bottom:10px;'>ÉCURIE A</div>", unsafe_allow_html=True)
            team_a = st.radio("Team A", constructor_names, key="team_a_select", label_visibility="collapsed")
        with colB:
            st.markdown("<div class='f1-subtitle' style='color:#64748b; margin-bottom:10px;'>ÉCURIE B</div>", unsafe_allow_html=True)
            default_index = 1 if len(constructor_names) > 1 else 0
            team_b = st.radio("Team B", constructor_names, index=default_index, key="team_b_select", label_visibility="collapsed")

        if team_a == team_b:
            st.warning("Please select two different teams for comparison.")

        team_stats_map = {
            row['name']: row
            for _, row in constructor_stats.iterrows()
        }

        def get_team_drivers(constructor_id, top_n=2):
            team_rows = enriched_results[enriched_results['constructorId'] == constructor_id].copy()
            team_rows = team_rows.sort_values('date', ascending=False)

            latest_unique = team_rows.drop_duplicates(subset=['driverId'], keep='first')
            latest_unique = latest_unique[latest_unique['driver_name'].str.len() > 0]

            selected = []
            selected_ids = set()
            for _, row in latest_unique.iterrows():
                if row['driverId'] in selected_ids:
                    continue
                selected.append({'driver_id': int(row['driverId']), 'driver_name': row['driver_name']})
                selected_ids.add(row['driverId'])
                if len(selected) >= top_n:
                    break

            if len(selected) < top_n:
                fallback_rank = team_rows.groupby(['driverId', 'driver_name'], as_index=False)['points'].sum()
                fallback_rank = fallback_rank.sort_values('points', ascending=False)
                for _, row in fallback_rank.iterrows():
                    if row['driverId'] in selected_ids or not str(row['driver_name']).strip():
                        continue
                    selected.append({'driver_id': int(row['driverId']), 'driver_name': row['driver_name']})
                    selected_ids.add(row['driverId'])
                    if len(selected) >= top_n:
                        break

            return selected

        team_a_id = int(team_stats_map[team_a]['constructorId'])
        team_b_id = int(team_stats_map[team_b]['constructorId'])
        team_a_drivers = get_team_drivers(team_a_id, top_n=2)
        team_b_drivers = get_team_drivers(team_b_id, top_n=2)
        team_a_driver_text = " - ".join([d['driver_name'] for d in team_a_drivers]) or "No driver data"
        team_b_driver_text = " - ".join([d['driver_name'] for d in team_b_drivers]) or "No driver data"

        st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)

        def get_team_color(name):
            palette = ['#e10600', '#00d2be', '#ff8700', '#1e5bc6', '#2d826d', '#ed1c24', '#005aff', '#b6babd']
            return palette[abs(hash(name)) % len(palette)]

        color_a = get_team_color(team_a)
        color_b = get_team_color(team_b)

        comp_col1, comp_col2, comp_col3 = st.columns([1, 0.1, 1])
        with comp_col1:
            st.markdown(f"""
            <div class='f1-card' style='text-align: center; border-color: {color_a}; background: rgba(255,255,255,0.02);'>
                <h2 style='color: {color_a} !important; margin:0;'>{team_a}</h2>
                <div style='color: #64748b; font-size: 0.9rem;'>{team_a_driver_text}</div>
            </div>
            """, unsafe_allow_html=True)
        with comp_col2:
            st.markdown("<div style='text-align: center; margin-top: 30px; font-weight: bold; color: #64748b;'>VS</div>", unsafe_allow_html=True)
        with comp_col3:
            st.markdown(f"""
            <div class='f1-card' style='text-align: center; border-color: {color_b}; background: rgba(255,255,255,0.02);'>
                <h2 style='color: {color_b} !important; margin:0;'>{team_b}</h2>
                <div style='color: #64748b; font-size: 0.9rem;'>{team_b_driver_text}</div>
            </div>
            """, unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["📊 COMPÉTENCES", "🏁 SIMULATION"])

        with tab1:
            st.markdown("<h3 style='margin-top: 20px; font-size: 1.2rem; color: #94a3b8;'>COMPARAISON DES ATTRIBUTS (CSV)</h3>", unsafe_allow_html=True)

            team_a_row = team_stats_map[team_a]
            team_b_row = team_stats_map[team_b]

            attributes = [
                ("Performance points", int(round(team_a_row['points_strength'])), int(round(team_b_row['points_strength']))),
                ("Fiabilité", int(round(team_a_row['reliability_strength'])), int(round(team_b_row['reliability_strength']))),
                ("Podiums", int(round(team_a_row['podium_strength'])), int(round(team_b_row['podium_strength']))),
                ("Qualifs (grille inverse)", int(round(team_a_row['quali_strength'])), int(round(team_b_row['quali_strength']))),
                ("Régularité", int(round(team_a_row['consistency_strength'])), int(round(team_b_row['consistency_strength'])))
            ]

            for attr, valA, valB in attributes:
                st.markdown(f"""
                <div style='background: #131a31; padding: 15px 20px; border-radius: 8px; margin-bottom: 15px; border: 1px solid rgba(255,255,255,0.05);'>
                    <div style='display:flex; justify-content: space-between; margin-bottom: 8px;'>
                        <span style='color: {color_a}; font-family: Teko, sans-serif; font-size: 1.4rem;'>{valA}</span>
                        <span style='color: #94a3b8; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;'>{attr}</span>
                        <span style='color: {color_b}; font-family: Teko, sans-serif; font-size: 1.4rem;'>{valB}</span>
                    </div>
                    <div style='display:flex; gap: 8px;'>
                        <div style='flex: 1; height: 6px; background: rgba(255,255,255,0.05); border-radius: 3px; direction: rtl;'>
                            <div style='width: {valA}%; height: 100%; background: {color_a}; border-radius: 3px;'></div>
                        </div>
                        <div style='flex: 1; height: 6px; background: rgba(255,255,255,0.05); border-radius: 3px;'>
                            <div style='width: {valB}%; height: 100%; background: {color_b}; border-radius: 3px;'></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with tab2:
            st.markdown("<br>", unsafe_allow_html=True)
            mode = st.radio("Mode", ["COURSE UNIQUE", "SAISON COMPLÈTE"], horizontal=True, label_visibility="collapsed")

            st.markdown("<div class='f1-subtitle' style='margin-top:30px; margin-bottom: 10px;'>MÉTÉO</div>", unsafe_allow_html=True)
            weather = st.radio("Weather Condition", ["DRY", "RAIN", "WIND", "MIXED"], horizontal=True, label_visibility="collapsed")

            if mode == "COURSE UNIQUE":
                circuit_df = loader.circuits[['circuitId', 'name']].dropna(subset=['circuitId', 'name']).copy()
                circuit_df = circuit_df.sort_values('name')
                selected_circuit_name = st.selectbox("Circuit", circuit_df['name'].tolist(), key='ecuries_circuit')
                selected_circuit_id = int(circuit_df[circuit_df['name'] == selected_circuit_name]['circuitId'].iloc[0])
            else:
                ecuries_races_default = min(10, max_season_races_limit)
                season_races = st.slider(
                    "Nombre de courses",
                    2,
                    max_season_races_limit,
                    ecuries_races_default,
                    key='ecuries_season_races'
                )
                season_sims = st.slider("Nombre de simulations", 1, 10, 1, key='ecuries_season_sims')

            def weather_factor_from_choice(choice):
                if choice == "DRY":
                    return WeatherSimulator.get_weather_factor('sunny')
                if choice == "RAIN":
                    return WeatherSimulator.get_weather_factor('light_rain')
                if choice == "WIND":
                    return WeatherSimulator.get_weather_factor('cloudy')
                _, factor = WeatherSimulator.get_random_weather()
                return float(factor)

            st.markdown("<br><br>", unsafe_allow_html=True)
            if st.button("LANCER LA COMPARAISON"):
                if team_a == team_b:
                    st.error("Please select two different teams.")
                else:
                    simulator = GPSimulator(predictor, loader, engineer, deterministic=True)

                    selected_drivers = []
                    for team_name, team_id, team_drivers in [
                        (team_a, team_a_id, team_a_drivers),
                        (team_b, team_b_id, team_b_drivers)
                    ]:
                        for driver in team_drivers:
                            selected_drivers.append({
                                'driver_id': driver['driver_id'],
                                'driver_name': driver['driver_name'],
                                'constructor_id': team_id,
                                'constructor': team_name
                            })

                    if len(selected_drivers) < 2:
                        st.error("Not enough driver data for a valid comparison.")
                    else:
                        if mode == "COURSE UNIQUE":
                            race_drivers = assign_deterministic_grid_positions(
                                loader,
                                selected_drivers,
                                race_index=0
                            )

                            weather_factor = weather_factor_from_choice(weather)
                            race_results = simulator.simulate_race(
                                circuit_id=selected_circuit_id,
                                drivers_info=race_drivers,
                                weather_factor=weather_factor,
                                safety_car=False
                            )

                            team_points = race_results.groupby('constructor')['actual_points'].sum()
                            team_points = team_points.reindex([team_a, team_b]).fillna(0)

                            if team_points.iloc[0] == team_points.iloc[1]:
                                headline = "ÉGALITÉ"
                            else:
                                headline = f"{team_points.idxmax()} remporte le duel"

                            st.markdown(f"""
                            <div style='text-align: center; margin-top: 40px; padding: 30px; background: #131a31; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);'>
                                <div style='color: #64748b; letter-spacing: 2px; font-size: 0.9rem; margin-bottom: 10px;'>RÉSULTAT COURSE 🏁</div>
                                <h2 style='font-size: 2rem !important;'>{headline}</h2>
                                <div style='display: flex; justify-content: center; gap: 60px; margin-top: 30px;'>
                                    <div style='text-align: center;'>
                                        <div style='font-size: 4rem; color: {color_a}; font-family: Teko, sans-serif; line-height: 1;'>{int(team_points.iloc[0])}</div>
                                        <div style='color: #94a3b8; font-size: 0.9rem; text-transform: uppercase;'>{team_a}</div>
                                    </div>
                                    <div style='text-align: center;'>
                                        <div style='font-size: 4rem; color: {color_b}; font-family: Teko, sans-serif; line-height: 1;'>{int(team_points.iloc[1])}</div>
                                        <div style='color: #94a3b8; font-size: 0.9rem; text-transform: uppercase;'>{team_b}</div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            st.dataframe(
                                race_results[['driver_name', 'constructor', 'grid_position', 'predicted_position', 'actual_points']],
                                use_container_width=True
                            )
                        else:
                            circuit_ids = loader.circuits['circuitId'].dropna().astype(int).tolist()
                            if not circuit_ids:
                                st.error("No circuit data found in CSV files.")
                            else:
                                races_data = []
                                for race_idx in range(season_races):
                                    race_drivers = assign_deterministic_grid_positions(
                                        loader,
                                        selected_drivers,
                                        race_index=race_idx
                                    )

                                    if weather == "MIXED":
                                        race_weather_factor = weather_factor_from_choice("MIXED")
                                    else:
                                        race_weather_factor = weather_factor_from_choice(weather)

                                    races_data.append({
                                        'circuit_id': int(circuit_ids[race_idx % len(circuit_ids)]),
                                        'drivers_info': race_drivers,
                                        'weather_factor': float(race_weather_factor)
                                    })

                                championship = simulator.simulate_season(races_data, season_sims)
                                team_points = championship.groupby('constructor')['points'].sum()
                                team_points = team_points.reindex([team_a, team_b]).fillna(0)

                                if team_points.iloc[0] == team_points.iloc[1]:
                                    headline = "ÉGALITÉ SUR LA SAISON"
                                else:
                                    headline = f"{team_points.idxmax()} remporte la saison"

                                st.markdown(f"""
                                <div style='text-align: center; margin-top: 40px; padding: 30px; background: #131a31; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);'>
                                    <div style='color: #64748b; letter-spacing: 2px; font-size: 0.9rem; margin-bottom: 10px;'>RÉSULTAT SAISON 🏆</div>
                                    <h2 style='font-size: 2rem !important;'>{headline}</h2>
                                    <div style='display: flex; justify-content: center; gap: 60px; margin-top: 30px;'>
                                        <div style='text-align: center;'>
                                            <div style='font-size: 4rem; color: {color_a}; font-family: Teko, sans-serif; line-height: 1;'>{round(team_points.iloc[0], 1)}</div>
                                            <div style='color: #94a3b8; font-size: 0.9rem; text-transform: uppercase;'>{team_a}</div>
                                        </div>
                                        <div style='text-align: center;'>
                                            <div style='font-size: 4rem; color: {color_b}; font-family: Teko, sans-serif; line-height: 1;'>{round(team_points.iloc[1], 1)}</div>
                                            <div style='color: #94a3b8; font-size: 0.9rem; text-transform: uppercase;'>{team_b}</div>
                                        </div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                                filtered_championship = championship[
                                    championship['constructor'].isin([team_a, team_b])
                                ].sort_values('points', ascending=False)
                                st.dataframe(
                                    filtered_championship[['position', 'driver_name', 'constructor', 'points', 'wins']],
                                    use_container_width=True
                                )

st.divider()
st.markdown("---")
st.markdown("🏁 **F1 Race Prediction System** | Modified UI styling matching F1 Weather Lab Aesthetics")
