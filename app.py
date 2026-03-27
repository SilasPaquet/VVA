"""
Streamlit Dashboard for F1 Race Prediction System
Refactored for modularity and maintainability.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import os

from data_loader import F1DataLoader
from feature_engineer import FeatureEngineer
from model_trainer import F1Predictor
from gp_simulator import GPSimulator
from config import WeatherSimulator

pio.templates.default = "plotly_dark"
custom_template = pio.templates["plotly_dark"]
custom_template.layout.paper_bgcolor = "rgba(0,0,0,0)"
custom_template.layout.plot_bgcolor = "rgba(0,0,0,0)"
custom_template.layout.font.family = "Inter, sans-serif"
custom_template.layout.title.font.family = "Teko, sans-serif"
custom_template.layout.title.font.size = 24
pio.templates.default = custom_template


# =====================================================================
# DATA & MODEL HELPERS
# =====================================================================

@st.cache_resource
def load_and_train_models(force_rebuild_data=False, use_clean_cache=True):
    """Load data and train models"""
    with st.spinner("Loading and training models..."):
        loader = F1DataLoader()
        loader.load_all_data(
            use_clean_cache=use_clean_cache,
            force_rebuild=force_rebuild_data
        )
        
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
    constructors_df = constructors_df.drop_duplicates(subset=['constructorId']).sort_values('name').reset_index(drop=True)

    latest_results = loader.results.merge(loader.races[['raceId', 'date']], on='raceId', how='left')
    latest_results['date'] = pd.to_datetime(latest_results['date'], errors='coerce')
    latest_results = latest_results.sort_values('date').dropna(subset=['driverId', 'constructorId'])
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
            key=lambda item: (str(item[1].get('driver_name', '')), int(item[1].get('driver_id', 0)), int(item[0]))
        )
    else:
        avg_grid = historical_grids.groupby('driverId')['grid'].mean().to_dict()
        ranked_drivers = sorted(
            indexed_drivers,
            key=lambda item: (float(avg_grid.get(int(item[1]['driver_id']), 1000.0)), str(item[1].get('driver_name', '')), int(item[1]['driver_id']), int(item[0]))
        )

    shift = race_index % len(ranked_drivers)
    rotated_drivers = ranked_drivers[shift:] + ranked_drivers[:shift]
    grid_by_index = {int(orig_idx): idx + 1 for idx, (orig_idx, _) in enumerate(rotated_drivers)}

    drivers_with_grid = []
    for orig_idx, driver in indexed_drivers:
        driver_with_grid = driver.copy()
        driver_with_grid['grid'] = int(grid_by_index[orig_idx])
        drivers_with_grid.append(driver_with_grid)

    return drivers_with_grid


# =====================================================================
# ÉCURIES PAGE HELPERS (Extracted)
# =====================================================================

def normalize_score(series, value, invert=False):
    values = pd.to_numeric(series, errors='coerce').dropna()
    if values.empty:
        return 0.0
    min_val, max_val = float(values.min()), float(values.max())
    if max_val == min_val:
        return 50.0
    score = ((float(value) - min_val) / (max_val - min_val)) * 100.0
    if invert:
        score = 100.0 - score
    return float(np.clip(score, 0, 100))


def get_team_drivers(enriched_results, constructor_id, top_n=2):
    team_rows = enriched_results[enriched_results['constructorId'] == constructor_id].copy()
    team_rows = team_rows.sort_values('date', ascending=False)
    latest_unique = team_rows.drop_duplicates(subset=['driverId'], keep='first')
    latest_unique = latest_unique[latest_unique['driver_name'].str.len() > 0]

    selected = []
    selected_ids = set()
    for _, row in latest_unique.iterrows():
        if row['driverId'] not in selected_ids:
            selected.append({'driver_id': int(row['driverId']), 'driver_name': row['driver_name']})
            selected_ids.add(row['driverId'])
            if len(selected) >= top_n: break

    if len(selected) < top_n:
        fallback_rank = team_rows.groupby(['driverId', 'driver_name'], as_index=False)['points'].sum()
        fallback_rank = fallback_rank.sort_values('points', ascending=False)
        for _, row in fallback_rank.iterrows():
            if row['driverId'] not in selected_ids and str(row['driver_name']).strip():
                selected.append({'driver_id': int(row['driverId']), 'driver_name': row['driver_name']})
                selected_ids.add(row['driverId'])
                if len(selected) >= top_n: break

    return selected


def get_team_color(name):
    palette = ['#e10600', '#00d2be', '#ff8700', '#1e5bc6', '#2d826d', '#ed1c24', '#005aff', '#b6babd']
    return palette[abs(hash(name)) % len(palette)]


def weather_factor_from_choice(choice):
    if choice == "DRY": return WeatherSimulator.get_weather_factor('sunny')
    if choice == "RAIN": return WeatherSimulator.get_weather_factor('light_rain')
    if choice == "WIND": return WeatherSimulator.get_weather_factor('cloudy')
    _, factor = WeatherSimulator.get_random_weather()
    return float(factor)


def prepare_ecuries_stats(loader):
    constructor_df = loader.constructors[['constructorId', 'name']].drop_duplicates(subset=['constructorId']).copy()
    enriched = loader.results.merge(constructor_df, on='constructorId', how='left')
    enriched = enriched.merge(loader.drivers[['driverId', 'forename', 'surname']], on='driverId', how='left')
    enriched = enriched.merge(loader.races[['raceId', 'date']], on='raceId', how='left')

    enriched['driver_name'] = (enriched['forename'].fillna('') + ' ' + enriched['surname'].fillna('')).str.strip()
    enriched['date'] = pd.to_datetime(enriched['date'], errors='coerce')
    enriched['points'] = pd.to_numeric(enriched['points'], errors='coerce').fillna(0)
    enriched['grid'] = pd.to_numeric(enriched['grid'], errors='coerce')
    enriched['position_num'] = pd.to_numeric(enriched['position'], errors='coerce')
    enriched['positionOrder'] = pd.to_numeric(enriched['positionOrder'], errors='coerce')
    enriched['finished'] = enriched['position_num'].notna().astype(int)
    enriched['podium'] = (enriched['positionOrder'] <= 3).fillna(False).astype(int)

    constructor_stats = enriched.groupby(['constructorId', 'name']).agg(
        avg_points=('points', 'mean'), finish_rate=('finished', 'mean'), podium_rate=('podium', 'mean'),
        avg_grid=('grid', lambda s: s[s > 0].mean()), points_std=('points', 'std')
    ).reset_index().dropna(subset=['name']).fillna(0)
    
    return constructor_stats, enriched


def calculate_ecuries_strength(constructor_stats):
    constructor_stats['points_strength'] = constructor_stats['avg_points'].apply(
        lambda v: normalize_score(constructor_stats['avg_points'], v))
    constructor_stats['reliability_strength'] = (constructor_stats['finish_rate'] * 100.0).clip(0, 100)
    constructor_stats['podium_strength'] = (constructor_stats['podium_rate'] * 100.0).clip(0, 100)
    constructor_stats['quali_strength'] = constructor_stats['avg_grid'].apply(
        lambda v: normalize_score(constructor_stats['avg_grid'], v, invert=True))
    constructor_stats['consistency_strength'] = constructor_stats['points_std'].apply(
        lambda v: normalize_score(constructor_stats['points_std'], v, invert=True))
    return constructor_stats


# =====================================================================
# PAGE RENDERERS
# =====================================================================

def render_qualifs_page(loader):
    st.title("F1 Race Predictions Dashboard")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Races", len(loader.races))
    col2.metric("Total Drivers", len(loader.drivers))
    col3.metric("Total Constructors", len(loader.constructors))
    st.divider()
    
    st.subheader("Historical Statistics")
    colA, colB = st.columns(2)
    with colA:
        races_per_year = loader.races.groupby('year').size()
        fig1 = px.line(x=races_per_year.index, y=races_per_year.values, labels={'x': 'Year', 'y': 'Number of Races'}, title='Races per Season')
        st.plotly_chart(fig1, use_container_width=True)
    with colB:
        drivers_per_year = loader.results.merge(loader.races[['raceId', 'year']], on='raceId').groupby('year')['driverId'].nunique()
        fig2 = px.line(x=drivers_per_year.index, y=drivers_per_year.values, labels={'x': 'Year', 'y': 'Number of Drivers'}, title='Active Drivers per Season')
        st.plotly_chart(fig2, use_container_width=True)
    
    render_qualifs_top_drivers(loader)


def render_qualifs_top_drivers(loader):
    st.subheader("Top Drivers (Historical)")
    top_drivers = loader.results.groupby('driverId').agg({'points': 'sum'}).nlargest(10, 'points').reset_index()
    top_drivers = top_drivers.merge(loader.drivers[['driverId', 'forename', 'surname']], on='driverId')
    top_drivers['name'] = top_drivers['forename'] + ' ' + top_drivers['surname']
    fig = px.bar(top_drivers, x='points', y='name', orientation='h', title='Top 10 Drivers by Career Points')
    st.plotly_chart(fig, use_container_width=True)


def build_course_driver_selections(num_drivers, driver_labels, driver_label_to_id, constructor_names, constructor_name_to_id, latest_team_by_driver):
    drivers_info = []
    for i in range(num_drivers):
        c1, c2, c3 = st.columns(3)
        def_driver_idx = i if i < len(driver_labels) else 0
        with c1:
            drv_lbl = st.selectbox(f"Driver {i+1}", driver_labels, index=def_driver_idx, key=f"d_sel_{i}")
        drv_id = driver_label_to_id[drv_lbl]
        with c2:
            grid = st.number_input(f"Grid Size {i+1}", min_value=1, max_value=num_drivers, value=i+1, key=f"g_sel_{i}")
        
        def_const_id = latest_team_by_driver.get(drv_id, constructor_name_to_id[constructor_names[0]])
        try:
            def_const_idx = constructor_names.index(next(k for k, v in constructor_name_to_id.items() if v == def_const_id))
        except StopIteration:
            def_const_idx = 0
            
        with c3:
            team_val = st.selectbox(f"Team {i+1}", constructor_names, index=def_const_idx, key=f"t_sel_{i}")
        
        drivers_info.append({'driver_id': drv_id, 'driver_name': drv_lbl, 'grid': int(grid), 'constructor_id': constructor_name_to_id[team_val], 'constructor': team_val})
    return drivers_info


def render_course_page(loader, predictor, engineer, limits):
    st.title("Single Race Simulator")
    df_d, df_c, latest_team = get_driver_team_options(loader)
    if df_d.empty or df_c.empty:
        st.error("Missing data from CSV files."); st.stop()

    max_drv = max(2, min(len(df_d), limits['max_drivers_per_race']))
    col1, col2 = st.columns(2)
    with col1:
        circuits = loader.circuits.sort_values('name')
        circuit_name = st.selectbox("Select Circuit", circuits['name'])
        circuit_id = circuits[circuits['name'] == circuit_name]['circuitId'].values[0]
        num_drivers = st.slider("Number of drivers", 2, max_drv, min(10, max_drv))
    with col2:
        weather_factor = st.slider("Weather Impact", 0.5, 2.0, 1.0)
        safety_car = st.checkbox("Safety Car Intervention")

    st.subheader("Drivers in Race")
    drivers_info = build_course_driver_selections(num_drivers, df_d['driver_name'].tolist(), dict(zip(df_d['driver_name'], df_d['driverId'].astype(int))), df_c['name'].tolist(), dict(zip(df_c['name'], df_c['constructorId'].astype(int))), latest_team)

    selected_ids = [e['driver_id'] for e in drivers_info]
    selected_grids = [e['grid'] for e in drivers_info]
    if len(set(selected_ids)) != len(selected_ids): st.warning("Same driver selected multiple times.")
    if len(set(selected_grids)) != len(selected_grids): st.warning("Same grid position shared.")

    if st.button("Simulate Race"):
        if len(set(selected_ids)) != len(selected_ids) or len(set(selected_grids)) != len(selected_grids):
            st.error("Please fix duplicate drivers or grids."); return
        execute_course_sim(GPSimulator(predictor, loader, engineer, deterministic=True), circuit_id, drivers_info, weather_factor, safety_car)


def execute_course_sim(simulator, circuit_id, drivers_info, weather_factor, safety_car):
    results = simulator.simulate_race(circuit_id=circuit_id, drivers_info=drivers_info, weather_factor=weather_factor, safety_car=safety_car)
    st.success("Race Simulation Complete!")
    st.dataframe(results[['driver_name', 'grid_position', 'predicted_position', 'predicted_points', 'actual_points', 'finish_probability']], use_container_width=True)
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(px.bar(results[results['finished']], x='driver_name', y='predicted_points', title='Predicted Points'), use_container_width=True)
    with c2: st.plotly_chart(px.scatter(results, x='grid_position', y='predicted_position', hover_data=['driver_name'], title='Grid vs Predicted Position'), use_container_width=True)


def build_saison_roster(num_drvs, d_lbls, d_map, c_lbls, c_map, latest_team):
    roster = []
    for i in range(num_drvs):
        c1, c2 = st.columns(2)
        idx = i if i < len(d_lbls) else 0
        with c1: d_val = st.selectbox(f"Driver {i+1}", d_lbls, index=idx, key=f"s_d_{i}")
        d_id = d_map[d_val]
        def_cid = latest_team.get(d_id, c_map[c_lbls[0]])
        try:
            def_cidx = c_lbls.index(next(k for k, v in c_map.items() if v == def_cid))
        except StopIteration:
            def_cidx = 0
        with c2: c_val = st.selectbox(f"Team {i+1}", c_lbls, index=def_cidx, key=f"s_t_{i}")
        roster.append({'driver_id': d_id, 'driver_name': d_val, 'constructor_id': c_map[c_val], 'constructor': c_val})
    return roster


def execute_season_sim(simulator, loader, roster, num_races, weather_mode, num_sims):
    circuit_ids = loader.circuits['circuitId'].dropna().astype(int).tolist() or [1]
    races_data = []
    for r_idx in range(num_races):
        drivers_for_race = assign_deterministic_grid_positions(loader, roster, race_index=r_idx)
        w_factor = 1.0 if weather_mode != "Wet (stable)" else 0.8
        races_data.append({'circuit_id': circuit_ids[r_idx % len(circuit_ids)], 'drivers_info': drivers_for_race, 'weather_factor': w_factor})
    
    championship = simulator.simulate_season(races_data, num_sims)
    st.success("Season Simulation Complete!")
    st.dataframe(championship[['position', 'driver_name', 'constructor', 'points', 'wins']], use_container_width=True)
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(px.bar(championship.head(10), x='driver_name', y='points', title='Top 10 Championship Points'), use_container_width=True)
    with c2: st.plotly_chart(px.scatter(championship, x='wins', y='points', hover_data=['driver_name'], title='Wins vs Championship Points', size='races'), use_container_width=True)


def render_saison_page(loader, predictor, engineer, limits):
    st.title("Season Simulator")
    df_d, df_c, latest_team = get_driver_team_options(loader)
    if df_d.empty or df_c.empty: st.error("Missing data."); st.stop()

    max_season_drivers = max(2, min(len(df_d), limits['max_drivers_per_race']))
    col1, col2, col3 = st.columns(3)
    with col1: num_races = st.slider("Number of Races", 2, limits['max_races_per_season'], min(10, limits['max_races_per_season']))
    with col2: num_sims = st.slider("Simulations", 1, 10, 1)
    with col3: num_season_drivers = st.slider("Drivers in season", max(5, 2), max_season_drivers, min(10, max_season_drivers))

    weather_mode = st.selectbox("Weather Mode", ["Mixed (random)", "Dry (stable)", "Wet (stable)"], index=0)
    st.subheader("Season Driver Roster")

    d_lbls, d_map = df_d['driver_name'].tolist(), dict(zip(df_d['driver_name'], df_d['driverId'].astype(int)))
    c_lbls, c_map = df_c['name'].tolist(), dict(zip(df_c['name'], df_c['constructorId'].astype(int)))
    season_roster = build_saison_roster(num_season_drivers, d_lbls, d_map, c_lbls, c_map, latest_team)

    selected_ids = [e['driver_id'] for e in season_roster]
    if len(set(selected_ids)) != len(selected_ids): st.warning("Duplicate drivers.")

    if st.button("Simulate Season"):
        if len(set(selected_ids)) != len(selected_ids): st.error("Fix duplicate drivers."); return
        with st.spinner("Simulating..."):
            execute_season_sim(GPSimulator(predictor, loader, engineer, deterministic=True), loader, season_roster, num_races, weather_mode, num_sims)


def _build_weather_exploration_lineup(loader):
    """Create a representative, recent race lineup for weather sensitivity exploration."""
    merged = loader.results.merge(
        loader.races[['raceId', 'date', 'circuitId']],
        on='raceId',
        how='left'
    )
    merged['date'] = pd.to_datetime(merged['date'], errors='coerce')
    merged['grid'] = pd.to_numeric(merged['grid'], errors='coerce')
    merged = merged.dropna(subset=['raceId', 'driverId', 'constructorId', 'circuitId', 'grid', 'date'])
    merged = merged[merged['grid'] > 0]

    if merged.empty:
        return None, [], None

    race_meta = (
        merged.groupby('raceId')
        .agg(date=('date', 'max'), drivers=('driverId', 'nunique'))
        .reset_index()
        .sort_values('date', ascending=False)
    )

    selected_race_id = None
    for _, row in race_meta.iterrows():
        if int(row['drivers']) >= 10:
            selected_race_id = int(row['raceId'])
            break

    if selected_race_id is None:
        selected_race_id = int(race_meta.iloc[0]['raceId'])

    race_rows = merged[merged['raceId'] == selected_race_id].copy()
    race_rows = race_rows.merge(
        loader.drivers[['driverId', 'forename', 'surname']],
        on='driverId',
        how='left'
    )
    race_rows = race_rows.merge(
        loader.constructors[['constructorId', 'name']],
        on='constructorId',
        how='left'
    )

    race_rows['driver_name'] = (
        race_rows['forename'].fillna('') + ' ' + race_rows['surname'].fillna('')
    ).str.strip()
    race_rows['constructor'] = race_rows['name'].fillna('Unknown')

    race_rows = race_rows.sort_values('grid').drop_duplicates(subset=['driverId'], keep='first')

    drivers_info = []
    for _, row in race_rows.iterrows():
        if not row['driver_name']:
            continue
        drivers_info.append({
            'driver_id': int(row['driverId']),
            'driver_name': row['driver_name'],
            'constructor_id': int(row['constructorId']),
            'constructor': str(row['constructor']),
            'grid': int(row['grid'])
        })

    if not drivers_info:
        return None, [], None

    circuit_id = int(race_rows['circuitId'].iloc[0])
    reference_date = race_rows['date'].max()
    return circuit_id, drivers_info, reference_date


def _compute_weather_sensitivity(simulator, circuit_id, drivers_info):
    scenarios = [
        ('Very Wet', 0.50),
        ('Wet', 0.70),
        ('Damp', 0.85),
        ('Neutral', 1.00),
        ('Warm', 1.15),
    ]

    driver_rows = []
    team_rows = []
    summary_rows = []

    for scenario_name, weather_factor in scenarios:
        race_results = simulator.simulate_race(
            circuit_id=circuit_id,
            drivers_info=drivers_info,
            weather_factor=float(weather_factor),
            safety_car=False
        )

        scenario_driver = race_results.copy()
        scenario_driver['scenario'] = scenario_name
        scenario_driver['weather_factor'] = float(weather_factor)
        scenario_driver['predicted_position'] = pd.to_numeric(
            scenario_driver['predicted_position'], errors='coerce'
        )
        driver_rows.extend(scenario_driver[[
            'scenario', 'weather_factor', 'driver_name', 'constructor',
            'grid_position', 'predicted_position', 'actual_points', 'finished'
        ]].to_dict(orient='records'))

        team_points = race_results.groupby('constructor', as_index=False)['actual_points'].sum()
        for _, row in team_points.iterrows():
            team_rows.append({
                'scenario': scenario_name,
                'weather_factor': float(weather_factor),
                'constructor': row['constructor'],
                'team_points': float(row['actual_points'])
            })

        summary_rows.append({
            'scenario': scenario_name,
            'weather_factor': float(weather_factor),
            'avg_predicted_position': float(pd.to_numeric(race_results['predicted_position'], errors='coerce').mean()),
            'finish_rate': float(race_results['finished'].mean() * 100.0),
            'avg_points': float(race_results['actual_points'].mean())
        })

    return pd.DataFrame(summary_rows), pd.DataFrame(team_rows), pd.DataFrame(driver_rows)


def render_analytics_page(loader, predictor, engineer):
    st.title("Detailed Analytics")
    race_data = engineer.data
    if race_data is None:
        race_data = engineer.create_training_data()
    
    c1, c2 = st.columns(2)
    with c1:
        grid_impact = race_data.groupby('grid')['points_scored'].mean()
        st.plotly_chart(px.line(x=grid_impact.index, y=grid_impact.values, labels={'x': 'Grid Position', 'y': 'Avg Points'}, title='Avg Points by Grid'), use_container_width=True)
    with c2:
        finish_rate = race_data.groupby('grid')['finished'].mean()
        st.plotly_chart(px.bar(x=finish_rate.index, y=finish_rate.values, labels={'x': 'Grid', 'y': 'Finish Rate'}, title='Finish Rate by Grid'), use_container_width=True)
    
    st.subheader("Constructor Performance Over Time")
    tops = race_data['name'].value_counts().head(5).index
    constructor_years = race_data[race_data['name'].isin(tops)].groupby(['year', 'name'])['points_scored'].mean().reset_index()
    st.plotly_chart(px.line(constructor_years, x='year', y='points_scored', color='name', title='Top 5 Constructors Avg Points'), use_container_width=True)

    st.divider()
    st.subheader("Weather Impact Exploratory Analysis")

    st.markdown(
        "**Sources used for weather-analysis framing:**\n"
        "- FIA Regulations portal (official sporting/technical framework): https://www.fia.com/regulation/category/110\n"
        "- FastF1 `Session.weather_data` reference: https://docs.fastf1.dev/core.html#fastf1.core.Session.weather_data\n"
        "- FastF1 weather channels (`AirTemp`, `Rainfall`, `TrackTemp`, `WindSpeed`): https://docs.fastf1.dev/api.html#fastf1.api.weather_data\n"
        "- FastF1 tire compounds include `INTERMEDIATE` and `WET`: https://docs.fastf1.dev/core.html#fastf1.core.Laps.pick_compounds\n"
        "- OpenF1 weather endpoint fields (`rainfall`, `air_temperature`, `track_temperature`, `wind_speed`): https://openf1.org/docs#weather"
    )

    circuit_id, drivers_info, reference_date = _build_weather_exploration_lineup(loader)
    if circuit_id is None or len(drivers_info) < 2:
        st.warning("Not enough data to run weather exploration graphs.")
        return

    simulator = GPSimulator(predictor, loader, engineer, deterministic=True)
    weather_summary, team_weather, driver_weather = _compute_weather_sensitivity(simulator, circuit_id, drivers_info)

    if weather_summary.empty or team_weather.empty or driver_weather.empty:
        st.warning("Weather sensitivity computation produced no data.")
        return

    if reference_date is not None and not pd.isna(reference_date):
        st.caption(f"Reference lineup date: {reference_date.date()} | Drivers in sample: {len(drivers_info)}")

    scenario_order = ['Very Wet', 'Wet', 'Damp', 'Neutral', 'Warm']

    fig_driver_dist = px.box(
        driver_weather,
        x='scenario',
        y='predicted_position',
        points='all',
        color='scenario',
        category_orders={'scenario': scenario_order},
        title='Driver Finishing Position Distribution by Weather Scenario',
        labels={'scenario': 'Scenario', 'predicted_position': 'Predicted finishing position (lower is better)'}
    )
    fig_driver_dist.update_yaxes(autorange='reversed')
    st.plotly_chart(fig_driver_dist, use_container_width=True)

    baseline_points = (
        team_weather[team_weather['scenario'] == 'Neutral'][['constructor', 'team_points']]
        .rename(columns={'team_points': 'neutral_team_points'})
    )
    team_delta = team_weather.merge(baseline_points, on='constructor', how='left')
    team_delta['delta_vs_neutral'] = team_delta['team_points'] - team_delta['neutral_team_points']

    top_teams = baseline_points.sort_values('neutral_team_points', ascending=False).head(6)['constructor']
    team_delta = team_delta[team_delta['constructor'].isin(top_teams)].copy()

    fig_team_delta = px.bar(
        team_delta,
        x='constructor',
        y='delta_vs_neutral',
        color='scenario',
        barmode='group',
        category_orders={'scenario': scenario_order},
        title='Top Teams: Points Delta vs Neutral Weather Scenario',
        labels={'constructor': 'Constructor', 'delta_vs_neutral': 'Points delta vs Neutral'}
    )
    st.plotly_chart(fig_team_delta, use_container_width=True)


def display_ecurie_attributes(val_a, val_b, attr_name, col_a, col_b):
    st.markdown(f"""
    <div style='background: #131a31; padding: 15px 20px; border-radius: 8px; margin-bottom: 15px; border: 1px solid rgba(255,255,255,0.05);'>
        <div style='display:flex; justify-content: space-between; margin-bottom: 8px;'>
            <span style='color: {col_a}; font-family: Teko, sans-serif; font-size: 1.4rem;'>{val_a}</span>
            <span style='color: #94a3b8; font-size: 0.9rem; text-transform: uppercase;'>{attr_name}</span>
            <span style='color: {col_b}; font-family: Teko, sans-serif; font-size: 1.4rem;'>{val_b}</span>
        </div>
        <div style='display:flex; gap: 8px;'>
            <div style='flex: 1; height: 6px; background: rgba(255,255,255,0.05); border-radius: 3px; direction: rtl;'>
                <div style='width: {val_a}%; height: 100%; background: {col_a}; border-radius: 3px;'></div>
            </div>
            <div style='flex: 1; height: 6px; background: rgba(255,255,255,0.05); border-radius: 3px;'>
                <div style='width: {val_b}%; height: 100%; background: {col_b}; border-radius: 3px;'></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def simulate_ecurie_duel_race(simulator, loader, drvs, c_id, weather_f, t_a, t_b, c_a, c_b):
    race_drivers = assign_deterministic_grid_positions(loader, drvs, race_index=0)
    res = simulator.simulate_race(circuit_id=c_id, drivers_info=race_drivers, weather_factor=weather_f, safety_car=False)
    pts = res.groupby('constructor')['actual_points'].sum().reindex([t_a, t_b]).fillna(0)
    
    head = "ÉGALITÉ" if pts.iloc[0] == pts.iloc[1] else f"{pts.idxmax()} remporte le duel"
    render_ecurie_duel_result(head, pts.iloc[0], pts.iloc[1], t_a, t_b, c_a, c_b, "COURSE 🏁")
    st.dataframe(res[['driver_name', 'constructor', 'grid_position', 'predicted_position', 'actual_points']], use_container_width=True)


def simulate_ecurie_duel_season(simulator, loader, drvs, weather, sims, limit, t_a, t_b, c_a, c_b):
    c_ids = loader.circuits['circuitId'].dropna().astype(int).tolist()
    if not c_ids: st.error("No circuits."); return
    races_data = [{'circuit_id': int(c_ids[i % len(c_ids)]), 'drivers_info': assign_deterministic_grid_positions(loader, drvs, i), 'weather_factor': float(weather_factor_from_choice("MIXED") if weather == "MIXED" else weather_factor_from_choice(weather))} for i in range(limit)]
    
    champ = simulator.simulate_season(races_data, sims)
    pts = champ.groupby('constructor')['points'].sum().reindex([t_a, t_b]).fillna(0)
    head = "ÉGALITÉ" if pts.iloc[0] == pts.iloc[1] else f"{pts.idxmax()} remporte la saison"
    
    render_ecurie_duel_result(head, round(pts.iloc[0], 1), round(pts.iloc[1], 1), t_a, t_b, c_a, c_b, "SAISON 🏆")
    st.dataframe(champ[champ['constructor'].isin([t_a, t_b])].sort_values('points', ascending=False)[['position', 'driver_name', 'constructor', 'points', 'wins']], use_container_width=True)


def render_ecurie_duel_result(head, ptA, ptB, tA, tB, cA, cB, label):
    st.markdown(f"""
    <div style='text-align: center; margin-top: 40px; padding: 30px; background: #131a31; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);'>
        <div style='color: #64748b; letter-spacing: 2px; font-size: 0.9rem;'>RÉSULTAT {label}</div>
        <h2 style='font-size: 2rem !important;'>{head}</h2>
        <div style='display: flex; justify-content: center; gap: 60px; margin-top: 30px;'>
            <div style='text-align: center;'><div style='font-size: 4rem; color: {cA}; font-family: Teko, sans-serif;'>{ptA}</div><div style='color: #94a3b8; font-size: 0.9rem;'>{tA}</div></div>
            <div style='text-align: center;'><div style='font-size: 4rem; color: {cB}; font-family: Teko, sans-serif;'>{ptB}</div><div style='color: #94a3b8; font-size: 0.9rem;'>{tB}</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def build_duel_driver_list(team_a, t_id_a, drv_a, team_b, t_id_b, drv_b):
    drvs = []
    for t_name, t_id, drv_lst in [(team_a, t_id_a, drv_a), (team_b, t_id_b, drv_b)]:
        for d in drv_lst:
            drvs.append({'driver_id': d['driver_id'], 'driver_name': d['driver_name'], 'constructor_id': t_id, 'constructor': t_name})
    return drvs


def render_ecuries_page(loader, predictor, engineer, limits):
    st.markdown("<h2 style='text-align: left; margin-bottom: 2rem; color: #94a3b8; font-size: 1.2rem; letter-spacing: 2px;'>CHOISIR LES DEUX ÉCURIES</h2>", unsafe_allow_html=True)
    c_stats, enr_res = prepare_ecuries_stats(loader)
    if c_stats.empty: st.error("No constructor data."); return
    c_stats = calculate_ecuries_strength(c_stats)
    
    names = c_stats['name'].sort_values().tolist()
    c1, c2 = st.columns(2)
    with c1: t_a = st.radio("Team A", names, key="ta", label_visibility="collapsed")
    with c2: t_b = st.radio("Team B", names, index=1 if len(names)>1 else 0, key="tb", label_visibility="collapsed")
    
    if t_a == t_b: st.warning("Select different teams.")
    
    t_map = {r['name']: r for _, r in c_stats.iterrows()}
    ida, idb = int(t_map[t_a]['constructorId']), int(t_map[t_b]['constructorId'])
    da, db = get_team_drivers(enr_res, ida), get_team_drivers(enr_res, idb)
    ca, cb = get_team_color(t_a), get_team_color(t_b)
    
    ca_str, cb_str = " - ".join([d['driver_name'] for d in da]), " - ".join([d['driver_name'] for d in db])
    cx1, cx2, cx3 = st.columns([1, 0.1, 1])
    with cx1: st.markdown(f"<div class='f1-card' style='text-align:center; border-color:{ca};'><h2 style='color:{ca};margin:0;'>{t_a}</h2><div style='color:#64748b;'>{ca_str}</div></div>", unsafe_allow_html=True)
    with cx2: st.markdown("<div style='text-align:center;margin-top:30px;font-weight:bold;color:#64748b;'>VS</div>", unsafe_allow_html=True)
    with cx3: st.markdown(f"<div class='f1-card' style='text-align:center; border-color:{cb};'><h2 style='color:{cb};margin:0;'>{t_b}</h2><div style='color:#64748b;'>{cb_str}</div></div>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["COMPÉTENCES", "SIMULATION"])
    with tab1:
        ra, rb = t_map[t_a], t_map[t_b]
        for attr, v1, v2 in [("Points", ra['points_strength'], rb['points_strength']), ("Fiabilité", ra['reliability_strength'], rb['reliability_strength']), ("Podiums", ra['podium_strength'], rb['podium_strength']), ("Qualifs", ra['quali_strength'], rb['quali_strength']), ("Régularité", ra['consistency_strength'], rb['consistency_strength'])]:
            display_ecurie_attributes(int(round(v1)), int(round(v2)), attr, ca, cb)
            
    with tab2:
        mode = st.radio("Mode", ["COURSE UNIQUE", "SAISON COMPLÈTE"], horizontal=True, label_visibility="collapsed")
        weather = st.radio("Weather Condition", ["DRY", "RAIN", "WIND", "MIXED"], horizontal=True)
        if mode == "COURSE UNIQUE":
            c_df = loader.circuits[['circuitId', 'name']].dropna().sort_values('name')
            c_name = st.selectbox("Circuit", c_df['name'].tolist())
            c_id = int(c_df[c_df['name'] == c_name]['circuitId'].iloc[0])
        else:
            sr = st.slider("Courses", 2, limits['max_races_per_season'], min(10, limits['max_races_per_season']))
            ss = st.slider("Simulations", 1, 10, 1)
            
        if st.button("LANCER LA COMPARAISON") and t_a != t_b:
            drvs = build_duel_driver_list(t_a, ida, da, t_b, idb, db)
            if len(drvs) < 2: st.error("Not enough drivers."); return
            sim = GPSimulator(predictor, loader, engineer, deterministic=True)
            if mode == "COURSE UNIQUE": simulate_ecurie_duel_race(sim, loader, drvs, c_id, weather_factor_from_choice(weather), t_a, t_b, ca, cb)
            else: simulate_ecurie_duel_season(sim, loader, drvs, weather, ss, sr, t_a, t_b, ca, cb)


# =====================================================================
# MAIN APP ROUTING
# =====================================================================

def render_page(page, loader, predictor, engineer, dataset_limits):
    if page == "QUALIFS":
        render_qualifs_page(loader)
    elif page == "COURSE":
        render_course_page(loader, predictor, engineer, dataset_limits)
    elif page == "SAISON":
        render_saison_page(loader, predictor, engineer, dataset_limits)
    elif page == "ANALYTICS":
        render_analytics_page(loader, predictor, engineer)
    elif page == "ÉCURIES":
        render_ecuries_page(loader, predictor, engineer, dataset_limits)


def main():
    st.set_page_config(page_title="F1 Predictor", layout="wide", initial_sidebar_state="collapsed")
    
    try:
        with open("assets/style.css") as f:
            st.markdown(f"<style>{{f.read()}}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

    st.markdown('''
    <div class="f1-header-container">
        <div class="f1-title-box">
            <h1 class="f1-main-title">F1 PREDICTOR</h1>
            <div class="f1-sub-title">RACE PROJECTION SYSTEM</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    pages = ["COURSE", "QUALIFS", "SAISON", "ÉCURIES", "ANALYTICS"]
    page = st.radio("Navigation", pages, horizontal=True, label_visibility="collapsed")

    if 'predictor' not in st.session_state:
        st.session_state.predictor = None
    if 'loader' not in st.session_state:
        st.session_state.loader = None
    if 'engineer' not in st.session_state:
        st.session_state.engineer = None

    force_rebuild_data = os.environ.get('F1_FORCE_REBUILD_DATA', '0') == '1'
    use_clean_cache = os.environ.get('F1_USE_CLEAN_CACHE', '1') == '1'

    loader, engineer, predictor = load_and_train_models(
        force_rebuild_data=force_rebuild_data,
        use_clean_cache=use_clean_cache
    )
    st.session_state.loader = loader
    st.session_state.engineer = engineer
    st.session_state.predictor = predictor
    dataset_limits = loader.get_dataset_limits()

    render_page(page, loader, predictor, engineer, dataset_limits)

    st.divider()
    st.markdown("---")
    st.markdown("**F1 Race Prediction System**")


if __name__ == "__main__":
    main()
