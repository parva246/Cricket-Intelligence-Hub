"""
Cricket Intelligence Hub — IPL Match Predictor
================================================
XGBoost ML + Player Analysis | Powered by Cricsheet Data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import warnings
import os
import sys
import zipfile

warnings.filterwarnings('ignore')

# Add parent dir to path so we can import squads
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)
from squads.squads_data import SQUADS, NAME_VARIATIONS, UNAVAILABLE

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Cricket Intelligence Hub",
    page_icon="🏏",
    layout="wide"
)

# ============================================================
# CONSTANTS
# ============================================================
TEAMS = sorted(SQUADS.keys())

TEAM_SHORT = {
    'Chennai Super Kings': 'CSK', 'Delhi Capitals': 'DC',
    'Gujarat Titans': 'GT', 'Kolkata Knight Riders': 'KKR',
    'Lucknow Super Giants': 'LSG', 'Mumbai Indians': 'MI',
    'Punjab Kings': 'PBKS', 'Rajasthan Royals': 'RR',
    'Royal Challengers Bengaluru': 'RCB', 'Sunrisers Hyderabad': 'SRH'
}

TEAM_COLORS = {
    'Chennai Super Kings': '#FDB913', 'Delhi Capitals': '#004C93',
    'Gujarat Titans': '#1B2133', 'Kolkata Knight Riders': '#3A225D',
    'Lucknow Super Giants': '#005DA0', 'Mumbai Indians': '#004BA0',
    'Punjab Kings': '#ED1B24', 'Rajasthan Royals': '#EA1A85',
    'Royal Challengers Bengaluru': '#EC1C24', 'Sunrisers Hyderabad': '#FF822A'
}

VENUES = {
    'Chennai': 'MA Chidambaram Stadium, Chepauk',
    'Mumbai': 'Wankhede Stadium',
    'Bengaluru': 'M Chinnaswamy Stadium',
    'Kolkata': 'Eden Gardens',
    'Hyderabad': 'Rajiv Gandhi International Stadium, Uppal',
    'Jaipur': 'Sawai Mansingh Stadium',
    'Delhi': 'Arun Jaitley Stadium',
    'Mohali': 'Punjab Cricket Association IS Bindra Stadium, Mohali',
    'Ahmedabad': 'Narendra Modi Stadium, Ahmedabad',
    'Lucknow': 'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium',
    'Dharamsala': 'Himachal Pradesh Cricket Association Stadium',
    'Visakhapatnam': 'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
}

# Map old/variant team names to current canonical names
TEAM_MAPPING = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Deccan Chargers': 'Sunrisers Hyderabad',
    'Gujarat Lions': 'Gujarat Titans',
    'Rising Pune Supergiant': 'Rising Pune Supergiants',
    'Rising Pune Supergiants': 'Rising Pune Supergiants',
    'Kings XI Punjab': 'Punjab Kings',
    'Pune Warriors': 'Rising Pune Supergiants',
    'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
    'Kochi Tuskers Kerala': 'Kochi Tuskers Kerala',
}

# Map venue name variants to canonical names
VENUE_MAPPING = {
    'M.Chinnaswamy Stadium': 'M Chinnaswamy Stadium',
    'M Chinnaswamy Stadium, Bengaluru': 'M Chinnaswamy Stadium',
    'M Chinnaswamy Stadium, Bangalore': 'M Chinnaswamy Stadium',
    'Wankhede Stadium, Mumbai': 'Wankhede Stadium',
    'MA Chidambaram Stadium': 'MA Chidambaram Stadium, Chepauk',
    'MA Chidambaram Stadium, Chepauk, Chennai': 'MA Chidambaram Stadium, Chepauk',
    'Rajiv Gandhi International Stadium': 'Rajiv Gandhi International Stadium, Uppal',
    'Rajiv Gandhi International Stadium, Uppal, Hyderabad': 'Rajiv Gandhi International Stadium, Uppal',
    'Rajiv Gandhi International Cricket Stadium': 'Rajiv Gandhi International Stadium, Uppal',
    'Eden Gardens, Kolkata': 'Eden Gardens',
    'Feroz Shah Kotla': 'Arun Jaitley Stadium',
    'Arun Jaitley Stadium, Delhi': 'Arun Jaitley Stadium',
    'Punjab Cricket Association Stadium, Mohali': 'Punjab Cricket Association IS Bindra Stadium, Mohali',
    'Punjab Cricket Association IS Bindra Stadium': 'Punjab Cricket Association IS Bindra Stadium, Mohali',
    'Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh': 'Punjab Cricket Association IS Bindra Stadium, Mohali',
    'IS Bindra Stadium': 'Punjab Cricket Association IS Bindra Stadium, Mohali',
    'Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh': 'Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur',
    'Sawai Mansingh Stadium, Jaipur': 'Sawai Mansingh Stadium',
    'Sardar Patel Stadium, Motera': 'Narendra Modi Stadium, Ahmedabad',
    'Narendra Modi Stadium': 'Narendra Modi Stadium, Ahmedabad',
    'Maharashtra Cricket Association Stadium, Pune': 'Maharashtra Cricket Association Stadium',
    'Dr DY Patil Sports Academy': 'Dr DY Patil Sports Academy, Mumbai',
    'Brabourne Stadium': 'Brabourne Stadium, Mumbai',
    'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam': 'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
    'ACA-VDCA Stadium': 'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
    'Himachal Pradesh Cricket Association Stadium, Dharamsala': 'Himachal Pradesh Cricket Association Stadium',
    'HPCA Stadium': 'Himachal Pradesh Cricket Association Stadium',
    'Zayed Cricket Stadium, Abu Dhabi': 'Sheikh Zayed Stadium',
    'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow': 'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium',
    'Ekana Cricket Stadium': 'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium',
}

SEASON_MAPPING = {
    '2007/08': '2008', '2009/10': '2010', '2020/21': '2020',
}

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_and_clean_data():
    """Load matches.csv and deliveries.csv from data/processed/ folder."""
    data_dir = os.path.join(parent_dir, 'data', 'processed')

    # Fallback: check same directory as script, then parent
    matches_path = os.path.join(data_dir, 'matches.csv')
    deliveries_path = os.path.join(data_dir, 'deliveries.csv')

    if not os.path.exists(matches_path):
        matches_path = os.path.join(parent_dir, 'matches.csv')
        deliveries_path = os.path.join(parent_dir, 'deliveries.csv')

    if not os.path.exists(matches_path):
        matches_path = os.path.join(script_dir, 'matches.csv')
        deliveries_path = os.path.join(script_dir, 'deliveries.csv')

    if not os.path.exists(matches_path) or not os.path.exists(deliveries_path):
        st.error("Data files not found! Place matches.csv and deliveries.csv in data/processed/ folder.")
        st.info("Run scripts/data_pipeline.py on Google Colab first to generate these files from Cricsheet data.")
        st.stop()

    matches = pd.read_csv(matches_path)
    deliveries = pd.read_csv(deliveries_path)

    # Normalize team names
    for col in ['team1', 'team2', 'toss_winner', 'winner']:
        if col in matches.columns:
            matches[col] = matches[col].replace(TEAM_MAPPING)
    for col in ['batting_team', 'bowling_team']:
        if col in deliveries.columns:
            deliveries[col] = deliveries[col].replace(TEAM_MAPPING)

    # Normalize venue names
    if 'venue' in matches.columns:
        matches['venue'] = matches['venue'].replace(VENUE_MAPPING)

    # Normalize season format
    if 'season' in matches.columns:
        matches['season'] = matches['season'].astype(str).replace(SEASON_MAPPING)

    # Normalize city names
    city_mapping = {
        'Bangalore': 'Bengaluru',
        'Navi Mumbai': 'Mumbai',
        'New Chandigarh': 'Mohali',
    }
    if 'city' in matches.columns:
        matches['city'] = matches['city'].replace(city_mapping)

    # Filter and sort
    if 'result' in matches.columns:
        matches = matches[matches['result'] != 'no result']
    matches = matches.dropna(subset=['winner'])
    matches['date'] = pd.to_datetime(matches['date'])
    matches = matches.sort_values('date').reset_index(drop=True)

    # Handle column name differences (Cricsheet uses 'batter', Kaggle uses 'batsman')
    if 'batter' in deliveries.columns and 'batsman' not in deliveries.columns:
        deliveries['batsman'] = deliveries['batter']
    if 'batsman_runs' not in deliveries.columns and 'batsman' in deliveries.columns:
        # If batsman_runs is missing, it should already be there from pipeline
        pass

    return matches, deliveries


# ============================================================
# PLAYER STATS
# ============================================================
def find_player_in_data(player_name, deliveries):
    """Find how a player's name appears in the deliveries data."""
    bat_col = 'batter' if 'batter' in deliveries.columns else 'batsman'

    all_players = set()
    if bat_col in deliveries.columns:
        all_players.update(deliveries[bat_col].unique())
    if 'bowler' in deliveries.columns:
        all_players.update(deliveries['bowler'].unique())

    if player_name in all_players:
        return player_name

    # Check variations
    for data_name, squad_name in NAME_VARIATIONS.items():
        if squad_name == player_name and data_name in all_players:
            return data_name

    # Partial match — last name
    last_name = player_name.split()[-1]
    matches = [p for p in all_players if last_name in p]
    if len(matches) == 1:
        return matches[0]

    return None


@st.cache_data
def get_player_batting_stats(_deliveries, player_name):
    """Get batting stats for a player from historical data."""
    data_name = find_player_in_data(player_name, _deliveries)
    if data_name is None:
        return None

    bat_col = 'batter' if 'batter' in _deliveries.columns else 'batsman'
    runs_col = 'batsman_runs' if 'batsman_runs' in _deliveries.columns else 'runs_off_bat'

    player_data = _deliveries[_deliveries[bat_col] == data_name]
    if len(player_data) == 0:
        return None

    total_runs = player_data[runs_col].sum()

    # Balls faced (exclude wides)
    if 'wide_runs' in player_data.columns:
        balls_faced = len(player_data[player_data['wide_runs'] == 0])
    elif 'wides' in player_data.columns:
        balls_faced = len(player_data[player_data['wides'] == 0])
    else:
        balls_faced = len(player_data)

    strike_rate = (total_runs / max(balls_faced, 1)) * 100
    innings = player_data['match_id'].nunique()

    # Count dismissals
    dismissals = 0
    if 'player_dismissed' in _deliveries.columns:
        dismissals = int((_deliveries['player_dismissed'] == data_name).sum())
    elif 'is_wicket' in player_data.columns:
        dismissals = player_data['is_wicket'].sum()

    average = total_runs / max(dismissals, 1)

    # Powerplay (overs 0-5)
    pp_data = player_data[player_data['over'] < 6]
    pp_runs = pp_data[runs_col].sum()
    if 'wide_runs' in pp_data.columns:
        pp_balls = len(pp_data[pp_data['wide_runs'] == 0])
    elif 'wides' in pp_data.columns:
        pp_balls = len(pp_data[pp_data['wides'] == 0])
    else:
        pp_balls = len(pp_data)
    pp_sr = (pp_runs / max(pp_balls, 1)) * 100

    # Death overs (16-19)
    death_data = player_data[player_data['over'] >= 16]
    death_runs = death_data[runs_col].sum()
    if 'wide_runs' in death_data.columns:
        death_balls = len(death_data[death_data['wide_runs'] == 0])
    elif 'wides' in death_data.columns:
        death_balls = len(death_data[death_data['wides'] == 0])
    else:
        death_balls = len(death_data)
    death_sr = (death_runs / max(death_balls, 1)) * 100

    # Boundaries
    fours = len(player_data[player_data[runs_col] == 4])
    sixes = len(player_data[player_data[runs_col] == 6])

    return {
        'name': player_name, 'data_name': data_name,
        'innings': innings, 'runs': total_runs, 'balls': balls_faced,
        'average': round(average, 1), 'strike_rate': round(strike_rate, 1),
        'fours': fours, 'sixes': sixes,
        'pp_sr': round(pp_sr, 1), 'death_sr': round(death_sr, 1),
        'dismissals': dismissals
    }


@st.cache_data
def get_player_bowling_stats(_deliveries, player_name):
    """Get bowling stats for a player from historical data."""
    data_name = find_player_in_data(player_name, _deliveries)
    if data_name is None:
        return None

    player_data = _deliveries[_deliveries['bowler'] == data_name]
    if len(player_data) == 0:
        return None

    runs_conceded = player_data['total_runs'].sum()
    balls_bowled = len(player_data)
    overs = balls_bowled / 6
    economy = runs_conceded / max(overs, 1)

    wickets = player_data['is_wicket'].sum() if 'is_wicket' in player_data.columns else 0
    innings = player_data['match_id'].nunique()
    bowling_avg = runs_conceded / max(wickets, 1)
    bowling_sr = balls_bowled / max(wickets, 1)

    # Powerplay bowling
    pp_data = player_data[player_data['over'] < 6]
    pp_runs = pp_data['total_runs'].sum()
    pp_economy = pp_runs / max(len(pp_data) / 6, 1)
    pp_wickets = pp_data['is_wicket'].sum() if 'is_wicket' in pp_data.columns else 0

    # Death overs bowling
    death_data = player_data[player_data['over'] >= 16]
    death_runs = death_data['total_runs'].sum()
    death_economy = death_runs / max(len(death_data) / 6, 1)
    death_wickets = death_data['is_wicket'].sum() if 'is_wicket' in death_data.columns else 0

    return {
        'name': player_name, 'data_name': data_name,
        'innings': innings, 'wickets': wickets,
        'runs_conceded': runs_conceded, 'economy': round(economy, 2),
        'average': round(bowling_avg, 1), 'strike_rate': round(bowling_sr, 1),
        'pp_economy': round(pp_economy, 2), 'pp_wickets': pp_wickets,
        'death_economy': round(death_economy, 2), 'death_wickets': death_wickets
    }


@st.cache_data
def get_matchup_stats(_deliveries, batsman_name, bowler_name):
    """Get head-to-head stats between a batsman and bowler."""
    bat_data_name = find_player_in_data(batsman_name, _deliveries)
    bowl_data_name = find_player_in_data(bowler_name, _deliveries)
    if bat_data_name is None or bowl_data_name is None:
        return None

    bat_col = 'batter' if 'batter' in _deliveries.columns else 'batsman'
    runs_col = 'batsman_runs' if 'batsman_runs' in _deliveries.columns else 'runs_off_bat'

    matchup = _deliveries[
        (_deliveries[bat_col] == bat_data_name) &
        (_deliveries['bowler'] == bowl_data_name)
    ]
    if len(matchup) == 0:
        return None

    runs = matchup[runs_col].sum()
    if 'wide_runs' in matchup.columns:
        balls = len(matchup[matchup['wide_runs'] == 0])
    elif 'wides' in matchup.columns:
        balls = len(matchup[matchup['wides'] == 0])
    else:
        balls = len(matchup)
    sr = (runs / max(balls, 1)) * 100

    dismissals = 0
    if 'is_wicket' in matchup.columns and 'player_dismissed' in matchup.columns:
        dismissals = matchup[matchup['player_dismissed'] == bat_data_name]['is_wicket'].sum()
    elif 'is_wicket' in matchup.columns:
        dismissals = matchup['is_wicket'].sum()

    fours = len(matchup[matchup[runs_col] == 4])
    sixes = len(matchup[matchup[runs_col] == 6])

    return {
        'batsman': batsman_name, 'bowler': bowler_name,
        'runs': runs, 'balls': balls, 'strike_rate': round(sr, 1),
        'dismissals': dismissals, 'fours': fours, 'sixes': sixes,
        'dominance': 'Batsman' if sr > 130 and dismissals == 0 else ('Bowler' if dismissals >= 2 or sr < 100 else 'Even')
    }


# ============================================================
# TEAM FEATURE FUNCTIONS
# ============================================================
def get_team_win_pct(team, date, df, last_n=10):
    past = df[(df['date'] < date) & ((df['team1'] == team) | (df['team2'] == team))].tail(last_n)
    if len(past) == 0: return 0.5
    return (past['winner'] == team).sum() / len(past)

def get_head_to_head(team1, team2, date, df):
    past = df[(df['date'] < date) & (((df['team1'] == team1) & (df['team2'] == team2)) | ((df['team1'] == team2) & (df['team2'] == team1)))]
    if len(past) == 0: return 0.5
    return (past['winner'] == team1).sum() / len(past)

def get_venue_win_rate(team, venue, date, df):
    past = df[(df['date'] < date) & (df['venue'] == venue) & ((df['team1'] == team) | (df['team2'] == team))]
    if len(past) == 0: return 0.5
    return (past['winner'] == team).sum() / len(past)

def get_team_batting_stats(team, date, matches_df, deliveries_df, last_n=10):
    past_ids = matches_df[(matches_df['date'] < date) & ((matches_df['team1'] == team) | (matches_df['team2'] == team))].tail(last_n)['id'].values
    batting = deliveries_df[(deliveries_df['match_id'].isin(past_ids)) & (deliveries_df['batting_team'] == team)]
    if len(batting) == 0: return 7.5, 140
    total_runs = batting['total_runs'].sum()
    total_overs = batting.groupby('match_id')['over'].nunique().sum()
    return round(total_runs / max(total_overs, 1), 2), round(batting.groupby('match_id')['total_runs'].sum().mean(), 2)

def get_team_bowling_stats(team, date, matches_df, deliveries_df, last_n=10):
    past_ids = matches_df[(matches_df['date'] < date) & ((matches_df['team1'] == team) | (matches_df['team2'] == team))].tail(last_n)['id'].values
    bowling = deliveries_df[(deliveries_df['match_id'].isin(past_ids)) & (deliveries_df['bowling_team'] == team)]
    if len(bowling) == 0: return 8.0, 3.0
    total_runs = bowling['total_runs'].sum()
    total_overs = bowling.groupby('match_id')['over'].nunique().sum()
    wickets = bowling['is_wicket'].sum() if 'is_wicket' in bowling.columns else 0
    return round(total_runs / max(total_overs, 1), 2), round(wickets / max(len(past_ids), 1), 2)

def get_phase_avg(team, date, matches_df, deliveries_df, phase='pp', last_n=10):
    """Get average runs in a match phase. phase: 'pp' (0-5), 'middle' (6-15), 'death' (16-19)"""
    past_ids = matches_df[(matches_df['date'] < date) & ((matches_df['team1'] == team) | (matches_df['team2'] == team))].tail(last_n)['id'].values
    batting = deliveries_df[(deliveries_df['match_id'].isin(past_ids)) & (deliveries_df['batting_team'] == team)]

    if phase == 'pp':
        phase_data = batting[batting['over'] < 6]
        default = 45
    elif phase == 'middle':
        phase_data = batting[(batting['over'] >= 6) & (batting['over'] < 16)]
        default = 70
    else:  # death
        phase_data = batting[batting['over'] >= 16]
        default = 40

    if len(phase_data) == 0: return default
    return round(phase_data.groupby('match_id')['total_runs'].sum().mean(), 2)

def get_chase_rate(venue, date, df):
    past = df[(df['date'] < date) & (df['venue'] == venue)]
    if len(past) == 0: return 0.52
    if 'win_by_wickets' in df.columns:
        chasing_wins = past[past['win_by_wickets'] > 0].shape[0]
    elif 'result_margin' in df.columns and 'result' in df.columns:
        chasing_wins = past[past['result'] == 'wickets'].shape[0]
    else:
        chasing_wins = past[past['winner'] == past['team2']].shape[0]
    return round(chasing_wins / len(past), 2)


# ============================================================
# MODEL
# ============================================================
@st.cache_resource
def train_model(_matches, _deliveries):
    features = []
    for idx, row in _matches.iterrows():
        t1_rr, t1_avg = get_team_batting_stats(row['team1'], row['date'], _matches, _deliveries)
        t2_rr, t2_avg = get_team_batting_stats(row['team2'], row['date'], _matches, _deliveries)
        t1_econ, t1_wpm = get_team_bowling_stats(row['team1'], row['date'], _matches, _deliveries)
        t2_econ, t2_wpm = get_team_bowling_stats(row['team2'], row['date'], _matches, _deliveries)

        features.append({
            'team1_win_pct': get_team_win_pct(row['team1'], row['date'], _matches),
            'team2_win_pct': get_team_win_pct(row['team2'], row['date'], _matches),
            'head_to_head': get_head_to_head(row['team1'], row['team2'], row['date'], _matches),
            'toss_win': 1 if row['toss_winner'] == row['team1'] else 0,
            'chose_bat': 1 if row['toss_decision'] == 'bat' else 0,
            'venue_win_rate': get_venue_win_rate(row['team1'], row['venue'], row['date'], _matches),
            'chase_rate_venue': get_chase_rate(row['venue'], row['date'], _matches),
            'team1_run_rate': t1_rr, 'team2_run_rate': t2_rr,
            'team1_avg_score': t1_avg, 'team2_avg_score': t2_avg,
            'team1_bowl_economy': t1_econ, 'team2_bowl_economy': t2_econ,
            'team1_wickets_pm': t1_wpm, 'team2_wickets_pm': t2_wpm,
            'team1_powerplay': get_phase_avg(row['team1'], row['date'], _matches, _deliveries, 'pp'),
            'team2_powerplay': get_phase_avg(row['team2'], row['date'], _matches, _deliveries, 'pp'),
            'team1_death': get_phase_avg(row['team1'], row['date'], _matches, _deliveries, 'death'),
            'team2_death': get_phase_avg(row['team2'], row['date'], _matches, _deliveries, 'death'),
            'team1_middle': get_phase_avg(row['team1'], row['date'], _matches, _deliveries, 'middle'),
            'team2_middle': get_phase_avg(row['team2'], row['date'], _matches, _deliveries, 'middle'),
            'target': 1 if row['winner'] == row['team1'] else 0
        })

    feature_df = pd.DataFrame(features)
    X = feature_df.drop('target', axis=1)
    y = feature_df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

    return model, accuracy, feature_importance


# ============================================================
# PREDICTION
# ============================================================
def make_prediction(model, team1, team2, venue, toss_winner, toss_decision, matches, deliveries):
    ref_date = pd.Timestamp('2026-04-01')
    t1_rr, t1_avg = get_team_batting_stats(team1, ref_date, matches, deliveries)
    t2_rr, t2_avg = get_team_batting_stats(team2, ref_date, matches, deliveries)
    t1_econ, t1_wpm = get_team_bowling_stats(team1, ref_date, matches, deliveries)
    t2_econ, t2_wpm = get_team_bowling_stats(team2, ref_date, matches, deliveries)

    input_data = {
        'team1_win_pct': get_team_win_pct(team1, ref_date, matches),
        'team2_win_pct': get_team_win_pct(team2, ref_date, matches),
        'head_to_head': get_head_to_head(team1, team2, ref_date, matches),
        'toss_win': 1 if toss_winner == team1 else 0,
        'chose_bat': 1 if toss_decision == 'Bat' else 0,
        'venue_win_rate': get_venue_win_rate(team1, venue, ref_date, matches),
        'chase_rate_venue': get_chase_rate(venue, ref_date, matches),
        'team1_run_rate': t1_rr, 'team2_run_rate': t2_rr,
        'team1_avg_score': t1_avg, 'team2_avg_score': t2_avg,
        'team1_bowl_economy': t1_econ, 'team2_bowl_economy': t2_econ,
        'team1_wickets_pm': t1_wpm, 'team2_wickets_pm': t2_wpm,
        'team1_powerplay': get_phase_avg(team1, ref_date, matches, deliveries, 'pp'),
        'team2_powerplay': get_phase_avg(team2, ref_date, matches, deliveries, 'pp'),
        'team1_death': get_phase_avg(team1, ref_date, matches, deliveries, 'death'),
        'team2_death': get_phase_avg(team2, ref_date, matches, deliveries, 'death'),
        'team1_middle': get_phase_avg(team1, ref_date, matches, deliveries, 'middle'),
        'team2_middle': get_phase_avg(team2, ref_date, matches, deliveries, 'middle'),
    }

    prob = model.predict_proba(pd.DataFrame([input_data]))[0]

    return {
        'team1_win_prob': prob[1], 'team2_win_prob': prob[0],
        'team1_win_pct': input_data['team1_win_pct'], 'team2_win_pct': input_data['team2_win_pct'],
        'head_to_head': input_data['head_to_head'],
        'team1_run_rate': t1_rr, 'team2_run_rate': t2_rr,
        'team1_avg_score': t1_avg, 'team2_avg_score': t2_avg,
        'team1_bowl_economy': t1_econ, 'team2_bowl_economy': t2_econ,
        'team1_powerplay': input_data['team1_powerplay'], 'team2_powerplay': input_data['team2_powerplay'],
        'team1_death': input_data['team1_death'], 'team2_death': input_data['team2_death'],
        'team1_middle': input_data['team1_middle'], 'team2_middle': input_data['team2_middle'],
        'venue_win_rate': input_data['venue_win_rate'],
        'chase_rate': input_data['chase_rate_venue'],
    }


# ============================================================
# MAIN APP
# ============================================================
def main():
    st.markdown("""
    <h1 style='text-align: center;'>🏏 Cricket Intelligence Hub</h1>
    <p style='text-align: center; color: gray;'>XGBoost ML + Player Analysis | Powered by Cricsheet Data</p>
    <hr>
    """, unsafe_allow_html=True)

    with st.spinner("Loading IPL data..."):
        matches, deliveries = load_and_clean_data()

    with st.spinner("Training model... (cached after first run)"):
        model, accuracy, feature_importance = train_model(matches, deliveries)

    total_seasons = matches['season'].nunique()
    st.markdown(f"<p style='text-align:center;'>Model Accuracy: <strong>{accuracy:.1%}</strong> | Matches: <strong>{len(matches)}</strong> | Seasons: <strong>{total_seasons}</strong> | Algorithm: <strong>XGBoost</strong></p><hr>", unsafe_allow_html=True)

    # ── TABS ──
    tab1, tab2, tab3 = st.tabs(["🏏 Match Prediction", "👤 Player Stats", "⚔️ Player Matchups"])

    # ============================================================
    # TAB 1: MATCH PREDICTION
    # ============================================================
    with tab1:
        st.subheader("Match Details")
        col1, col2, col3 = st.columns(3)
        with col1:
            team1 = st.selectbox("Team A", TEAMS, index=TEAMS.index('Sunrisers Hyderabad') if 'Sunrisers Hyderabad' in TEAMS else 0, key='t1')
        with col2:
            team2_opts = [t for t in TEAMS if t != team1]
            default_t2 = team2_opts.index('Royal Challengers Bengaluru') if 'Royal Challengers Bengaluru' in team2_opts else 0
            team2 = st.selectbox("Team B", team2_opts, index=default_t2, key='t2')
        with col3:
            venue_city = st.selectbox("Venue", list(VENUES.keys()), key='venue')
            venue = VENUES[venue_city]

        col4, col5 = st.columns(2)
        with col4:
            toss_winner = st.selectbox("Toss Won By", [team1, team2], key='toss')
        with col5:
            toss_decision = st.selectbox("Toss Decision", ['Field', 'Bat'], key='decision')

        # Playing XI selection
        st.markdown("---")
        st.subheader("Playing XI (Optional — for player analysis)")

        xi_col1, xi_col2 = st.columns(2)
        with xi_col1:
            available1 = [p for p in SQUADS[team1] if p not in UNAVAILABLE.get(team1, [])]
            xi_team1 = st.multiselect(
                f"{TEAM_SHORT[team1]} Playing XI (select 11)",
                available1, default=available1[:11],
                max_selections=11, key='xi1'
            )
        with xi_col2:
            available2 = [p for p in SQUADS[team2] if p not in UNAVAILABLE.get(team2, [])]
            xi_team2 = st.multiselect(
                f"{TEAM_SHORT[team2]} Playing XI (select 11)",
                available2, default=available2[:11],
                max_selections=11, key='xi2'
            )

        st.markdown("<br>", unsafe_allow_html=True)
        predict_clicked = st.button("🏏  Predict Match", type="primary", use_container_width=True)

        if predict_clicked:
            with st.spinner("Analysing..."):
                stats = make_prediction(model, team1, team2, venue, toss_winner, toss_decision, matches, deliveries)

            color1 = TEAM_COLORS.get(team1, '#333')
            color2 = TEAM_COLORS.get(team2, '#666')

            # Winner prediction bar
            st.markdown("---")
            st.subheader("Match Winner Prediction")

            fig = go.Figure()
            fig.add_trace(go.Bar(y=[''], x=[stats['team1_win_prob']*100], orientation='h', name=TEAM_SHORT[team1],
                                 marker_color=color1, text=f"{TEAM_SHORT[team1]} {stats['team1_win_prob']:.1%}", textposition='inside', textfont=dict(size=16, color='white')))
            fig.add_trace(go.Bar(y=[''], x=[stats['team2_win_prob']*100], orientation='h', name=TEAM_SHORT[team2],
                                 marker_color=color2, text=f"{TEAM_SHORT[team2]} {stats['team2_win_prob']:.1%}", textposition='inside', textfont=dict(size=16, color='white')))
            fig.update_layout(barmode='stack', height=80, margin=dict(l=0,r=0,t=0,b=0), showlegend=False,
                              xaxis=dict(visible=False), yaxis=dict(visible=False), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

            if stats['team1_win_prob'] > 0.65: verdict = f"STRONG PREDICTION: {team1}"
            elif stats['team2_win_prob'] > 0.65: verdict = f"STRONG PREDICTION: {team2}"
            elif stats['team1_win_prob'] > 0.55: verdict = f"SLIGHT EDGE: {team1}"
            elif stats['team2_win_prob'] > 0.55: verdict = f"SLIGHT EDGE: {team2}"
            else: verdict = "VERY CLOSE — Could go either way!"
            st.markdown(f"<h3 style='text-align:center;'>{verdict}</h3>", unsafe_allow_html=True)

            # Phase breakdown
            st.markdown("---")
            st.subheader("Phase-by-Phase Breakdown")
            pp_col, mid_col, death_col = st.columns(3)
            with pp_col:
                pp_leader = team1 if stats['team1_powerplay'] > stats['team2_powerplay'] else team2
                st.metric("Powerplay Leader (1-6)", TEAM_SHORT[pp_leader], f"+{abs(stats['team1_powerplay']-stats['team2_powerplay']):.0f} avg runs")
                st.caption(f"{TEAM_SHORT[team1]}: {stats['team1_powerplay']:.0f} | {TEAM_SHORT[team2]}: {stats['team2_powerplay']:.0f}")
            with mid_col:
                mid_leader = team1 if stats['team1_middle'] > stats['team2_middle'] else team2
                st.metric("Middle Overs (7-16)", TEAM_SHORT[mid_leader], f"+{abs(stats['team1_middle']-stats['team2_middle']):.0f} avg runs")
                st.caption(f"{TEAM_SHORT[team1]}: {stats['team1_middle']:.0f} | {TEAM_SHORT[team2]}: {stats['team2_middle']:.0f}")
            with death_col:
                death_leader = team1 if stats['team1_death'] > stats['team2_death'] else team2
                st.metric("Death Overs (17-20)", TEAM_SHORT[death_leader], f"+{abs(stats['team1_death']-stats['team2_death']):.0f} avg runs")
                st.caption(f"{TEAM_SHORT[team1]}: {stats['team1_death']:.0f} | {TEAM_SHORT[team2]}: {stats['team2_death']:.0f}")

            # Phase comparison chart
            st.markdown("---")
            phases = ['Powerplay (1-6)', 'Middle (7-16)', 'Death (17-20)']
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(name=TEAM_SHORT[team1], x=phases, y=[stats['team1_powerplay'], stats['team1_middle'], stats['team1_death']], marker_color=color1,
                                  text=[f"{v:.0f}" for v in [stats['team1_powerplay'], stats['team1_middle'], stats['team1_death']]], textposition='outside'))
            fig2.add_trace(go.Bar(name=TEAM_SHORT[team2], x=phases, y=[stats['team2_powerplay'], stats['team2_middle'], stats['team2_death']], marker_color=color2,
                                  text=[f"{v:.0f}" for v in [stats['team2_powerplay'], stats['team2_middle'], stats['team2_death']]], textposition='outside'))
            fig2.update_layout(title="Average Runs by Match Phase", barmode='group', height=350, margin=dict(l=40,r=20,t=50,b=40),
                               legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5))
            st.plotly_chart(fig2, use_container_width=True)

            # Player XI Stats
            if len(xi_team1) > 0 and len(xi_team2) > 0:
                st.markdown("---")
                st.subheader("Playing XI — Key Player Stats")

                pcol1, pcol2 = st.columns(2)
                with pcol1:
                    st.markdown(f"**{TEAM_SHORT[team1]} Batsmen**")
                    bat_stats1 = [s for p in xi_team1 if (s := get_player_batting_stats(deliveries, p)) and s['innings'] >= 5]
                    if bat_stats1:
                        df1 = pd.DataFrame(bat_stats1)[['name','innings','runs','average','strike_rate','pp_sr','death_sr']].sort_values('runs', ascending=False)
                        df1.columns = ['Player','Inn','Runs','Avg','SR','PP SR','Death SR']
                        st.dataframe(df1.set_index('Player'), use_container_width=True)
                    else:
                        st.info("No historical batting data found")

                with pcol2:
                    st.markdown(f"**{TEAM_SHORT[team2]} Batsmen**")
                    bat_stats2 = [s for p in xi_team2 if (s := get_player_batting_stats(deliveries, p)) and s['innings'] >= 5]
                    if bat_stats2:
                        df2 = pd.DataFrame(bat_stats2)[['name','innings','runs','average','strike_rate','pp_sr','death_sr']].sort_values('runs', ascending=False)
                        df2.columns = ['Player','Inn','Runs','Avg','SR','PP SR','Death SR']
                        st.dataframe(df2.set_index('Player'), use_container_width=True)
                    else:
                        st.info("No historical batting data found")

                bcol1, bcol2 = st.columns(2)
                with bcol1:
                    st.markdown(f"**{TEAM_SHORT[team1]} Bowlers**")
                    bowl_stats1 = [s for p in xi_team1 if (s := get_player_bowling_stats(deliveries, p)) and s['innings'] >= 5 and s['wickets'] >= 3]
                    if bowl_stats1:
                        bdf1 = pd.DataFrame(bowl_stats1)[['name','innings','wickets','economy','average','pp_economy','death_economy']].sort_values('wickets', ascending=False)
                        bdf1.columns = ['Player','Inn','Wkts','Econ','Avg','PP Econ','Death Econ']
                        st.dataframe(bdf1.set_index('Player'), use_container_width=True)
                    else:
                        st.info("No historical bowling data found")

                with bcol2:
                    st.markdown(f"**{TEAM_SHORT[team2]} Bowlers**")
                    bowl_stats2 = [s for p in xi_team2 if (s := get_player_bowling_stats(deliveries, p)) and s['innings'] >= 5 and s['wickets'] >= 3]
                    if bowl_stats2:
                        bdf2 = pd.DataFrame(bowl_stats2)[['name','innings','wickets','economy','average','pp_economy','death_economy']].sort_values('wickets', ascending=False)
                        bdf2.columns = ['Player','Inn','Wkts','Econ','Avg','PP Econ','Death Econ']
                        st.dataframe(bdf2.set_index('Player'), use_container_width=True)
                    else:
                        st.info("No historical bowling data found")

                # Key Matchups
                st.markdown("---")
                st.subheader("Key Player Matchups")

                matchups_found = []
                for bat in xi_team1:
                    for bowl in xi_team2:
                        m = get_matchup_stats(deliveries, bat, bowl)
                        if m and m['balls'] >= 6:
                            m['context'] = f"{TEAM_SHORT[team1]} bat vs {TEAM_SHORT[team2]} bowl"
                            matchups_found.append(m)
                for bat in xi_team2:
                    for bowl in xi_team1:
                        m = get_matchup_stats(deliveries, bat, bowl)
                        if m and m['balls'] >= 6:
                            m['context'] = f"{TEAM_SHORT[team2]} bat vs {TEAM_SHORT[team1]} bowl"
                            matchups_found.append(m)

                if matchups_found:
                    matchups_found.sort(key=lambda x: x['balls'], reverse=True)
                    mdf = pd.DataFrame(matchups_found[:15])[['batsman','bowler','runs','balls','strike_rate','dismissals','fours','sixes','dominance']]
                    mdf.columns = ['Batsman','Bowler','Runs','Balls','SR','Dismissals','4s','6s','Edge']

                    def color_edge(val):
                        if val == 'Batsman': return 'color: green; font-weight: bold'
                        elif val == 'Bowler': return 'color: red; font-weight: bold'
                        return ''

                    st.dataframe(mdf.style.map(color_edge, subset=['Edge']), use_container_width=True, hide_index=True)
                else:
                    st.info("No historical matchup data found between selected players")

            # Venue insight
            st.markdown("---")
            st.subheader("Venue Insight")
            chase_pct = stats['chase_rate'] * 100
            st.markdown(f"**{venue}** ({venue_city})")
            st.markdown(f"- Chasing wins: **{chase_pct:.0f}%** | Defending wins: **{100-chase_pct:.0f}%**")
            if chase_pct > 55: st.markdown("- This venue **favours chasing**")
            elif chase_pct < 45: st.markdown("- This venue **favours defending**")
            else: st.markdown("- This venue is **neutral**")

    # ============================================================
    # TAB 2: PLAYER STATS
    # ============================================================
    with tab2:
        st.subheader("Individual Player Stats Lookup")

        ps_team = st.selectbox("Select Team", TEAMS, key='ps_team')
        ps_player = st.selectbox("Select Player", SQUADS[ps_team], key='ps_player')

        if st.button("Get Stats", key='ps_btn'):
            with st.spinner("Loading stats..."):
                bat = get_player_batting_stats(deliveries, ps_player)
                bowl = get_player_bowling_stats(deliveries, ps_player)

            if bat is None and bowl is None:
                st.warning(f"No historical IPL data found for {ps_player}. They may be a new/uncapped player.")
            else:
                if bat and bat['innings'] > 0:
                    st.markdown("### Batting Stats")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Innings", bat['innings'])
                    m2.metric("Runs", bat['runs'])
                    m3.metric("Average", bat['average'])
                    m4.metric("Strike Rate", bat['strike_rate'])

                    m5, m6, m7, m8 = st.columns(4)
                    m5.metric("4s", bat['fours'])
                    m6.metric("6s", bat['sixes'])
                    m7.metric("PP Strike Rate", bat['pp_sr'])
                    m8.metric("Death SR", bat['death_sr'])

                if bowl and bowl['wickets'] > 0:
                    st.markdown("### Bowling Stats")
                    b1, b2, b3, b4 = st.columns(4)
                    b1.metric("Innings", bowl['innings'])
                    b2.metric("Wickets", bowl['wickets'])
                    b3.metric("Economy", bowl['economy'])
                    b4.metric("Average", bowl['average'])

                    b5, b6, b7, b8 = st.columns(4)
                    b5.metric("PP Economy", bowl['pp_economy'])
                    b6.metric("PP Wickets", bowl['pp_wickets'])
                    b7.metric("Death Economy", bowl['death_economy'])
                    b8.metric("Death Wickets", bowl['death_wickets'])

    # ============================================================
    # TAB 3: PLAYER MATCHUPS
    # ============================================================
    with tab3:
        st.subheader("Head-to-Head: Batsman vs Bowler")

        mu_col1, mu_col2 = st.columns(2)
        with mu_col1:
            mu_team1 = st.selectbox("Batsman's Team", TEAMS, key='mu_t1')
            mu_batsman = st.selectbox("Select Batsman", SQUADS[mu_team1], key='mu_bat')
        with mu_col2:
            mu_team2 = st.selectbox("Bowler's Team", [t for t in TEAMS if t != mu_team1], key='mu_t2')
            mu_bowler = st.selectbox("Select Bowler", SQUADS[mu_team2], key='mu_bowl')

        if st.button("Get Matchup", key='mu_btn'):
            with st.spinner("Searching matchup data..."):
                matchup = get_matchup_stats(deliveries, mu_batsman, mu_bowler)

            if matchup is None:
                st.warning(f"No head-to-head data found between {mu_batsman} and {mu_bowler}")
            else:
                st.markdown(f"### {mu_batsman} vs {mu_bowler}")

                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Runs Scored", matchup['runs'])
                mc2.metric("Balls Faced", matchup['balls'])
                mc3.metric("Strike Rate", matchup['strike_rate'])
                mc4.metric("Dismissals", matchup['dismissals'])

                mc5, mc6, mc7 = st.columns(3)
                mc5.metric("Fours", matchup['fours'])
                mc6.metric("Sixes", matchup['sixes'])
                mc7.metric("Edge", matchup['dominance'])

                if matchup['dominance'] == 'Batsman':
                    st.success(f"{mu_batsman} has dominated this matchup!")
                elif matchup['dominance'] == 'Bowler':
                    st.error(f"{mu_bowler} has the upper hand in this matchup!")
                else:
                    st.info("This is an evenly balanced matchup")

    # ── SIDEBAR ──
    with st.sidebar:
        st.markdown("### About")
        st.markdown(f"""
        **Cricket Intelligence Hub** — IPL Match Predictor with player-level analysis.

        **Data Source:** [Cricsheet.org](https://cricsheet.org) (free, open data)

        **Features:**
        - Match winner prediction
        - Phase-by-phase breakdown
        - Playing XI selection
        - Individual player stats
        - Batsman vs Bowler matchups
        - Powerplay / Death overs analysis
        """)
        st.markdown("---")
        st.markdown(f"**Model Accuracy:** {accuracy:.1%}")
        st.markdown(f"**Matches:** {len(matches)}")
        st.markdown(f"**Seasons:** {total_seasons}")
        st.markdown(f"**Algorithm:** XGBoost")
        st.markdown("---")
        st.markdown("### Top Factors")
        for feat, imp in feature_importance.head(5).items():
            st.markdown(f"- {feat.replace('_',' ').title()}: **{imp:.1%}**")
        st.markdown("---")
        st.markdown("*Built with Cricsheet open data*")


if __name__ == '__main__':
    main()
