import os
import pandas as pd
import numpy as np


def load_worldcup_data():
    base_path = os.path.join(os.path.dirname(__file__), 'data-csv')
    
    datasets = {
        'matches': 'matches.csv',
        'team_appearances': 'team_appearances.csv',
        'teams': 'teams.csv',
        'qualified_teams': 'qualified_teams.csv'
    }
    
    data = {}
    for name, filename in datasets.items():
        file_path = os.path.join(base_path, filename)
        try:
            df = pd.read_csv(file_path)
            data[name] = df
            print(f"✅ Loaded {name}: {df.shape}")
        except Exception as e:
            print(f"❌ Failed to load {filename} from {file_path}. Error: {e}")
    
    return data


def calculate_elo_ratings(matches):
    # Initialize ratings
    elo_ratings = {}
    k_factor = 32
    
    # Sort by date
    matches = matches.sort_values('match_date')
    
    # Arrays to store elo ratings for each match
    home_elos = []
    away_elos = []
    
    for idx, row in matches.iterrows():
        home_team = row['home_team_code']
        away_team = row['away_team_code']
        
        # Get current ratings (default 1500)
        home_rating = elo_ratings.get(home_team, 1500)
        away_rating = elo_ratings.get(away_team, 1500)
        
        home_elos.append(home_rating)
        away_elos.append(away_rating)
        
        # Calculate expected score
        # Expected score = 1 / (1 + 10^((rating_B - rating_A) / 400))
        expected_home = 1 / (1 + 10 ** ((away_rating - home_rating) / 400))
        expected_away = 1 / (1 + 10 ** ((home_rating - away_rating) / 400))
        
        # Actual score (1=win, 0.5=draw, 0=loss)
        if row['home_team_score'] > row['away_team_score']:
            actual_home = 1
            actual_away = 0
        elif row['home_team_score'] == row['away_team_score']:
            actual_home = 0.5
            actual_away = 0.5
        else:
            actual_home = 0
            actual_away = 1
            
        # Update ratings
        new_home_rating = home_rating + k_factor * (actual_home - expected_home)
        new_away_rating = away_rating + k_factor * (actual_away - expected_away)
        
        elo_ratings[home_team] = new_home_rating
        elo_ratings[away_team] = new_away_rating
        
    matches['home_elo'] = home_elos
    matches['away_elo'] = away_elos
    
    return matches, elo_ratings


def create_team_strength_features(matches, team_appearances):
    # Create match outcomes (0=home loss, 1=draw, 2=home win)
    matches = matches.dropna(subset=['home_team_code', 'away_team_code', 'home_team_score', 'away_team_score'])
    matches['outcome'] = np.where(matches['home_team_score'] > matches['away_team_score'], 2,
                                 np.where(matches['home_team_score'] == matches['away_team_score'], 1, 0))
    
    # Add Host Country Features
    matches['is_home_host'] = (matches['home_team_name'] == matches['country_name']).astype(int)
    matches['is_away_host'] = (matches['away_team_name'] == matches['country_name']).astype(int)
    
    # Calculate Elo Ratings
    matches, current_elo_ratings = calculate_elo_ratings(matches)
    
    # Team performance stats from appearances
    team_stats = team_appearances.groupby('team_code').agg({
        'goals_for': ['mean', 'sum', 'std'],
        'goals_against': ['mean', 'sum', 'std'],
        'win': 'sum',
        'draw': 'sum', 
        'lose': 'sum',
        'match_id': 'count',
        'goal_differential': 'mean'
    }).round(3)
    
    team_stats.columns = ['gf_avg', 'gf_total', 'gf_std', 
                         'ga_avg', 'ga_total', 'ga_std',
                         'wins_total', 'draws_total', 'losses_total', 
                         'matches_total', 'gd_avg']
    team_stats['win_rate'] = team_stats['wins_total'] / team_stats['matches_total']
    team_stats['gf_per_match'] = team_stats['gf_total'] / team_stats['matches_total']
    team_stats['ga_per_match'] = team_stats['ga_total'] / team_stats['matches_total']
    
    # Add current Elo rating to team stats
    team_stats['current_elo'] = team_stats.index.map(current_elo_ratings).fillna(1500)
    
    # Rolling form (last 5 matches performance)
    appearances_sorted = team_appearances.sort_values(['team_code', 'match_date'])
    appearances_sorted['recent_form'] = appearances_sorted.groupby('team_code')['goal_differential'].transform(
        lambda x: x.rolling(5, min_periods=1).mean().shift(1))
    
    recent_form = appearances_sorted.groupby('team_code')['recent_form'].mean()
    team_stats = team_stats.join(recent_form)
    
    return matches, team_stats


def prepare_match_features(matches, team_stats):
    # Merge team stats
    matches = matches.merge(team_stats, left_on='home_team_code', right_index=True)
    matches = matches.merge(team_stats, left_on='away_team_code', right_index=True, suffixes=('_home', '_away'))
    
    # Calculate Elo difference
    matches['elo_diff'] = matches['home_elo'] - matches['away_elo']
    
    # Key features for prediction
    features = [
        # Elo Ratings
        'home_elo', 'away_elo', 'elo_diff',
        # Host Advantage
        'is_home_host', 'is_away_host',
        # Home team strength
        'gf_avg_home', 'ga_avg_home', 'win_rate_home', 'gd_avg_home', 'gf_per_match_home',
        # Away team strength  
        'gf_avg_away', 'ga_avg_away', 'win_rate_away', 'gd_avg_away', 'gf_per_match_away',
        # Relative strength
        'win_rate_home', 'win_rate_away',
        'gf_avg_home', 'gf_avg_away',
        'ga_avg_home', 'ga_avg_away'
    ]
    
    # Clean features
    X = matches[features].fillna(0)
    y = matches['outcome']
    
    return X, y, features