import data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n📈 Model Performance")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Feature Importance (Top 5):")
    for i, (feat, imp) in enumerate(list(zip(X.columns, model.feature_importances_))[-5:]):
        print(f"  {feat}: {imp:.3f}")
    
    return model, scaler

def predict_canada_vs_italy(model, scaler, canada_stats, italy_stats, features):
    # Create feature vector (Canada home, Italy away)
    
    # Get Elo ratings (default to 1500 if missing)
    canada_elo = canada_stats.get('current_elo', 1500)
    italy_elo = italy_stats.get('current_elo', 1500)
    
    feature_data = {
        # Elo Ratings
        'home_elo': canada_elo,
        'away_elo': italy_elo,
        'elo_diff': canada_elo - italy_elo,
        
        # Host Advantage (Canada is host in 2026)
        'is_home_host': 1,
        'is_away_host': 0,
        
        # Home team stats
        'gf_avg_home': canada_stats['gf_avg'],
        'ga_avg_home': canada_stats['ga_avg'], 
        'win_rate_home': canada_stats['win_rate'],
        'gd_avg_home': canada_stats['gd_avg'],
        'gf_per_match_home': canada_stats['gf_per_match'],
        
        # Away team stats
        'gf_avg_away': italy_stats['gf_avg'],
        'ga_avg_away': italy_stats['ga_avg'],
        'win_rate_away': italy_stats['win_rate'],
        'gd_avg_away': italy_stats['gd_avg'],
        'gf_per_match_away': italy_stats['gf_per_match']
    }
    
    feature_data['win_rate_home'] = canada_stats['win_rate']
    feature_data['win_rate_away'] = italy_stats['win_rate']
    feature_data['gf_avg_home'] = canada_stats['gf_avg']
    feature_data['gf_avg_away'] = italy_stats['gf_avg']
    feature_data['ga_avg_home'] = canada_stats['ga_avg']
    feature_data['ga_avg_away'] = italy_stats['ga_avg']
    
    X_pred = pd.DataFrame([feature_data])[features].fillna(0)
    X_pred_scaled = scaler.transform(X_pred)
    
    prediction = model.predict(X_pred_scaled)[0]
    probabilities = model.predict_proba(X_pred_scaled)[0]
    
    outcomes = {0: '🇮🇹 Italy Wins', 1: '🤝 Draw', 2: '🇨🇦 Canada Wins'}
    
    print(f"\n🎯 CANADA vs ITALY PREDICTION (June 12, 2026)")
    print("=" * 50)
    print(f"Match Context: Canada (Home/Host) vs Italy (Away)")
    print(f"Elo Ratings: Canada {canada_elo:.0f} vs Italy {italy_elo:.0f}")
    print("-" * 50)
    print(f"Predicted Result: {outcomes[prediction]}")
    print(f"Win Probabilities:")
    print(f"  🇨🇦 Canada: {probabilities[2]:.1%}")
    print(f"  🤝 Draw:    {probabilities[1]:.1%}") 
    print(f"  🇮🇹 Italy:   {probabilities[0]:.1%}")
    
    return prediction, probabilities

if __name__ == "__main__":
    print("🏆 World Cup Match Prediction Model - Canada vs Italy")
    print("=" * 60)

    print("📥 Step 1: Loading World Cup data from local 'data-csv' folder...")
    data_dict = data.load_worldcup_data()
    
    if not data_dict:
        print("❌ No data loaded. Exiting.")
        exit()

    print("\n🏗️  Step 2: Creating features from historical data...")
    matches, team_stats = data.create_team_strength_features(data_dict['matches'], data_dict['team_appearances'])
    X, y, features = data.prepare_match_features(matches, team_stats)
    
    print(f"\n🤖 Step 3: Training model on {len(X)} historical matches...")
    model, scaler = train_model(X, y)
    
    teams_df = data_dict['teams']
    canada_info = teams_df[teams_df['team_name'] == 'Canada']
    italy_info = teams_df[teams_df['team_name'] == 'Italy']
    
    if not canada_info.empty and not italy_info.empty:
        canada_code = canada_info.iloc[0]['team_code']
        italy_code = italy_info.iloc[0]['team_code']
        
        print(f"\nFound Team Codes: Canada={canada_code}, Italy={italy_code}")
        
        if canada_code in team_stats.index and italy_code in team_stats.index:
            canada_stats = team_stats.loc[canada_code]
            italy_stats = team_stats.loc[italy_code]
            
            predict_canada_vs_italy(model, scaler, canada_stats, italy_stats, features)
        else:
            print("❌ Could not find stats for Canada or Italy in the processed team stats.")
    else:
        print("❌ Could not find 'Canada' or 'Italy' in the teams dataset.")
