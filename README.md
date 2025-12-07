# Canada vs Italy Prediction Model

This project is a machine learning implementation designed to predict the outcome of the upcoming World Cup 2026 match between **Canada** and **Italy**.

Using historical World Cup data, the model analyzes team performance, goal statistics, and calculates Elo ratings to determine win probabilities for both sides.

## 📊 Implementation

The prediction engine relies on a **Random Forest Classifier** trained on historical match data. Key features include:

*   **Elo Ratings**: Dynamic rating system updated after every historical match to gauge current team strength.
*   **Team Statistics**: Average goals for/against, win rates, and goal differentials derived from historical appearances.
*   **Host Advantage**: Considers the impact of playing on home soil (Canada is a host nation for 2026).
*   **Rolling Form**: Analyzing the most recent performance trends.

## 📂 Project Structure

*   `model.py`: The main entry point. Trains the model and outputs the specific prediction for Canada vs. Italy.
*   `data.py`: Handles data loading, feature engineering, and Elo rating calculations.
*   `data-csv/`: Directory containing historical World Cup data (matches, players, teams, etc.).
*   `requirements.txt`: Python dependencies.

## 🚀 Getting Started

### Prerequisites

*   Python 3.8+
*   `pip` package manager

### Installation

1.  **Clone the repository:**
    ```shell
    git clone https://github.com/JummyJoeJackson/canada-vs-italy-prediction-model.git
    cd canada-vs-italy-prediction-model
    ```

2.  **Create and activate a virtual environment (optional but recommended):**
    ```shell
    # Windows
    python -m venv .venv
    .venv\Scripts\activate

    # macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```shell
    pip install -r requirements.txt
    ```

## 🏃 Usage

Run the main model script to train the classifier and see the prediction:

```shell
python model.py
```

### Example Output

```text
🏆 World Cup Match Prediction Model - Canada vs Italy
============================================================
...
📈 Model Performance
Accuracy: 0.585
Feature Importance (Top 5):
  elo_diff: 0.142
  away_elo: 0.115
  home_elo: 0.108
...

🎯 CANADA vs ITALY PREDICTION (June 12, 2026)
==================================================
Match Context: Canada (Home/Host) vs Italy (Away)
Elo Ratings: Canada 1850 vs Italy 1980
--------------------------------------------------
Predicted Result: 🇮🇹 Italy Wins
Win Probabilities:
  🇨🇦 Canada: 25.4%
  🤝 Draw:    28.1%
  🇮🇹 Italy:   46.5%
```

## 📝 Data Source
The data used in this project includes historical records of World Cup matches, team appearances, and tournament results.

## 📄 License
This project is open-source.
