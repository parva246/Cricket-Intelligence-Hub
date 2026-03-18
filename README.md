# Cricket Intelligence Hub

**AI-powered IPL match prediction platform using machine learning and 17+ years of ball-by-ball cricket data.**

Built for cricket fans, analysts, and fantasy league players who want data-driven insights — not just gut feeling.

---

## What It Does

- **Match Winner Prediction** — Predicts match outcomes using XGBoost ML trained on every IPL match ever played (2008–2025)
- **Phase-by-Phase Breakdown** — Compares teams across Powerplay (1-6), Middle (7-16), and Death (17-20) overs
- **Player Stats Engine** — Full batting and bowling career stats for any player in the current IPL squads
- **Head-to-Head Matchups** — How does Virat Kohli perform against Jasprit Bumrah? Get exact numbers.
- **Playing XI Analysis** — Select 11 players per side and get squad-level insights before the match starts
- **Venue Intelligence** — Chase vs defend win rates for every IPL ground

---

## Tech Stack

| Technology | Purpose |
|---|---|
| **Python 3.10+** | Core language |
| **Streamlit** | Interactive web application framework |
| **XGBoost** | Gradient boosting ML model for match prediction |
| **Pandas / NumPy** | Data processing and feature engineering |
| **Plotly** | Interactive charts and visualizations |
| **scikit-learn** | Model training, evaluation, train/test split |

---

## Data Source

All data comes from [Cricsheet.org](https://cricsheet.org) — a free, open-source repository of ball-by-ball cricket data.

- **1,100+ IPL matches** (2008–2025)
- **200,000+ ball-by-ball deliveries**
- **Updated each season** — no paid API required

---

## How the ML Model Works

### Features Used (20 input features)

| Feature | Description |
|---|---|
| `team1_win_pct` / `team2_win_pct` | Recent win percentage (last 10 matches) |
| `head_to_head` | Historical head-to-head record between the two teams |
| `toss_win` | Whether Team A won the toss |
| `chose_bat` | Whether the toss winner chose to bat |
| `venue_win_rate` | Team's win rate at the specific venue |
| `chase_rate_venue` | How often chasing teams win at this venue |
| `team1_run_rate` / `team2_run_rate` | Average run rate (last 10 matches) |
| `team1_avg_score` / `team2_avg_score` | Average total score (last 10 matches) |
| `team1_bowl_economy` / `team2_bowl_economy` | Bowling economy rate |
| `team1_wickets_pm` / `team2_wickets_pm` | Average wickets taken per match |
| `team1_powerplay` / `team2_powerplay` | Average powerplay runs |
| `team1_death` / `team2_death` | Average death overs runs |
| `team1_middle` / `team2_middle` | Average middle overs runs |

### Model Configuration

```python
XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8
)
```

- **Algorithm:** XGBoost (Extreme Gradient Boosting)
- **Train/Test Split:** 80/20, chronological (no data leakage)
- **Current Accuracy:** ~54-58% (on unseen matches)

---

## Project Structure

```
Cricket-Intelligence-Hub/
├── app/
│   └── streamlit_app.py       # Main Streamlit application
├── scripts/
│   └── data_pipeline.py       # Cricsheet → CSV converter (run on Colab)
├── squads/
│   └── squads_data.py         # IPL 2026 squad data + name mappings
├── data/
│   ├── raw/                   # Raw Cricsheet zip (gitignored)
│   └── processed/             # matches.csv + deliveries.csv
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup & Run

### 1. Generate Data (Google Colab)

```
- Open scripts/data_pipeline.py in Google Colab
- Upload ipl_male_csv2.zip from Cricsheet.org
- Download the generated matches.csv and deliveries.csv
- Place them in data/processed/
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run app/streamlit_app.py
```

---

## Updating for a New Season

1. Download latest data from [Cricsheet.org](https://cricsheet.org/downloads/ipl_male_csv2.zip)
2. Re-run `scripts/data_pipeline.py` on Colab
3. Replace `data/processed/matches.csv` and `deliveries.csv`
4. Update `squads/squads_data.py` with new team rosters
5. Redeploy

---

## Roadmap

- [ ] Live match data integration
- [ ] Win probability during live innings (ball-by-ball)
- [ ] Fantasy points predictor
- [ ] Player form index (weighted recent performance)
- [ ] Injury impact analysis
- [ ] API endpoint for third-party apps
- [ ] Mobile-responsive redesign

---

## License

This project uses open data from Cricsheet.org. For commercial use, please review their [terms](https://cricsheet.org/licence/).

---

*Built with Python, XGBoost, and Cricsheet open data.*
