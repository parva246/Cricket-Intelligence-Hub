# Cricket Intelligence Hub — Project Pitch

---

## The Problem

Cricket is the **2nd most popular sport globally** (2.5B+ fans), and IPL is the **richest cricket league** in the world ($16B+ valuation). Yet:

- **Fantasy cricket players** make picks based on gut feeling, not data
- **Cricket analysts** rely on basic stats (averages, strike rates) without contextual intelligence
- **Broadcasters & media** need real-time predictive content to engage viewers
- **Betting/fantasy platforms** lack transparent, explainable prediction models

There is no accessible, affordable, ML-powered cricket intelligence platform for the Indian market.

---

## The Solution

**Cricket Intelligence Hub** is an AI-powered cricket analytics platform that uses machine learning on 17+ years of ball-by-ball IPL data to deliver:

### Core Features (Built & Working)

| Feature | What It Does | Who Benefits |
|---|---|---|
| **Match Prediction** | XGBoost ML model predicts match winners with probability % | Fantasy players, analysts |
| **Phase Analysis** | Compares teams across Powerplay, Middle, Death overs | Coaches, strategists |
| **Player Stats Engine** | Full batting/bowling career breakdowns | Fantasy players, scouts |
| **Head-to-Head Matchups** | Batsman vs Bowler historical performance | Commentators, analysts |
| **Playing XI Analyzer** | Select 11 per side, get squad-level insights | Team management, fantasy |
| **Venue Intelligence** | Chase vs defend win rates per ground | Captains, analysts |

### Planned Features (Roadmap)

| Feature | Description | Revenue Potential |
|---|---|---|
| **Live Win Probability** | Ball-by-ball prediction updates during matches | Premium subscription |
| **Fantasy Points Predictor** | Predict Dream11/fantasy scores for each player | High — 180M+ fantasy users in India |
| **Player Form Index** | Weighted recent performance scoring | Subscription / API |
| **API for Third Parties** | Sell predictions as API to apps, media, platforms | B2B revenue |
| **Mobile App** | React Native app for on-the-go insights | Consumer market |

---

## Technology Architecture

### Current Stack

```
┌─────────────────────────────────────────┐
│           STREAMLIT FRONTEND            │
│  Interactive UI, Charts, Dropdowns      │
├─────────────────────────────────────────┤
│           PREDICTION ENGINE             │
│  XGBoost ML Model (20 features)         │
│  Trained on 1,100+ matches              │
├─────────────────────────────────────────┤
│         FEATURE ENGINEERING             │
│  Team stats, venue stats, phase stats   │
│  Player stats, head-to-head matchups    │
├─────────────────────────────────────────┤
│            DATA PIPELINE                │
│  Cricsheet.org → CSV processing         │
│  200,000+ ball-by-ball records          │
└─────────────────────────────────────────┘
```

### Technology Details

| Component | Technology | Why This Choice |
|---|---|---|
| **ML Model** | XGBoost (Gradient Boosting) | Industry standard for tabular data, interpretable, fast inference |
| **Web Framework** | Streamlit | Rapid prototyping, built-in caching, easy deployment |
| **Data Processing** | Pandas + NumPy | Python data science standard, handles 200K+ rows efficiently |
| **Visualization** | Plotly | Interactive charts, mobile-responsive, professional quality |
| **Model Evaluation** | scikit-learn | Train/test split, accuracy metrics, cross-validation |
| **Data Source** | Cricsheet.org | Free, open-source, ball-by-ball granularity, updated each season |

### How the ML Model Works

1. **Data Collection** — 17 years of IPL ball-by-ball data (every ball bowled since 2008)
2. **Feature Engineering** — 20 features calculated per match:
   - Team form (recent win %), head-to-head record
   - Batting & bowling stats (run rate, economy, wickets per match)
   - Phase performance (powerplay, middle, death overs)
   - Venue factors (chase rate, team's venue win rate)
   - Toss impact (who won, what they chose)
3. **Model Training** — XGBoost classifier on 80% of matches (chronological split, no data leakage)
4. **Prediction** — Outputs win probability for each team with top contributing factors

---

## Market Opportunity

### IPL & Cricket Market

- **IPL brand value:** $16.4 billion (2024)
- **IPL viewership:** 600M+ per season
- **Fantasy cricket users in India:** 180M+ (Dream11, MPL, etc.)
- **Cricket betting market (legal):** $150B+ globally
- **Sports analytics market:** $3.4B globally, growing 30% CAGR

### Target Users

| Segment | Size | Willingness to Pay |
|---|---|---|
| Fantasy cricket players | 180M+ in India | High — already spending on tips/tools |
| Cricket content creators | 50K+ YouTube/Instagram | Medium — need data for content |
| Media & broadcasters | Star Sports, JioCinema, etc. | High — need predictive content |
| Team analysts & scouts | 10 IPL teams + domestic | High — competitive advantage |
| Betting platforms (legal markets) | Global operators | Very High — prediction APIs |

---

## Revenue Model

### Phase 1: Freemium (Year 1)
- Free: Basic match predictions, team stats
- Premium ($5-10/month): Player matchups, Playing XI analysis, phase breakdowns
- Target: 10,000 premium users = $50K-100K ARR

### Phase 2: API & B2B (Year 2)
- Prediction API for fantasy apps, media companies
- White-label analytics dashboard for cricket teams
- Target: 5-10 B2B clients = $200K-500K ARR

### Phase 3: Live & Expanded (Year 3)
- Live match win probability (requires real-time data feed)
- Expand beyond IPL: BBL, PSL, T20 World Cup, ODIs
- Mobile app with push notifications
- Target: $1M+ ARR

---

## Competitive Advantage

| Us | Competitors |
|---|---|
| **Open data** — no expensive API lock-in | Dependent on $29-75/month paid APIs |
| **ML-based predictions** with explainable features | Most use simple stat comparisons |
| **Ball-by-ball granularity** — phase & matchup analysis | Match-level stats only |
| **Free to start** — low cost to validate & iterate | High infrastructure costs from day 1 |
| **Modular architecture** — easy to add features | Monolithic, hard to extend |

---

## Current Status

- **Working MVP** deployed on Streamlit Cloud
- **1,100+ matches** processed (2008–2025)
- **200,000+ deliveries** in the database
- **6 core features** built and functional
- **Model accuracy:** 54-58% (above baseline of 50%)
- **Data pipeline:** Automated Cricsheet → app conversion
- **Cost to run:** $0/month (free Streamlit hosting + free data)

---

## What We Need

### Immediate (Seed / Pre-Seed)

| Need | Amount | Purpose |
|---|---|---|
| Real-time data API | $350-900/year | SportMonks or similar for live data |
| Cloud infrastructure | $50-100/month | Dedicated hosting for production |
| Domain & branding | $200 one-time | Professional domain, logo |
| **Total Year 1** | **~$2,000-3,000** | |

### Growth (With Funding)

| Need | Amount | Purpose |
|---|---|---|
| Full-time development | $30-50K/year | Build mobile app, live features, API |
| Marketing | $10-20K/year | User acquisition, content marketing |
| Data infrastructure | $5-10K/year | Database, caching, scaling |
| **Total with funding** | **$50-80K/year** | |

---

## Team

- **Founder** — IT professional with data science & ML skills, deep cricket domain knowledge
- Building with modern AI tools (Claude, GitHub Copilot) for rapid development
- Looking for: technical co-founder (ML/backend) and business co-founder (growth/partnerships)

---

## Demo

**Live app:** [Streamlit Cloud link — add after deployment]

**GitHub:** [Repository link — add after creation]

---

*Cricket Intelligence Hub — Making cricket smarter, one ball at a time.*
