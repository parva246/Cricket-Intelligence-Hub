"""
Cricsheet CSV2 → matches.csv + deliveries.csv converter
========================================================
Run this on Google Colab:
  1. Upload ipl_male_csv2.zip when prompted
  2. Script produces matches.csv and deliveries.csv
  3. Download both files

These CSVs are in the exact format the Streamlit app expects.
"""

import zipfile
import csv
import os
import io
import pandas as pd
from google.colab import files

# ── Step 1: Upload the Cricsheet zip ──
print("Upload ipl_male_csv2.zip ...")
uploaded = files.upload()
zip_filename = list(uploaded.keys())[0]
print(f"Uploaded: {zip_filename}")

# ── Step 2: Extract and parse all match CSVs ──
matches_list = []
deliveries_list = []
match_id_counter = 1000001  # Start IDs from here

with zipfile.ZipFile(zip_filename, 'r') as zf:
    csv_files = sorted([f for f in zf.namelist() if f.endswith('.csv') and not f.startswith('__')])
    print(f"Found {len(csv_files)} match files")

    for file_idx, csv_file in enumerate(csv_files):
        if file_idx % 100 == 0:
            print(f"Processing {file_idx}/{len(csv_files)}...")

        with zf.open(csv_file) as f:
            content = f.read().decode('utf-8')
            reader = csv.reader(io.StringIO(content))
            rows = list(reader)

        # ── Parse info rows ──
        info = {}
        teams = []
        players = {}  # team -> list of players
        for row in rows:
            if len(row) < 2:
                continue
            if row[0] == 'info':
                key = row[1]
                value = row[2] if len(row) > 2 else ''
                if key == 'team':
                    teams.append(value)
                elif key == 'player':
                    team_name = value
                    player_name = row[3] if len(row) > 3 else ''
                    if team_name not in players:
                        players[team_name] = []
                    players[team_name].append(player_name)
                elif key == 'outcome' and value == 'no result':
                    info['result'] = 'no result'
                elif key == 'outcome' and value == 'winner':
                    info['winner'] = row[3] if len(row) > 3 else ''
                elif key == 'outcome' and value == 'result':
                    info['result_type'] = row[3] if len(row) > 3 else ''
                elif key == 'outcome' and value == 'by':
                    # e.g., info,outcome,by,runs,48 or info,outcome,by,wickets,5
                    if len(row) >= 5:
                        info['result_method'] = row[3]  # runs or wickets
                        info['result_margin'] = row[4]
                elif key == 'supersub':
                    continue
                else:
                    info[key] = value

        if len(teams) < 2:
            continue

        # Skip non-result matches
        if info.get('result') == 'no result':
            continue

        match_id = match_id_counter
        match_id_counter += 1

        # Build match record
        season = info.get('season', '')
        date = info.get('date', '')
        city = info.get('city', '')
        venue = info.get('venue', '')
        team1 = teams[0]
        team2 = teams[1]
        toss_winner = info.get('toss_winner', '')
        toss_decision = info.get('toss_decision', '')
        winner = info.get('winner', '')
        player_of_match = info.get('player_of_match', '')
        match_type = info.get('match_type', 'League')
        result_method = info.get('result_method', '')
        result_margin = info.get('result_margin', '')
        method = info.get('method', '')
        super_over = 'Y' if info.get('super_over', '') == 'True' else 'N'
        umpire1 = info.get('umpire', '')

        # Determine result type
        if result_method == 'runs':
            result = 'runs'
        elif result_method == 'wickets':
            result = 'wickets'
        elif result_method == 'innings':
            result = 'innings'
        else:
            result = 'normal'

        # Calculate target_runs and target_overs (will be filled from ball data)
        target_runs = ''
        target_overs = ''

        if not winner:
            continue

        match_record = {
            'id': match_id,
            'season': season,
            'city': city,
            'date': date,
            'match_type': match_type,
            'player_of_match': player_of_match,
            'venue': venue,
            'team1': team1,
            'team2': team2,
            'toss_winner': toss_winner,
            'toss_decision': toss_decision,
            'winner': winner,
            'result': result,
            'result_margin': result_margin,
            'target_runs': target_runs,
            'target_overs': target_overs,
            'super_over': super_over,
            'method': method,
            'umpire1': umpire1,
            'umpire2': ''
        }

        # ── Parse ball/delivery rows ──
        innings_runs = {}  # track runs per innings for target calculation
        for row in rows:
            if len(row) < 2 or row[0] != 'ball':
                continue

            # Cricsheet CSV2 ball format:
            # ball, innings, over.ball, batting_team, striker, non_striker, bowler,
            # runs_off_bat, extras, wides, noballs, byes, legbyes, penalty,
            # wicket_type, player_dismissed, other_wicket_type, other_player_dismissed
            innings_num = int(row[1]) if len(row) > 1 else 1
            over_ball = row[2] if len(row) > 2 else '0.1'
            batting_team = row[3] if len(row) > 3 else ''
            striker = row[4] if len(row) > 4 else ''
            non_striker = row[5] if len(row) > 5 else ''
            bowler = row[6] if len(row) > 6 else ''
            runs_off_bat = int(row[7]) if len(row) > 7 and row[7] else 0
            extras = int(row[8]) if len(row) > 8 and row[8] else 0
            wides = int(row[9]) if len(row) > 9 and row[9] else 0
            noballs = int(row[10]) if len(row) > 10 and row[10] else 0
            byes = int(row[11]) if len(row) > 11 and row[11] else 0
            legbyes = int(row[12]) if len(row) > 12 and row[12] else 0
            penalty = int(row[13]) if len(row) > 13 and row[13] else 0
            wicket_type = row[14] if len(row) > 14 else ''
            player_dismissed = row[15] if len(row) > 15 else ''

            # Parse over and ball number
            over_parts = over_ball.split('.')
            over_num = int(over_parts[0]) if over_parts[0] else 0
            ball_num = int(over_parts[1]) if len(over_parts) > 1 and over_parts[1] else 1

            # Determine bowling team
            if batting_team == team1:
                bowling_team = team2
            else:
                bowling_team = team1

            total_runs = runs_off_bat + extras

            # Track innings total for target calculation
            if innings_num not in innings_runs:
                innings_runs[innings_num] = 0
            innings_runs[innings_num] += total_runs

            # Determine extras type
            extras_type = ''
            if wides > 0:
                extras_type = 'wides'
            elif noballs > 0:
                extras_type = 'noballs'
            elif byes > 0:
                extras_type = 'byes'
            elif legbyes > 0:
                extras_type = 'legbyes'
            elif penalty > 0:
                extras_type = 'penalty'

            is_wicket = 1 if wicket_type else 0
            dismissal_kind = wicket_type if wicket_type else 'NA'
            dismissed_player = player_dismissed if player_dismissed else 'NA'

            # Map wide_runs and other extras for compatibility
            wide_runs = wides
            noball_runs = noballs
            bye_runs = byes
            legbye_runs = legbyes
            extra_runs = extras

            delivery_record = {
                'match_id': match_id,
                'inning': innings_num,
                'batting_team': batting_team,
                'bowling_team': bowling_team,
                'over': over_num,
                'ball': ball_num,
                'batter': striker,
                'bowler': bowler,
                'non_striker': non_striker,
                'batsman_runs': runs_off_bat,
                'extra_runs': extra_runs,
                'total_runs': total_runs,
                'extras_type': extras_type,
                'is_wicket': is_wicket,
                'player_dismissed': dismissed_player,
                'dismissal_kind': dismissal_kind,
                'fielder': 'NA',
                'wide_runs': wide_runs,
            }
            deliveries_list.append(delivery_record)

        # Set target runs (1st innings total + 1)
        if 1 in innings_runs:
            match_record['target_runs'] = innings_runs[1] + 1
            match_record['target_overs'] = 20

        matches_list.append(match_record)

print(f"\nParsed {len(matches_list)} matches and {len(deliveries_list)} deliveries")

# ── Step 3: Create DataFrames and save ──
matches_df = pd.DataFrame(matches_list)
deliveries_df = pd.DataFrame(deliveries_list)

# Sort matches by date
matches_df['date'] = pd.to_datetime(matches_df['date'])
matches_df = matches_df.sort_values('date').reset_index(drop=True)

print(f"\nMatches: {len(matches_df)}")
print(f"Date range: {matches_df['date'].min()} to {matches_df['date'].max()}")
print(f"Seasons: {sorted(matches_df['season'].unique())}")
print(f"\nDeliveries: {len(deliveries_df)}")

# Show team names in data (important for mapping)
print(f"\nTeams found:")
all_teams = set(matches_df['team1'].unique()) | set(matches_df['team2'].unique())
for t in sorted(all_teams):
    count = len(matches_df[(matches_df['team1'] == t) | (matches_df['team2'] == t)])
    print(f"  {t}: {count} matches")

print(f"\nVenues found: {matches_df['venue'].nunique()}")

# Save CSVs
matches_df.to_csv('matches.csv', index=False)
deliveries_df.to_csv('deliveries.csv', index=False)
print("\nSaved matches.csv and deliveries.csv")

# ── Step 4: Download ──
print("\nDownloading files...")
files.download('matches.csv')
files.download('deliveries.csv')
print("Done! Upload these to your GitHub repo under data/processed/")
