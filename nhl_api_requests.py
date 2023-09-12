import requests
from datetime import datetime
import pandas as pd

# Define the NHL API endpoint for the schedule endpoint
api_url = "https://statsapi.web.nhl.com/api/v1/schedule"

# Define a list of seasons you want to retrieve data for
start_season = 20212022
current_year = datetime.now().year
current_month = currentMonth = datetime.now().month

if current_month >= 8:
    end_season = int(str(current_year) + str(current_year + 1))
else:
    end_season = int(str(current_year - 1) + str(current_year))

seasons = []
for i in range(start_season, end_season + 10001, 10001):
    seasons.append(str(i))

# Initialize an empty list to store game data
all_games = []

# Loop through each season and retrieve game data
for season in seasons:
    # Define parameters for the API request for the current season
    params = {
        "hydrate": "team,linescore,game(content(media(epg))),broadcasts(all)",
        "site": "en_nhl",
        "season": season,
    }

    # Send a GET request to the NHL API
    response = requests.get(api_url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()

        # Extract game data from the response
        games = []
        for date in data["dates"]:
            for game in date["games"]:
                games.append(game)

        # Add a "Season" column to the game data
        for game in games:
            game["Season"] = season

        # Extend the list of all games with games from the current season
        all_games.extend(games)
    else:
        print(f"Failed to retrieve data for season {season}. Status code:", response.status_code)

# Create a Pandas DataFrame from all the game data
df = pd.DataFrame(all_games)

# Extract Meaninful values of columns and ensure data types are adjusted properly:
df['game_id'] = df['gamePk'].astype(str)
df.rename(columns = {'Season' : 'season', 'gameType' : 'type', 'gameDate' : 'date_time_GMT'}, inplace = True)

def extract_away_team_id(row):
    return row['away']['team']['id']

def extract_home_team_id(row):
    return row['home']['team']['id']

def extract_away_goals(row):
    return row['away']['score']

def extract_home_goals(row):
    return row['home']['score']

def extract_game_status(row):
    return row['detailedState']

# Apply the extraction functions to create new columns
df['away_team_id'] = df['teams'].apply(extract_away_team_id)
df['home_team_id'] = df['teams'].apply(extract_home_team_id)
df['away_goals'] = df['teams'].apply(extract_away_goals)
df['home_goals'] = df['teams'].apply(extract_home_goals)
df['game_status'] = df['status'].apply(extract_game_status)

# Calculate the outcome based on goals
df['outcome'] = df.apply(lambda row: 'Home Win' if row['home_goals'] > row['away_goals'] else 'Away Win' if row['away_goals'] > row['home_goals'] else 'Tie', axis=1)

df_game = df[['game_id', 'season', 'type', 'date_time_GMT', 'away_team_id', 'home_team_id', 'away_goals', 'home_goals', 'outcome', 'game_status']]

df_game[['away_goals', 'home_goals']] = df_game[['away_goals', 'home_goals']].astype(int)

