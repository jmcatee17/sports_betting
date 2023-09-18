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

# Define the NHL API endpoint for all NHL teams
api_url = "https://statsapi.web.nhl.com/api/v1/teams"

# Send a GET request to the NHL API to retrieve data about all NHL teams
response = requests.get(api_url)

# Check if the request was successful
if response.status_code == 200:
    data = response.json()

    # Extract the team IDs from the response
    team_ids = [team["id"] for team in data["teams"]]

    # Display the list of team IDs
    print("Most Recent Team IDs:", team_ids)

else:
    print("Failed to retrieve data. Status code:", response.status_code)

# Create an empty list to store skater data
skater_data = []
    
# Iterate through each season
for season in seasons:
    # Iterate through each team ID
    for team_id in team_ids:
        # Define the NHL API endpoint for a specific team's game roster
        api_url = f"https://statsapi.web.nhl.com/api/v1/teams/{team_id}?expand=team.roster"

        # Send a GET request to the NHL API to retrieve the team's roster
        response = requests.get(api_url)

        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()

            # Extract the team's roster
            roster = data["teams"][0]["roster"]["roster"]

            # Iterate through each player on the roster
            for player in roster:
                player_id = player["person"]["id"]
                player_name = player["person"]["fullName"]

                # Define the NHL API endpoint for player game logs with the season parameter
                player_game_url = f"https://statsapi.web.nhl.com/api/v1/people/{player_id}/stats?stats=gameLog&season={season}"

                # Send a GET request to the NHL API to retrieve player game logs
                response = requests.get(player_game_url)

                # Check if the request was successful
                if response.status_code == 200:
                    player_game_data = response.json()


                # Extract skater statistics for each game
                for game_stats in player_game_data["stats"][0]["splits"]:
                    game_data = {
                        "team_id": team_id,
                        "team_name": data["teams"][0]["name"],
                        "opponent": game_stats["opponent"]["name"],
                        "player_id": player_id,
                        "player_name": player_name,
                        "game_id": game_stats["game"]["gamePk"],
                        "season": game_stats["season"],
                        "goals": game_stats["stat"].get("goals"),
                        "assists": game_stats["stat"].get("assists"),
                        "shots": game_stats["stat"].get("shots"),
                        "hits": game_stats["stat"].get("hits"),
                        "powerPlayGoals": game_stats["stat"].get("powerPlayGoals"),
                        "powerPlayAssists": game_stats["stat"].get("powerPlayAssists"),
                        "time_on_ice": game_stats["stat"]["timeOnIce"],
                        # Add more statistics as needed
                    }
                    skater_data.append(game_data)

    else:
        print(f"Failed to retrieve data for team ID {team_id}. Status code:", response.status_code)

# Create a Pandas DataFrame from the skater data
df_game_skater_stats = pd.DataFrame(skater_data)

# Define the NHL API endpoints
NHL_API_URL = "https://statsapi.web.nhl.com/api/v1/"
SCHEDULE_ENDPOINT = "schedule"
GAME_STATS_ENDPOINT = "game/{game_id}/boxscore"

# Function to fetch game IDs for a specific season and game type
def fetch_game_ids(season, game_type):
    endpoint = f"{SCHEDULE_ENDPOINT}?season={season}&gameType={game_type}"
    url = f"{NHL_API_URL}{endpoint}"

    response = requests.get(url)
    data = response.json()

    game_ids = []

    for date in data["dates"]:
        for game in date["games"]:
            game_ids.append(game["gamePk"])

    return game_ids

# Function to fetch game teams stats for a specific game ID and team ID
def fetch_game_teams_stats(game_id, team_id):
    endpoint = f"{GAME_STATS_ENDPOINT}".format(game_id=game_id)
    url = f"{NHL_API_URL}{endpoint}"

    response = requests.get(url)
    data = response.json()

    home_team_id = data["teams"]["home"]["team"]["id"]
    away_team_id = data["teams"]["away"]["team"]["id"]
    home_or_away = "home" if team_id == home_team_id else "away"
    
    # Calculate if the team won
    home_goals = data["teams"]["home"]["teamStats"]["teamSkaterStats"]["goals"]
    away_goals = data["teams"]["away"]["teamStats"]["teamSkaterStats"]["goals"]
    won = home_goals > away_goals

    game_teams_stats = {
        "game_id": game_id,
        "team_id": team_id,
        "HoA": home_or_away,
        "won": won,
        # "settled_in": data["decisions"]["winner"] if won else data["decisions"]["loser"],
        "head_coach": data["teams"]["home"]["coaches"][0]["person"]["fullName"],
        "goals": home_goals if home_or_away == "home" else away_goals,
        "shots": data["teams"][home_or_away]["teamStats"]["teamSkaterStats"]["shots"],
        "hits": data["teams"][home_or_away]["teamStats"]["teamSkaterStats"]["hits"],
        "pim": data["teams"][home_or_away]["teamStats"]["teamSkaterStats"]["pim"],
    }

    return game_teams_stats

# Specify the seasons and game types you want to retrieve data for
game_types = ["R", "P"]  # "R" for regular season, "P" for playoff

# Fetch game teams stats data for the specified seasons and game types
all_data = []

for season in seasons:
    for game_type in game_types:
        game_ids = fetch_game_ids(season, game_type)
        for game_id in game_ids:
            # Fetch stats for both home and away teams
            home_team_id = fetch_game_teams_stats(game_id, "home")
            away_team_id = fetch_game_teams_stats(game_id, "away")
            
            # Append the stats for each team to the list
            all_data.append(home_team_id)
            all_data.append(away_team_id)

# Create a DataFrame from the extracted data
df_game_teams_stats = pd.DataFrame(all_data)
