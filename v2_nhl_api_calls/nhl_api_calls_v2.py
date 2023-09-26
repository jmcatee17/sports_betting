import requests
from datetime import datetime
import pandas as pd

# Get Game ID List:
api_endpoint = 'https://api-web.nhle.com/v1/schedule/'

start_date = '2023-08-01'
end_date = '2023-09-30'

date_range = pd.date_range(start_date, end_date, 
              freq='W').strftime("%G-%m-%d").tolist()

game_id_list = []
for date in date_range:
    api_link = f"https://api-web.nhle.com/v1/schedule/{date}"
    response = requests.get(api_link)
    data = response.json()
    for day in data['gameWeek']:
        for game in day['games']:
            game_id_list.append(game['id'])
            
# Ensure no duplicates
game_id_list = set(game_id_list)

# Define the NHL API endpoint for the schedule endpoint
game_stats = []
game_team_stats = []
game_skater_stats = []
game_goalie_stats = []

for GAMEID in game_id_list:
    api_url = f"https://api-web.nhle.com/v1/gamecenter/{GAMEID}/boxscore"
    response = requests.get(api_url)
    data = response.json()
    
    outcome_prefix = 'NULL' if data["gameState"] == 'FUT' else 'home win ' if data["homeTeam"]["score"] > data["awayTeam"]["score"] else 'away win'
    outcome_suffix = '' if data["gameState"] == 'FUT' else'REG' if len(data['boxscore']['linescore']['byPeriod']) == 3 else 'OT'
    
    game_stats_dict = {
        "game_id" : GAMEID,
        "season" : data["season"],
        "type" : data["gameType"],
        "date_time_GMT" : data["startTimeUTC"],
        "away_team_id" : str(data["awayTeam"]["id"]),
        "home_team_id" : str(data["homeTeam"]["id"]),
        "away_goals" : data["awayTeam"]["score"] if data["gameState"] != 'FUT' else 'NULL',
        "home_goals" : data["homeTeam"]["score"] if data["gameState"] != 'FUT' else 'NULL',
        "outcome" : outcome_prefix + outcome_suffix,
        "venue": data["venue"],
        "game_state" : data["gameState"]
    }
    
    game_stats.append(game_stats_dict)
    
    # Only append stats for games that have started
    if datetime.strptime(data["startTimeUTC"], "%Y-%m-%dT%H:%M:%SZ") < datetime.now():
        home_team_stats_dict = {
            "game_id" : GAMEID,
            "team_id" : str(data["homeTeam"]["id"]),
            "HoA" : "home",
            "won" : True if data["homeTeam"]["score"] > data["awayTeam"]["score"] else False,
            "settled_in" : outcome_suffix,
            "head_coach" : data["boxscore"]["gameInfo"]["homeTeam"]["headCoach"],
            "goals" : data["homeTeam"]["score"],
            "shots" : data["homeTeam"]["sog"],
            "hits" : data["homeTeam"]["hits"],
            "pim" : data["homeTeam"]["pim"],
            "blocks" : data["homeTeam"]["blocks"],
            "faceoffWinningPctg" : data["homeTeam"]["faceoffWinningPctg"],
            "powerPlayConversion" : data["homeTeam"]["powerPlayConversion"],
        }

        away_team_stats_dict = {
            "game_id" : GAMEID,
            "team_id" : str(data["awayTeam"]["id"]),
            "HoA" : "away",
            "won" : False if data["homeTeam"]["score"] > data["awayTeam"]["score"] else True,
            "settled_in" : outcome_suffix,
            "head_coach" : data["boxscore"]["gameInfo"]["awayTeam"]["headCoach"],
            "goals" : data["awayTeam"]["score"],
            "shots" : data["awayTeam"]["sog"],
            "hits" : data["awayTeam"]["hits"],
            "pim" : data["awayTeam"]["pim"],
            "blocks" : data["awayTeam"]["blocks"],
            "faceoffWinningPctg" : data["awayTeam"]["faceoffWinningPctg"],
            "powerPlayConversion" : data["awayTeam"]["powerPlayConversion"],
        }

        for player_type in ['forwards', 'defense']:
            for player in data['boxscore']['playerByGameStats']['homeTeam'][player_type]:
                home_team_skater_stats_dict = {
                    "game_id" : GAMEID,
                    "player_id" : player['playerId'],
                    "team_id" : str(data["homeTeam"]["id"]),
                    "time_on_ice" : player['toi'],
                    "assists" : player['assists'],
                    "goals" : player['goals'],
                    "shots" : player['shots'],
                    "hits" : player['hits'],
                    "power_play_goals" : player['powerPlayGoals'],
                    "power_play_assists" : player['powerPlayPoints'] - player['powerPlayGoals'],
                }
                game_skater_stats.append(home_team_skater_stats_dict)

        for player_type in ['forwards', 'defense']:
            for player in data['boxscore']['playerByGameStats']['awayTeam'][player_type]:
                away_team_skater_stats_dict = {
                    "game_id" : GAMEID,
                    "player_id" : player['playerId'],
                    "team_id" : str(data["awayTeam"]["id"]),
                    "time_on_ice" : player['toi'],
                    "assists" : player['assists'],
                    "goals" : player['goals'],
                    "shots" : player['shots'],
                    "hits" : player['hits'],
                    "power_play_goals" : player['powerPlayGoals'],
                    "power_play_assists" : player['powerPlayPoints'] - player['powerPlayGoals'],
                }
                game_skater_stats.append(away_team_skater_stats_dict)

        for goalie in data['boxscore']['playerByGameStats']['homeTeam']['goalies']:
            home_team_goalie_stats_dict = {
                "game_id" : GAMEID,
                "player_id" : goalie['playerId'],
                "team_id" : str(data["homeTeam"]["id"]),
                "time_on_ice" : goalie['toi'],
                "pim" : goalie['pim'],
                "even_strength_shots_against" : goalie['evenStrengthShotsAgainst'],
                "saves_shots_against" : goalie['saveShotsAgainst'],
                "power_play_shots_against" : goalie['powerPlayShotsAgainst'],
                "goals_against" : goalie['goalsAgainst'],
            }
            if home_team_goalie_stats_dict['time_on_ice'] != '00:00':
                game_goalie_stats.append(home_team_goalie_stats_dict)

        for goalie in data['boxscore']['playerByGameStats']['awayTeam']['goalies']:
            away_team_goalie_stats_dict = {
                "game_id" : GAMEID,
                "player_id" : goalie['playerId'],
                "team_id" : str(data["awayTeam"]["id"]),
                "time_on_ice" : goalie['toi'],
                "pim" : goalie['pim'],
                "even_strength_shots_against" : goalie['evenStrengthShotsAgainst'],
                "saves_shots_against" : goalie['saveShotsAgainst'],
                "power_play_shots_against" : goalie['powerPlayShotsAgainst'],
                "goals_against" : goalie['goalsAgainst'],
            }
            if away_team_goalie_stats_dict['time_on_ice'] != '00:00':
                game_goalie_stats.append(away_team_goalie_stats_dict)


        # Append home and away stats
        game_team_stats.append(home_team_stats_dict)
        game_team_stats.append(away_team_stats_dict)

# Create a DataFrame from the extracted data
df_game = pd.DataFrame(game_stats)
df_game_team_stats = pd.DataFrame(game_team_stats)
df_game_skater_stats = pd.DataFrame(game_skater_stats)
df_game_goalie_stats = pd.DataFrame(game_goalie_stats)

print(df_game.head())