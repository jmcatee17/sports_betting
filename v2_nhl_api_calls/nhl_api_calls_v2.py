import requests
from datetime import datetime
import pandas as pd

# start_date = '2023-09-10'
# end_date = '2023-09-28'

def get_nhl_data(start_date, end_date):
    # Get Game ID List:
    api_endpoint = 'https://api-web.nhle.com/v1/schedule/'

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
        
        if data["homeTeam"].get("score") != None:
            outcome_prefix = 'home win ' if data["homeTeam"].get("score") > data["awayTeam"].get("score") else 'away win'
            outcome_suffix = 'REG' if len(data['boxscore']['linescore']['byPeriod']) == 3 else 'OT'
        
        game_stats_dict = {
            "game_id" : GAMEID,
            "season" : data["season"],
            "type" : data["gameType"],
            "date_time_GMT" : data["startTimeUTC"],
            "away_team_id" : str(data["awayTeam"]["id"]),
            "home_team_id" : str(data["homeTeam"]["id"]),
            "away_goals" : data["awayTeam"].get("score") if data["gameState"] != 'FUT' else None,
            "home_goals" : data["homeTeam"].get("score") if data["gameState"] != 'FUT' else None,
            "outcome" : None if data["gameState"] == 'FUT' else outcome_prefix + outcome_suffix,
            "venue": data["venue"],
            "game_state" : data["gameState"]
        }
        
        game_stats.append(game_stats_dict)
        
        # Only append stats for games that have started
        if datetime.strptime(data["startTimeUTC"], "%Y-%m-%dT%H:%M:%SZ") < datetime.now() and data.get("boxscore") != None:
            home_team_stats_dict = {
                "game_id" : GAMEID,
                "team_id" : str(data["homeTeam"].get("id")),
                "HoA" : "home",
                "won" : None if data["homeTeam"].get("score") is None else True if data["homeTeam"].get("score") > data["awayTeam"].get("score") else False,
                "settled_in" : outcome_suffix,
                "head_coach" : None if data.get("boxscore") == None else data.get("boxscore")["gameInfo"]["homeTeam"]["headCoach"],
                "goals" : data["homeTeam"].get("score"),
                "shots" : data["homeTeam"].get('sog'),
                "hits" : data["homeTeam"].get("hits"),
                "pim" : data["homeTeam"].get("pim"),
                "blocks" : data["homeTeam"].get("blocks"),
                "faceoffWinningPctg" : data["homeTeam"].get("faceoffWinningPctg"),
                "powerPlayConversion" : data["homeTeam"].get("powerPlayConversion"),
            }

            away_team_stats_dict = {
                "game_id" : GAMEID,
                "team_id" : str(data["awayTeam"].get("id")),
                "HoA" : "away",
                "won" : None if data["homeTeam"].get("score") is None else False if data["homeTeam"].get("score") > data["awayTeam"].get("score") else True,
                "settled_in" : outcome_suffix,
                "head_coach" : None if data.get("boxscore") == None else data.get("boxscore")["gameInfo"]["awayTeam"]["headCoach"],
                "goals" : data["awayTeam"].get("score"),
                "shots" : data["awayTeam"].get("sog"),
                "hits" : data["awayTeam"].get("hits"),
                "pim" : data["awayTeam"].get("pim"),
                "blocks" : data["awayTeam"].get("blocks"),
                "faceoffWinningPctg" : data["awayTeam"].get("faceoffWinningPctg"),
                "powerPlayConversion" : data["awayTeam"].get("powerPlayConversion"),
            }

            for player_type in ['forwards', 'defense']:
                for player in data['boxscore']['playerByGameStats']['homeTeam'][player_type]:
                    home_team_skater_stats_dict = {
                        "game_id" : GAMEID,
                        "player_id" : player.get('playerId'),
                        "team_id" : str(data["homeTeam"]["id"]),
                        "time_on_ice" : player.get('toi'),
                        "assists" : player.get('assists'),
                        "goals" : player.get('goals'),
                        "shots" : player.get('shots'),
                        "hits" : player.get('hits'),
                        "power_play_goals" : player.get('powerPlayGoals'),
                        "power_play_assists" : player.get('powerPlayPoints') - player.get('powerPlayGoals'),
                    }
                    game_skater_stats.append(home_team_skater_stats_dict)

            for player_type in ['forwards', 'defense']:
                for player in data['boxscore']['playerByGameStats']['awayTeam'][player_type]:
                    away_team_skater_stats_dict = {
                        "game_id" : GAMEID,
                        "player_id" : player.get('playerId'),
                        "team_id" : str(data["awayTeam"]["id"]),
                        "time_on_ice" : player.get('toi'),
                        "assists" : player.get('assists'),
                        "goals" : player.get('goals'),
                        "shots" : player.get('shots'),
                        "hits" : player.get('hits'),
                        "power_play_goals" : player.get('powerPlayGoals'),
                        "power_play_assists" : player.get('powerPlayPoints') - player.get('powerPlayGoals'),
                    }
                    game_skater_stats.append(away_team_skater_stats_dict)

            for goalie in data['boxscore']['playerByGameStats']['homeTeam']['goalies']:
                home_team_goalie_stats_dict = {
                    "game_id" : GAMEID,
                    "player_id" : goalie.get('playerId'),
                    "team_id" : str(data["homeTeam"]["id"]),
                    "time_on_ice" : goalie.get('toi'),
                    "pim" : goalie.get('pim'),
                    "even_strength_shots_against" : goalie.get('evenStrengthShotsAgainst'),
                    "saves_shots_against" : goalie.get('saveShotsAgainst'),
                    "power_play_shots_against" : goalie.get('powerPlayShotsAgainst'),
                    "goals_against" : goalie.get('goalsAgainst'),
                }
                if home_team_goalie_stats_dict['time_on_ice'] != '00:00':
                    game_goalie_stats.append(home_team_goalie_stats_dict)

            for goalie in data['boxscore']['playerByGameStats']['awayTeam']['goalies']:
                away_team_goalie_stats_dict = {
                    "game_id" : GAMEID,
                    "player_id" : goalie.get('playerId'),
                    "team_id" : str(data["awayTeam"]["id"]),
                    "time_on_ice" : goalie.get('toi'),
                    "pim" : goalie.get('pim'),
                    "even_strength_shots_against" : goalie.get('evenStrengthShotsAgainst'),
                    "saves_shots_against" : goalie.get('saveShotsAgainst'),
                    "power_play_shots_against" : goalie.get('powerPlayShotsAgainst'),
                    "goals_against" : goalie.get('goalsAgainst'),
                }
                if away_team_goalie_stats_dict['time_on_ice'] != '00:00':
                    game_goalie_stats.append(away_team_goalie_stats_dict)


            # Append home and away stats
            game_team_stats.append(home_team_stats_dict)
            game_team_stats.append(away_team_stats_dict)

    # Create a DataFrame from the extracted data
    df_game = pd.DataFrame(game_stats)
    df_game_teams_stats = pd.DataFrame(game_team_stats)
    df_game_skater_stats = pd.DataFrame(game_skater_stats)
    df_game_goalie_stats = pd.DataFrame(game_goalie_stats)

    # Create Necessary Data Engineering Manipulations to Match Scehma

    ## 1: df_game
    # Define a mapping dictionary for replacement
    replacement_dict = {1: 'PR', 2: 'R', 3: 'P', 4: 'A'}

    # Use the replace method to replace values in the column
    df_game['type'] = df_game['type'].replace(replacement_dict)

    # Get Selected API Columns and drop duplicate
    df_game = df_game[['game_id', 'season', 'type', 'date_time_GMT', 'away_team_id', 'home_team_id', 'away_goals', 'home_goals', 'outcome', 'venue', 'game_state']]
    df_game = df_game.drop_duplicates()

    ## 2: df_game_team_stats
    df_game_teams_stats['powerPlayOpportunities'] = df_game_teams_stats['powerPlayConversion'].apply(lambda x: x.split('/')[0]).astype(int)
    df_game_teams_stats['powerPlayGoals'] = df_game_teams_stats['powerPlayConversion'].apply(lambda x: x.split('/')[1]).astype(int)
    df_game_team_stats = df_game_teams_stats[['game_id', 'team_id', 'HoA', 'won', 'settled_in', 'head_coach', 'goals', 'shots', 'hits', 'pim', 'blocks', 'faceoffWinningPctg', 'powerPlayOpportunities', 'powerPlayGoals']]
    df_game_teams_stats = df_game_teams_stats.drop_duplicates()

    ## 3: df_game_skater_stats
    df_game_skater_stats = df_game_skater_stats[['game_id', 'player_id', 'team_id', 'time_on_ice', 'assists', 'goals', 'shots', 'hits', 'power_play_goals', 'power_play_assists']]
    df_game_skater_stats = df_game_skater_stats.drop_duplicates()

    ## 4: df_game_goalie_stats
    df_game_goalie_stats['shots'] = df_game_goalie_stats['saves_shots_against'].apply(lambda x: x.split('/')[1]).astype(int)
    df_game_goalie_stats['saves'] = df_game_goalie_stats['saves_shots_against'].apply(lambda x: x.split('/')[0]).astype(int)
    df_game_goalie_stats['power_play_saves'] = df_game_goalie_stats['power_play_shots_against'].apply(lambda x: x.split('/')[0]).astype(int)
    df_game_goalie_stats['short_handed_saves'] = df_game_goalie_stats['saves'] - df_game_goalie_stats['power_play_saves'] - df_game_goalie_stats['even_strength_shots_against'].apply(lambda x: x.split('/')[0]).astype(int)
    df_game_goalie_stats['save_percentage'] = df_game_goalie_stats['saves_shots_against'].apply(lambda x: None if int(x.split('/')[1]) == 0 else int(x.split('/')[0]) / int(x.split('/')[1]))
    df_game_goalie_stats = df_game_goalie_stats[['game_id', 'team_id', 'time_on_ice', 'pim', 'shots', 'saves', 'power_play_saves', 'short_handed_saves', 'goals_against', 'save_percentage']]
    df_game_goalie_stats = df_game_goalie_stats.drop_duplicates()

    return {'df_game' : df_game, 'df_game_teams_stats' : df_game_teams_stats, 'df_game_skater_stats' : df_game_skater_stats, 'df_game_goalie_stats' : df_game_goalie_stats}