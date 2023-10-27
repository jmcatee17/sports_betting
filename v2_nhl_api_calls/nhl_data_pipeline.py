import os
from nhl_api_calls_v2 import (get_nhl_data,
                            update_maintained_game, 
                            update_maintained_game_teams_stats, 
                            update_maintained_game_skater_stats, 
                            update_maintained_game_goalie_stats)
import pandas as pd
from datetime import datetime, timedelta

# Identify Working Directory
os.chdir('/Users/jdmcatee/Desktop/sports_betting/data/maintained_tables')

# Get each days data:
start_date = datetime.today() - timedelta(days=1)
start_date = start_date.strftime('%Y-%m-%d')
end_date = datetime.today().strftime('%Y-%m-%d')

# Get updated data for start date and end date
nhl_api_calls_dict = get_nhl_data(start_date = start_date, end_date = end_date)

# Update Pandas Data Frames to include all information
df_updated_game = update_maintained_game(nhl_api_calls_dict = nhl_api_calls_dict)
df_updated_game_teams_stats = update_maintained_game_teams_stats(nhl_api_calls_dict)
df_updated_game_skater_stats = update_maintained_game_skater_stats(nhl_api_calls_dict)
df_updated_game_goalie_stats = update_maintained_game_goalie_stats(nhl_api_calls_dict)

# Order by game_id Ascending
for df in [df_updated_game, df_updated_game_teams_stats, df_updated_game_skater_stats, df_updated_game_goalie_stats]:
    df = df.sort_values(by = 'game_id', ascending = True)

# Overwrite Files:
df_updated_game.to_csv('fct_maintained_game.csv', index = False)
df_updated_game_teams_stats.to_csv('fct_maintained_game_teams_stats.csv', index = False)
df_updated_game_skater_stats.to_csv('fct_maintained_game_skater_stats.csv',index = False)
df_updated_game_goalie_stats.to_csv('fct_maintained_game_goalie_stats.csv', index = False)

