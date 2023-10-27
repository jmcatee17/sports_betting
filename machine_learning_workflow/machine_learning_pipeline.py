# Import necessary packages
import pandas as pd
import numpy as np
import os

# Change Working directory
os.chdir('/Users/jdmcatee/Desktop/sports_betting/data/maintained_tables')

df_game = pd.read_csv('fct_maintained_game.csv', index_col = None)
df_game_skater_stats = pd.read_csv('fct_maintained_game_skater_stats.csv', index_col = None)
df_game_teams_stats = pd.read_csv('fct_maintained_game_teams_stats.csv', index_col = None)

# Convert game_id, team_id, opponent_id, season to strings
counter = 0
for t in [df_game, df_game_skater_stats, df_game_teams_stats]:
    if counter == 0:
        column_list = ['game_id', 'season', 'date_time_GMT', 'home_team_id', 'away_team_id', 'outcome', 'venue', 'game_state']
    if counter == 1:
        column_list = ['game_id', 'team_id', 'player_id']
    if counter == 2:
        column_list = ['game_id', 'team_id', 'HoA', 'won', 'settled_in', 'head_coach']
    for c in column_list:
        t[c] = t[c].astype(str)
    counter +=1

def top_scorers(df_game_skater_stats = df_game_skater_stats):
    # Find top 20 goal scorers by season:

    # For some reason there are duplicates in game_skater_stats dataframe
    # Ensure that goals per season matches nhl stats. There is a discrepancy somewhere
    df_result = pd.merge(df_game_skater_stats.drop_duplicates(), df_game[['game_id', 'season']].drop_duplicates(), how = 'left', on = ['game_id'])
    df_result_grouped = df_result.groupby(['season', 'player_id'])['goals'].agg('sum').reset_index().sort_values(by = 'goals', ascending = False)
    
    # Join names
    # df_result_grouped_names = pd.merge(df_result_grouped, df_player_info, how = 'left', on = ['player_id'])
    # df_top_scorers = df_result_grouped_names.groupby(['season'])['player_id'].apply(list).reset_index(name='top_scorers_players_ids')
    df_top_scorers = df_result_grouped.groupby(['season'])['player_id'].apply(list).reset_index(name='top_scorers_players_ids')
    df_top_scorers['top_scorers_players_ids'] = df_top_scorers['top_scorers_players_ids'].apply(lambda x: x[:20])

    return df_top_scorers

def count_overlaps(list1, list2):
    return len(set(list1) & set(list2))

def corsi_statistic(shots_for, shots_against, blocks_for, blocks_against):
    corsi_for = shots_for + blocks_for
    corsi_against = shots_against + blocks_against
    corsi_for_percent = corsi_for / (corsi_for + corsi_against)
    
    return corsi_for_percent

def pdo_statistic(goals_avg, shots_avg, goals_against_avg, shots_against_avg):
    #  theory that most teams will ultimately regress toward a sum of 100, 
    # is often viewed as a proxy for how lucky a team is
    goal_pct = goals_avg / shots_avg
    save_pct = (shots_against_avg - goals_against_avg) / shots_against_avg
    pdo = goal_pct + save_pct
    
    return pdo

def data_prep(df_game = df_game):
    '''
    This function takes in the pandas DataFrame df_game, drops duplicates, creates two rows for each game, and calculates values for the team of interest
    
    Parameters: 
        df_game (pandas DataFrame): The dataframe converted from game.csv downloaded from Kaggle
        ** default value is df_game
        
    Returns:
        df_game_1 (pandas DataFrame): Manipulated DataFrame down to game_id and team_id level (combination of two is the key)
    
    '''
    # Drop duplicates initially
    df_game = df_game.drop_duplicates()
    
    # Turn game_ids into list column and explode into integers:
    # This creates two rows for every game (one from perspective of each team)
    df_game['team_id'] = df_game.apply(lambda row: [row['home_team_id'], row['away_team_id']], axis=1)
    df_game_1 = df_game.explode('team_id')
    df_game_1['team_id'] = df_game_1['team_id'].astype(str).astype(int)

    # Create column for if team is home
    df_game_1.loc[df_game_1['home_team_id'] == df_game_1['team_id'], 'home'] = 1
    df_game_1['home'] = df_game_1['home'].fillna(0)
    df_game_1['home'] = df_game_1['home'].astype(int)

    # Get opponent id
    df_game_1['opponent_id'] = np.where(
                df_game_1['home'] == 1, 
                df_game_1['away_team_id'], 
                df_game_1['home_team_id']
                )
    # Get number of goals team scored
    df_game_1['goals'] = np.where(
                df_game_1['home'] == 1, 
                df_game_1['home_goals'], 
                df_game_1['away_goals']
                )

    df_game_1 = df_game_1[['game_id', 'team_id', 'opponent_id', 'season', 'date_time_GMT', 'type', 'venue', 'home', 'goals']].reset_index()
    
    # Convert game_id, team_id, opponent_id, season to strings
    for c in ['game_id', 'team_id', 'opponent_id', 'season']:
        df_game_1[c] = df_game_1[c].astype(str)
        
    return df_game_1

def create_rolling_stats(df_game_teams_stats = df_game_teams_stats, df_prep = data_prep(), game_window = 7):
    
    '''
    This function takes in the pandas DataFrame manipulated from the data_prep() function, merges data at the game and team stats levels \
    filters out null and missing values (ie starting from the 2002-2003 season), and creates rolling averages
    
    Parameters: 
        df_game_teams_stats (pandas DataFrame): The dataframe converted from game_teams_stats.csv downloaded from Kaggle
        ** default value is df_game_teams_stats
        
        df_prep (pandas DataFrame): The dataframe at the game level that we need to create rolling averages for
        ** default value is a call to the data_prep() function
        
        game_window (int): integer specifying how many games back to create rolling averages on
        
    Returns:
        df_rolling (pandas DataFrame): Manipulated DataFrame down to game_id and team_id level with rolling averages
    '''
    
    # Drop duplicates in df_game_team stats:
    df_game_teams_stats = df_game_teams_stats.drop_duplicates()
    df_game_teams_stats[['game_id', 'team_id']] = df_game_teams_stats[['game_id', 'team_id']].astype(str)

    # Join to df_game_teams_stats
    df_game_teams_stats_filter = df_game_teams_stats[['game_id', 'team_id', 'settled_in', 'won', 'hits', 'shots', 'pim', 'powerPlayOpportunities', 'powerPlayGoals',  'blocks']] # TODO: ADD 'giveaways', 'takeaways',
    df_merged = pd.merge(df_prep, df_game_teams_stats_filter, how = 'inner', on = ['game_id', 'team_id'])
    df_merged = pd.merge(df_merged, df_merged, left_on = ['game_id', 'opponent_id'], right_on = ['game_id', 'team_id'], suffixes = ['', '_against'], how = 'left')
    
    # Create column for game number in season
    df_merged['game_number'] = df_merged.sort_values(by = 'date_time_GMT', ascending = True).groupby(['season', 'team_id'])['date_time_GMT'].cumcount()

    # Drop null values (seasons 20002001 and 20012002), 8 instances in 20162017, 2017,2018
    df_merged = df_merged[(df_merged.season != '20002001') & (df_merged.season != '20012002')]
    df_merged = df_merged[~(df_merged.shots.isna())]

    # Ensure all games are regular season
    df_merged = df_merged[df_merged.type == 'R']
    # Create window functions for season averages

    df_rolling = df_merged.sort_values(by = ['season', 'team_id', 'game_number'], ascending = True)
    rolling_dictionary = {
                        "goals" : f"{game_window}_game_rolling_goals",
                        "shots" : f"{game_window}_game_rolling_shots",
                        "hits" : f"{game_window}_game_rolling_hits",
                        "won" : f"{game_window}_win_pct",
                        "pim" : f"{game_window}_game_rolling_pim",
                        "powerPlayOpportunities" : f"{game_window}_game_rolling_powerPlayOpportunities",
                        "powerPlayGoals" : f"{game_window}_game_rolling_powerPlayGoals",
                        # "giveaways" : f"{game_window}_game_rolling_giveaways",
                        # "takeaways" : f"{game_window}_game_rolling_takeaways",
                        "blocks" : f"{game_window}_game_rolling_blocked",
        
                        "goals_against" : f"{game_window}_game_rolling_goals_against",
                        "shots_against" : f"{game_window}_game_rolling_shots_against",
                        "hits_against" : f"{game_window}_game_rolling_hits_against",
                        "pim_against" : f"{game_window}_game_rolling_pim_against",
                        "powerPlayOpportunities_against" : f"{game_window}_game_rolling_powerPlayOpportunities_against",
                        "powerPlayGoals_against" : f"{game_window}_game_rolling_powerPlayGoals_against",
                        # "giveaways_against" : f"{game_window}_game_rolling_giveaways_against",
                        # "takeaways_against" : f"{game_window}_game_rolling_takeaways_against",
                        "blocks_against" : f"{game_window}_game_rolling_blocked_against",
                      }
    for key, value in rolling_dictionary.items():
        df_rolling[value] = df_rolling.groupby(['season', 'team_id'])[key].apply(lambda x: x.rolling(window = game_window, min_periods=1).mean().shift(1))
    
    # Drop Index Column
    df_rolling = df_rolling.drop(columns = ['index']).reset_index(drop = True)
    
    # Add opposing team stats (hits, blocked shots, powerplay opportunities)
    df_opposing = df_rolling[['game_id', 'team_id', 'game_number'] + list(rolling_dictionary.values())]
    
    # Rename columns appropriately
    df_opposing = df_opposing.rename(columns = {'team_id': 'opponent_id'})
    df_opposing = df_opposing.rename(columns={c: 'opposing_' + c for c in df_opposing.columns if c in list(rolling_dictionary.values()) or c == 'game_number'})
    
    # Merge team and opponent rolling data
    df_joined_rolling = pd.merge(df_rolling, df_opposing, left_on = ['game_id', 'opponent_id'], right_on = ['game_id', 'opponent_id'], how = 'left')
    
    # Drop first game of each season for both teams playing:
    df_joined_rolling = df_joined_rolling[df_joined_rolling.game_number != 0]
    df_joined_rolling = df_joined_rolling[df_joined_rolling.opposing_game_number != 0]
    
    # Calculate Corsi and POD Statistics:
    df_joined_rolling['corsi_statistic_team'] = df_joined_rolling.apply(lambda row : 
                                                                        corsi_statistic(
                                                                                        shots_for = row[f"{game_window}_game_rolling_shots"], 
                                                                                        shots_against = row[f"{game_window}_game_rolling_shots_against"], 
                                                                                        blocks_for = row[f"{game_window}_game_rolling_blocked"], 
                                                                                        blocks_against = row[f"{game_window}_game_rolling_blocked_against"]), 
                                                                        axis = 1)
    df_joined_rolling['corsi_statistic_opponent'] = df_joined_rolling.apply(lambda row : 
                                                                        corsi_statistic(
                                                                                        shots_for = row[f"opposing_{game_window}_game_rolling_shots"], 
                                                                                        shots_against = row[f"opposing_{game_window}_game_rolling_shots_against"], 
                                                                                        blocks_for = row[f"opposing_{game_window}_game_rolling_blocked"], 
                                                                                        blocks_against = row[f"opposing_{game_window}_game_rolling_blocked_against"]), 
                                                                        axis = 1)
    
    df_joined_rolling['pdo_statistic_team'] = df_joined_rolling.apply(lambda row : 
                                                                        pdo_statistic(
                                                                                        goals_avg = row[f"{game_window}_game_rolling_goals"], 
                                                                                        shots_avg = row[f"{game_window}_game_rolling_shots"], 
                                                                                        goals_against_avg = row[f"{game_window}_game_rolling_goals_against"], 
                                                                                        shots_against_avg = row[f"{game_window}_game_rolling_shots_against"]), 
                                                                        axis = 1) 
    df_joined_rolling['pdo_statistic_opponent'] = df_joined_rolling.apply(lambda row : 
                                                                        pdo_statistic(
                                                                                        goals_avg = row[f"opposing_{game_window}_game_rolling_goals"], 
                                                                                        shots_avg = row[f"opposing_{game_window}_game_rolling_shots"], 
                                                                                        goals_against_avg = row[f"opposing_{game_window}_game_rolling_goals_against"], 
                                                                                        shots_against_avg = row[f"opposing_{game_window}_game_rolling_shots_against"]), 
                                                                        axis = 1)
    return df_joined_rolling

def add_top_scorers_and_goalie_counts(df_game_skater_stats = df_game_skater_stats, df_top_scorers = top_scorers()):
    # Get unique list of skaters playing by game and team
    df_stg1 = df_game_skater_stats.groupby(['game_id', 'team_id'])['player_id'].apply(lambda x: list(np.unique(x))).reset_index(name='game_player_ids')

    # Map game back to season using df_game
    df_season_map = df_game[['game_id', 'season']].drop_duplicates()
    df_season_merge = pd.merge(df_stg1, df_season_map, on = ['game_id'], how = 'left')
    df_top_players_merge = pd.merge(df_season_merge, df_top_scorers, on = 'season', how = 'left')

    # Convert game_id, team_id, opponent_id, season to strings
    for c in ['game_id', 'team_id']:
        df_top_players_merge[c] = df_top_players_merge[c].astype(str)
    
    # Find unique overlaps between players and top scorers in each game (ie how many top scorers playing for each team)
    df_top_players_merge['count_top_players'] = df_top_players_merge.apply(lambda x: count_overlaps(x['top_scorers_players_ids'], x['game_player_ids']), axis = 1)
    df_top_players_merge = df_top_players_merge[['game_id', 'team_id', 'count_top_players']]
    
    df_top_players_opposing = df_top_players_merge[['game_id', 'team_id', 'count_top_players']].rename(columns={'count_top_players': 'opposing_count_top_players', 'team_id' : 'opponent_id'})
        
    # Get create_rolling_stats() output and left join with df_top_players_merge
    df = create_rolling_stats()
    
    df_output = pd.merge(df, df_top_players_merge, on = ['game_id', 'team_id'], how = 'left')
    df_output = pd.merge(df_output, df_top_players_opposing, on = ['game_id', 'opponent_id'], how = 'left')
    
    return df_output

print(add_top_scorers_and_goalie_counts().info())