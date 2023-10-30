# Import necessary packages
from functools import partial
import pickle
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold, GridSearchCV, StratifiedKFold
from xgboost import XGBRegressor

# Skopt functions
from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper, DeltaYStopper
from skopt.space import Real, Categorical, Integer

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

# Start Analysis from 20102011 season forwards (due to missing faceoffpercentage data before)
start_season = '20102011'
start_game = '2010020001'

# Filter DataFrames:
df_game = df_game[(df_game['season'] >= start_season) & (df_game['game_state'] != 'FUT')]
df_game_skater_stats = df_game_skater_stats[df_game_skater_stats['game_id'] >= start_game]
df_game_teams_stats = df_game_teams_stats[df_game_teams_stats['game_id'] >= start_game]

def time_to_seconds(time_str):
    time_str = str(time_str)
    time_parts = time_str.split(':')
    if len(time_parts) == 0:
        return time_str
    if len(time_parts) == 2:
        hours, minutes = map(int, time_parts)
        return hours * 3600 + minutes * 60
    elif len(time_parts) == 3:
        hours, minutes, seconds = map(int, time_parts)
        return hours * 3600 + minutes * 60 + seconds

# # Apply the function to the 'Time' column and create a new 'Seconds' column
df_game_skater_stats['time_on_ice'] = df_game_skater_stats['time_on_ice'].apply(lambda x: time_to_seconds(x))

for c in ['home_goals', 'away_goals']:
    df_game[c] = df_game[c].astype(int)

for c in ['time_on_ice', 'assists', 'goals', 'shots' ,'hits', 'power_play_goals', 'power_play_assists']:
    df_game_skater_stats[c] = df_game_skater_stats[c].fillna(0).astype(int)

df_game_teams_stats.loc[df_game_teams_stats['won'] == 'True', 'won'] = 1
df_game_teams_stats.loc[df_game_teams_stats['won'] == 'False', 'won'] = 0
for c in ['goals', 'shots', 'hits', 'won', 'pim', 'blocks', 'powerPlayOpportunities', 'powerPlayGoals']:
    df_game_teams_stats[c] = df_game_teams_stats[c].fillna(0).astype(int)

df_game_teams_stats['faceoffWinningPctg'] = df_game_teams_stats['faceoffWinningPctg'].astype(float)

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
    df_top_scorers['top_scorers_players_ids'] = df_top_scorers['top_scorers_players_ids'].apply(lambda x: list(x[:20]))

    return df_top_scorers

def count_overlaps(list1, list2):
    try:
        return len(set(list1) & set(list2))
    except TypeError:
            # Handle the case when list1 or list2 is not iterable
            return 0

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

    df_rolling = df_merged.sort_values(by = ['season', 'team_id', 'game_number'], ascending = True).reset_index(drop=True)
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
        df_rolling[value] = df_rolling.groupby(['season', 'team_id'])[key].apply(lambda x: x.rolling(window = game_window, min_periods=1).mean().shift(1)).reset_index(drop=True)
    
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

    df_top_players_merge['top_scorers_players_ids'] = df_top_players_merge['top_scorers_players_ids'].tolist()
    df_top_players_merge['game_player_ids'] = df_top_players_merge['game_player_ids'].tolist()

    # Find unique overlaps between players and top scorers in each game (ie how many top scorers playing for each team)
    df_top_players_merge['count_top_players'] = df_top_players_merge.apply(lambda x: count_overlaps(x['top_scorers_players_ids'], x['game_player_ids']), axis = 1)
    df_top_players_merge = df_top_players_merge[['game_id', 'team_id', 'count_top_players']]
    
    df_top_players_opposing = df_top_players_merge[['game_id', 'team_id', 'count_top_players']].rename(columns={'count_top_players': 'opposing_count_top_players', 'team_id' : 'opponent_id'})
        
    # Get create_rolling_stats() output and left join with df_top_players_merge
    df = create_rolling_stats()
    
    df_output = pd.merge(df, df_top_players_merge, on = ['game_id', 'team_id'], how = 'left')
    df_output = pd.merge(df_output, df_top_players_opposing, on = ['game_id', 'opponent_id'], how = 'left')
    
    return df_output

# Create Train - Test Split
df_i = add_top_scorers_and_goalie_counts()
df_i = df_i.dropna()
drop_list = ['game_id', 'team_id', 'season', 'date_time_GMT', 'venue', 'opponent_id', 'type', 'settled_in', 'won', 'hits', 'shots', 'pim', 'powerPlayOpportunities', 'powerPlayGoals', 'blocks']
append_list = ['home_against', 'index_against', 'goals_against']
for i in drop_list:
    if i != 'game_id':
        append_list.append(i + '_against')
drop_list = drop_list + append_list

df_i = df_i.drop(columns = drop_list)

# Set feature and target variables
X = df_i.drop(columns = 'goals')
y = df_i[['goals']]

# Apply train test split to the data, random state of 42 for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X.info())

# Baseline Random Forest Regressor
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

# Predict and compare:
rf_preds = rf.predict(X_test)
rf_preds = pd.DataFrame(rf_preds, columns = ['prediction'])

# Compute summary statistics:
mse = mean_squared_error(y_test, rf_preds)
rmse = mse**.5
print("Random Forest RMSE: ", rmse)
pickle.dump(rf, open('random_forest.sav', 'wb'))

# create an xgboost regression model
xgb = SGDRegressor(n_estimator = 800, eta = .01, max_depth = 10)

# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

# evaluate model
scores = cross_val_score(xgb, X, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
#force scores to be positive
scores = abs(scores)** (1/2)

xgb.fit(X_train, y_train)
print('XGBoost Mean RMSE: %.3f (%.3f)' % (scores.mean(), scores.std()))
pickle.dump(xgb, open('xgboost.sav', 'wb'))

# Reporting util for different optimizers
def report_perf(optimizer, X, y, title="model", callbacks=None):
    """
    A wrapper for measuring time and performances of different optmizers
    
    optimizer = a sklearn or a skopt optimizer
    X = the training set 
    y = our target
    title = a string label for the experiment
    """
    start = time()
    
    if callbacks is not None:
        optimizer.fit(X, y, callback=callbacks)
    else:
        optimizer.fit(X, y)
        
    d=pd.DataFrame(optimizer.cv_results_)
    best_score = optimizer.best_score_
    best_score_std = d.iloc[optimizer.best_index_].std_test_score
    best_params = optimizer.best_params_
    
    print((title + " took %.2f seconds,  candidates checked: %d, best CV score: %.3f "
           + u"\u00B1"+" %.3f") % (time() - start, 
                                   len(optimizer.cv_results_['params']),
                                   best_score,
                                   best_score_std))    
    print('Best parameters:')
    print(best_params)
    return best_params

# Setting the scoring function
scoring = make_scorer(partial(mean_squared_error, squared = False), 
                      greater_is_better = False)

# Setting the validation strategy
skf = StratifiedKFold(n_splits = 7,
                      shuffle = True, 
                      random_state = 42)

cv_strategy = list(skf.split(X_train, y_train))

# Setting the basic regressor
xgb_bayes = XGBRegressor(random_state = 42, booster = 'gbtree', objective = 'reg:squarederror', tree_method = 'hist')

# Setting the search space
search_spaces = {'learning_rate': Real(0.01, 0.1, 'uniform'),
                 'max_depth': Integer(2, 8),
                 'subsample': Real(0.1, 1.0, 'uniform'),
                 'colsample_bytree': Real(0.1, 1.0, 'uniform'), # subsample ratio of columns by tree
                 #'reg_lambda': Real(1e-9, 100., 'uniform'), # L2 regularization
                 #'reg_alpha': Real(1e-9, 100., 'uniform'), # L1 regularization
                 'n_estimators': Integer(350, 900)
   }


# Wrapping everything up into the Bayesian optimizer
opt = BayesSearchCV(estimator= xgb_bayes,                                    
                    search_spaces = search_spaces,                      
                    scoring = scoring,                                  
                    cv = cv_strategy,                                           
                    n_iter = 120,                                       # max number of trials
                    n_points = 1,                                       # number of hyperparameter sets evaluated at the same time
                    n_jobs = 1,                                       # if not iid it optimizes on the cv score
                    return_train_score = False,                         
                    refit = False,                                      
                    optimizer_kwargs = {'base_estimator': 'GP'},        # optmizer parameters: we use Gaussian Process (GP)
                    random_state = 42)                                   # random state for replicability

# Running the optimizer
overdone_control = DeltaYStopper(delta=0.0001)                    # We stop if the gain of the optimization becomes too small
time_limit_control = DeadlineStopper(total_time=60*60*4)          # We impose a time limit (4 hours)

best_params = report_perf(opt, X_train, y_train,'XGBoost_regression', 
                          callbacks = [overdone_control, time_limit_control])

# Transferring the best parameters to our basic regressor
xgb_bayes = XGBRegressor(random_state=0, booster='gbtree', objective='reg:squarederror', tree_method='hist', **best_params)
xgb_bayes.fit(X_train, y_train)

xgb_bayes_preds = xgb_bayes.predict(X_test)
xgb_bayes_preds = pd.DataFrame(xgb_bayes_preds, columns = ['prediction'])

# Compute summary statistics:
mse = mean_squared_error(y_test, xgb_bayes_preds)
rmse = mse**.5
print("XGBoost Hyperparameter Optimization RMSE: ", rmse)
pickle.dump(xgb_bayes, open('xgboost_bayes.sav', 'wb'))