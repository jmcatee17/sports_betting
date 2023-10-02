import os
import sys
from nhl_api_calls_v2 import get_nhl_data
from nhl_api_calls_v2 import (update_maintained_game, 
                                         update_maintained_game_teams_stats, 
                                         update_maintained_game_skater_stats, 
                                         update_maintained_game_goalie_stats)
import pandas as pd
from datetime import datetime
from airflow.models import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.models import Variable

# Identify Working Directory
os.chdir('/Users/jdmcatee/Desktop/sports_betting/data/maintained_tables')

# Get updated data for start date and end date
nhl_api_calls_dict = get_nhl_data(start_date = '2023-09-29', end_date = '2023-10-01')

# Update Pandas Data Frames to include all information
df_updated_game = update_maintained_game(nhl_api_calls_dict = nhl_api_calls_dict)
df_updated_game_teams_stats = update_maintained_game_teams_stats(nhl_api_calls_dict)
df_updated_game_skater_stats = update_maintained_game_skater_stats(nhl_api_calls_dict)
df_updated_game_goalie_stats = update_maintained_game_goalie_stats(nhl_api_calls_dict)

with DAG(
    dag_id = 'nhl_stats_dag',
    schedule_internal='@daily',
    start_date=datetime(year=2023, month=9, day=1),
    catchup = False
) as dag:

    # 1. Get Current datetime
    task_get_datetime = BashOperator(
        task_id='get_datetime',
        bash_command='date'

    )