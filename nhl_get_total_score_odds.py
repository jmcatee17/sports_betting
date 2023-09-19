import requests
import json
import csv
import pandas as pd
from datetime import datetime
import requests
import os
os.chdir('/Users/jdmcatee/Desktop/Sports Betting')

# An api key is emailed to you when you sign up to a plan
# Get a free API key at https://api.the-odds-api.com/
API_KEY = 'cdac905fd386c9776b2817e34b2b2224'

SPORT = 'icehockey_nhl' # use the sport_key from the /sports endpoint below, or use 'upcoming' to see the next 8 games across all sports

REGIONS = 'us' # uk | us | eu | au. Multiple can be specified if comma delimited

MARKETS = 'totals' #'h2h,spreads' # h2h | spreads | totals. Multiple can be specified if comma delimited

ODDS_FORMAT = 'american' # decimal | american

DATE_FORMAT = 'iso' # iso | unix


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#
# Now get a list of live & upcoming games for the sport you want, along with odds for different bookmakers
# This will deduct from the usage quota
# The usage quota cost = [number of markets specified] x [number of regions specified]
# For examples of usage quota costs, see https://the-odds-api.com/liveapi/guides/v4/#usage-quota-costs
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

odds_response = requests.get(
    f'https://api.the-odds-api.com/v4/sports/{SPORT}/odds',
    params={
        'api_key': API_KEY,
        'regions': REGIONS,
        'markets': MARKETS,
        'oddsFormat': ODDS_FORMAT,
        'dateFormat': DATE_FORMAT,
    }
)

if odds_response.status_code != 200:
    print(f'Failed to get odds: status_code {odds_response.status_code}, response body {odds_response.text}')

else:
    odds_json = odds_response.json()
    print('Number of events:', len(odds_json))
    print(odds_json)

    # Check the usage quota
    print('Remaining requests', odds_response.headers['x-requests-remaining'])
    print('Used requests', odds_response.headers['x-requests-used'])
        # Define the CSV file name

time_pulled = datetime.now()

# Create Dataframe from JSON Results:
df = pd.json_normalize(odds_json, record_path=['bookmakers', 'markets', 'outcomes'], 
                      meta=['id', 'sport_key', 'sport_title', 'commence_time', 'home_team', 'away_team',
                            ['bookmakers', 'key', 'title', 'last_update'],
                            ['bookmakers', 'key']], errors='ignore')  # Include the sportsbook key

# Change Over / Under to CAPS
df['name'] = df['name'].apply(lambda x: x.upper())

# Pivot the DataFrame using pivot_table
pivoted_df = df.pivot_table(index=['id', 'sport_key', 'sport_title', 'commence_time', 'home_team', 'away_team', 'point'], columns=['bookmakers.key', 'name'], values=['price', 'name'], aggfunc='first')

# Flatten the multi-level columns
pivoted_df.columns = [f'{col[1]}_{col[2]}' for col in pivoted_df.columns]

# Reset the index
pivoted_df.reset_index(inplace=True)

# Display the pivoted DataFrame
pivoted_df['time_pulled'] = time_pulled

# Define the CSV file path
csv_file_path = 'nhl_total_odds.csv'

# Load the existing CSV file if it exists
try:
    existing_df = pd.read_csv(csv_file_path)
except FileNotFoundError:
    existing_df = pd.DataFrame()

# Append the new data to the existing data
combined_df = existing_df.append(pivoted_df, ignore_index=True)

# Remove duplicates based on the unique identifier ('id' and 'point' in this case)
combined_df.drop_duplicates(subset=['id', 'point'], keep='last', inplace=True)

# Save the updated data to the CSV file
combined_df.to_csv(csv_file_path, index=False)