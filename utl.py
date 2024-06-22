import requests
import json
import os

def import_values(team_id, team, games):
    # Ensure the 'last_games' directory exists
    if not os.path.exists('last_games'):
        os.makedirs('last_games')

    # Construct the file path for the JSON file inside the last_games folder
    file_name = team + str(games) + '.json'
    file_path = os.path.join('last_games', file_name)

    # Check if the file already exists
    if os.path.exists(file_path):
        print(f"File {file_path} already exists. Skipping creation.")
        return

    # Define the URL and query parameters
    url = "https://api-football-v1.p.rapidapi.com/v3/fixtures"
    querystring = {"team": str(team_id), "last": str(games)}

    # Define the headers
    headers = {
        "x-rapidapi-key": "your-key",
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com"
    }

    # Make the request
    response = requests.get(url, headers=headers, params=querystring)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()

        # Save the JSON response to a file
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
        print(f"Data has been written to {file_path}")
    else:
        print(f"Failed to retrieve data: {response.status_code}")


def loop_leagues():
    with open('teams_leagues_id/euro_teams.json', 'r') as file:
        data_st = json.load(file)
    for i in data_st.get('response', []):
        print(i['team']['name'])
        import_values(i['team']['id'], i['team']['name'], 10)

def check_tier(team_id, team, games):
    none = 0
    tier = 0
    # Construct the file path for the JSON file inside the last_games folder
    file_name = team + str(games) + '.json'
    file_path = os.path.join('last_games', file_name)
    
    with open(file_path, 'r') as file:
        data_st = json.load(file)
    for fixture in data_st.get('response', []):
        home_team = fixture['teams']['home']['name']
        away_team = fixture['teams']['away']['name']
        date = fixture['fixture']['date']
        home_score = fixture['goals']['home']
        away_score = fixture['goals']['away']
        if home_score is not None or away_score is not None:
            if home_team == team:
                if home_score == away_score:
                    tier += 1
                elif home_score >= away_score:
                    tier += 3
            else:
                if home_score == away_score:
                    tier += 1
                elif home_score <= away_score:
                    tier += 3
        else:
            none += 1
    if none != 0:
        tier = ((games) * tier) / (games - 1)
    return tier

def train_test_split(df, test_size=10):
    train = df.iloc[test_size:]
    test = df.iloc[:test_size]
    return train, test

def delete_all_games ():
    # URL of the Flask application
    url = 'http://127.0.0.1:5000/delete_league'

    # Your secret API key
    headers = {'x-api-key': 'enter your api key'}

    response = requests.get(url, headers=headers)

    print(response.status_code)
    print(response.json())

delete_all_games()