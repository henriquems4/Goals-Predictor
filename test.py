import json
import requests

url = "https://api-football-v1.p.rapidapi.com/v3/teams"

querystring = {"league":"253","season":"2024"}

headers = {
	"x-rapidapi-key": "your-api",
	"x-rapidapi-host": "api-football-v1.p.rapidapi.com"
}

response = requests.get(url, headers=headers, params=querystring)


# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    data = response.json()
    
    # Save the JSON response to a file
    with open('MLS_leagues.json', 'w') as file:
        json.dump(data, file, indent=4)
    print("Data has been written to values.json")
else:
    print(f"Failed to retrieve data: {response.status_code}")
    
