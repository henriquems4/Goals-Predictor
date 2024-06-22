import json
import os
import predict_function
from utl import check_tier, import_values

def resolve_teams(team_1, team_2):
    teams = [team_1, team_2]
    for i in teams:
        games = 30
        import_values(i[1], i[0], games)
        i[2] = check_tier(i[1], i[0], games)
        
        # Construct the file path for the JSON file inside the last_games folder
        file_name = i[0] + str(games) + '.json'
        file_path = os.path.join('last_games', file_name)
        
        with open(file_path, 'r') as file:
            data_st = json.load(file)

        for fixture in data_st.get('response', []):
            home_team = fixture['teams']['home']['name']
            away_team = fixture['teams']['away']['name']
            date = fixture['fixture']['date']
            home_score = fixture['goals']['home']
            away_score = fixture['goals']['away']

            if home_team == i[0]:
                away_team_id = fixture['teams']['away']['id']
                import_values(away_team_id, away_team, 30)
                cur_tier = check_tier(away_team_id, away_team, 30)
                i[3].append(cur_tier)
                i[4].append(home_score)
                i[5].append(away_score)
            else:
                home_team_id = fixture['teams']['home']['id']
                import_values(home_team_id, home_team, 30)
                cur_tier = check_tier(home_team_id, home_team, 30)
                i[3].append(cur_tier)
                i[4].append(away_score)
                i[5].append(home_score)

    print(f"for team {team_1[0]} his tier is {team_1[2]}")
    print(f"for team {team_2[0]} his tier is {team_2[2]}")
    print(f"For team {team_1[0]} Tier list is {team_1[3]}")
    print(f"For team {team_1[0]} scored list is {team_1[4]}")
    print(f"For team {team_1[0]} suffered list is {team_1[5]}")
    print(f"For team {team_2[0]} Tier list is {team_2[3]}")
    print(f"For team {team_2[0]} scored list is {team_2[4]}")
    print(f"For team {team_2[0]} suffered list is {team_2[5]}")

    model1_team1, model1_team2, model2_team1, model2_team2,team1_goals_predicted_ln,team1_suffered_predicted_ln,team2_goals_predicted_ln,team2_suffered_predicted_ln,team1_goals_predicted_pr,team1_suffered_predicted_pr,team2_goals_predicted_pr,team2_suffered_predicted_pr,lr_mse_scored,pr_mse_scored,lr_mse_suffered,pr_mse_suffered,lr_mae_scored,pr_mae_scored,lr_mae_suffered,pr_mae_suffered,ln_graph1,ln_graph2,ps_graph1,ps_graph2 = predict_function.predict_scores(team_1, team_2)
    print (model1_team1, model1_team2, model2_team1, model2_team2,lr_mse_scored,pr_mse_scored,lr_mse_suffered,pr_mse_suffered,lr_mae_scored,pr_mae_scored,lr_mae_suffered,pr_mae_suffered,ln_graph1,ln_graph2,ps_graph1,ps_graph2)
    return (model1_team1, model1_team2, model2_team1, model2_team2,team_1[2],team_2[2],team1_goals_predicted_ln,team1_suffered_predicted_ln,team2_goals_predicted_ln,team2_suffered_predicted_ln,team1_goals_predicted_pr,team1_suffered_predicted_pr,team2_goals_predicted_pr,team2_suffered_predicted_pr,lr_mse_scored,pr_mse_scored,lr_mse_suffered,pr_mse_suffered,lr_mae_scored,pr_mae_scored,lr_mae_suffered,pr_mae_suffered,ln_graph1,ln_graph2,ps_graph1,ps_graph2)


#team_1=["Atletico-MG",str(1062),0,[],[],[]]
#team_2=["Palmeiras",str(121),0,[],[],[]]
#print (resolve_teams(team_1,team_2))