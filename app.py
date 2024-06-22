from flask import Flask,request,jsonify,send_from_directory
from goals_getvalues import resolve_teams
import os
import glob

app = Flask(__name__)

port = int(os.environ.get("PORT", 5000))
@app.route('/predict',methods=['POST'])
def predict():
    data = request.get_json(force=True)
    print (data)
    
    team_1_name = data.get('clubName1')
    team_1_ID = data.get('clubID1')
    team_2_name = data.get('clubName2')
    team_2_ID = data.get('clubID2')
    print (team_1_name,team_1_ID,team_2_name,team_2_ID)
    team_1=[str(team_1_name),str(team_1_ID),0,[],[],[]]
    team_2=[str(team_2_name),str(team_2_ID),0,[],[],[]]
    
    if not team_1_name or not team_2_name:
        return jsonify ({'error': 'Both team_1 and team_2 must be provided'}), 400
    
    try :
        model1_team1,model1_team2,model2_team1,model2_team2,tier1,tier2,team1_goals_predicted_ln,team1_suffered_predicted_ln,team2_goals_predicted_ln,team2_suffered_predicted_ln,team1_goals_predicted_pr,team1_suffered_predicted_pr,team2_goals_predicted_pr,team2_suffered_predicted_pr,lr_mse_scored,pr_mse_scored,lr_mse_suffered,pr_mse_suffered,lr_mae_scored,pr_mae_scored,lr_mae_suffered,pr_mae_suffered,ln_graph1,ln_graph2,ps_graph1,ps_graph2 = resolve_teams (team_1,team_2)
        return jsonify ({
            'model1_team1':model1_team1,
            'model1_team2':model1_team2,
            'model2_team1':model2_team1,
            'model2_team2':model2_team2,
            'tier1':tier1,
            'tier2':tier2,
            'ln_graph1':ln_graph1,
            'ln_graph2':ln_graph2,
            'ps_graph1':ps_graph1,
            'ps_graph2':ps_graph2,
            'lr_mse_scored':lr_mse_scored,
            'pr_mse_scored':pr_mse_scored,
            'lr_mse_suffered':lr_mse_suffered,
            'pr_mse_suffered':pr_mse_suffered,
            'lr_mae_scored':lr_mae_scored,
            'pr_mae_scored':pr_mae_scored,
            'lr_mae_suffered':lr_mae_suffered,
            'pr_mae_suffered':pr_mae_suffered,
            'team1_goals_predicted_ln':team1_goals_predicted_ln,
            'team1_suffered_predicted_ln':team1_suffered_predicted_ln,
            'team2_goals_predicted_ln':team2_goals_predicted_ln,
            'team2_suffered_predicted_ln':team2_suffered_predicted_ln,
            'team1_goals_predicted_pr':team1_goals_predicted_pr,
            'team1_suffered_predicted_pr':team1_suffered_predicted_pr,
            'team2_goals_predicted_pr':team2_goals_predicted_pr,
            'team2_suffered_predicted_pr':team2_suffered_predicted_pr,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500    
    
@app.route('/image/<filename>')
def get_image(filename):
    # Check if the file exists in the results_graph directory
    image_path = os.path.join('results_graph', filename)
    if os.path.exists(image_path):
        # Return the image file
        return send_from_directory('results_graph', filename)
    else:
        return jsonify({'error': 'Image not found'}), 404

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/')
def index():
    return 'Welcome to the Football Predictor!'

# Set your secret API key
API_KEY = 'Henriquemsantos10'

def check_api_key(api_key):
    return api_key == API_KEY

@app.route('/delete_league', methods=['GET'])
def delete_league():
    api_key = request.headers.get('x-api-key')
    if not api_key or not check_api_key(api_key):
        return jsonify(message="This is a restricted endpoint")
    else:
        # Path to the directory containing the files
        folder_path = 'last_games'
        
        # Get list of all files in the folder
        files = glob.glob(os.path.join(folder_path, '*'))
        
        # Delete each file
        for file in files:
            os.remove(file)
        
        return jsonify(message="All files deleted"), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port)
    
    
    