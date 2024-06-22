import numpy as np
from utl import train_test_split
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os


def predict_scores (team_1,team_2):
    # Data for team1 and team2
    
    team1_name=team_1[0]
    team1_tier = team_1[3]
    team1_goals_scored = team_1[4]
    team1_goals_suffered = team_1[5]

    team2_name=team_2[0]
    team2_tier = team_2[3]
    team2_goals_scored = team_2[4]
    team2_goals_suffered = team_2[5]

    # Create DataFrames for easier manipulation
    team1_df = pd.DataFrame({
        'tier': team1_tier,
        'goals_scored': team1_goals_scored,
        'goals_suffered': team1_goals_suffered
    })

    team2_df = pd.DataFrame({
        'tier': team2_tier,
        'goals_scored': team2_goals_scored,
        'goals_suffered': team2_goals_suffered
    })
    
    train1_df,test1_df = train_test_split(team1_df)
    train2_df,test2_df = train_test_split(team2_df)


    # Averaging goals for each tier
    team1_avg = train1_df.groupby('tier').mean().reset_index()
    team2_avg = train2_df.groupby('tier').mean().reset_index()

    team1_avg = team1_avg.dropna(subset=['goals_scored', 'goals_suffered'])
    team2_avg = team2_avg.dropna(subset=['goals_scored', 'goals_suffered'])
    
    
    # Prepare the training data
    X_team1 = team1_avg[['tier']]
    y_team1_scored = team1_avg['goals_scored']
    y_team1_suffered = team1_avg['goals_suffered']

    X_team2 = team2_avg[['tier']]
    y_team2_scored = team2_avg['goals_scored']
    y_team2_suffered = team2_avg['goals_suffered']

    # Initialize and train the models
    model_scored_team1 = LinearRegression().fit(X_team1, y_team1_scored)
    model_suffered_team1 = LinearRegression().fit(X_team1, y_team1_suffered)

    model_scored_team2 = LinearRegression().fit(X_team2, y_team2_scored)
    model_suffered_team2 = LinearRegression().fit(X_team2, y_team2_suffered)
    
    pr_scored_team1 = PoissonRegressor().fit(X_team1, y_team1_scored)
    pr_suffered_team1 = PoissonRegressor().fit(X_team1, y_team1_suffered)

    pr_scored_team2 = PoissonRegressor().fit(X_team2, y_team2_scored)
    pr_suffered_team2 = PoissonRegressor().fit(X_team2, y_team2_suffered)
    lr_mse_scored,pr_mse_scored,lr_mse_suffered,pr_mse_suffered,lr_mae_scored,pr_mae_scored,lr_mae_suffered,pr_mae_suffered = 0,0,0,0,0,0,0,0
    
    try:
    # Evaluate Models
        X_test1 = test1_df[['tier']]
        y_test1_scored = test1_df['goals_scored']
        y_test1_suffered = test1_df['goals_suffered']

        X_test2 = test2_df[['tier']]
        y_test2_scored = test2_df['goals_scored']
        y_test2_suffered = test2_df['goals_suffered']

        # Predictions on test data
        lr_y_test1_scored_pred = model_scored_team1.predict(X_test1)
        lr_y_test1_suffered_pred = model_suffered_team1.predict(X_test1)
        lr_y_test2_scored_pred = model_scored_team2.predict(X_test2)
        lr_y_test2_suffered_pred = model_suffered_team2.predict(X_test2)

        pr_y_test1_scored_pred = pr_scored_team1.predict(X_test1)
        pr_y_test1_suffered_pred = pr_suffered_team1.predict(X_test1)
        pr_y_test2_scored_pred = pr_scored_team2.predict(X_test2)
        pr_y_test2_suffered_pred = pr_suffered_team2.predict(X_test2)

        lr_mse_scored = mean_squared_error(np.concatenate([y_test1_scored, y_test2_scored]), 
                                           np.concatenate([lr_y_test1_scored_pred, lr_y_test2_scored_pred]))
        lr_mae_scored = mean_absolute_error(np.concatenate([y_test1_scored, y_test2_scored]), 
                                            np.concatenate([lr_y_test1_scored_pred, lr_y_test2_scored_pred]))

        pr_mse_scored = mean_squared_error(np.concatenate([y_test1_scored, y_test2_scored]), 
                                           np.concatenate([pr_y_test1_scored_pred, pr_y_test2_scored_pred]))
        pr_mae_scored = mean_absolute_error(np.concatenate([y_test1_scored, y_test2_scored]), 
                                            np.concatenate([pr_y_test1_scored_pred, pr_y_test2_scored_pred]))

        lr_mse_suffered = mean_squared_error(np.concatenate([y_test1_suffered, y_test2_suffered]), 
                                             np.concatenate([lr_y_test1_suffered_pred, lr_y_test2_suffered_pred]))
        lr_mae_suffered = mean_absolute_error(np.concatenate([y_test1_suffered, y_test2_suffered]), 
                                              np.concatenate([lr_y_test1_suffered_pred, lr_y_test2_suffered_pred]))

        pr_mse_suffered = mean_squared_error(np.concatenate([y_test1_suffered, y_test2_suffered]), 
                                             np.concatenate([pr_y_test1_suffered_pred, pr_y_test2_suffered_pred]))
        pr_mae_suffered = mean_absolute_error(np.concatenate([y_test1_suffered, y_test2_suffered]), 
                                              np.concatenate([pr_y_test1_suffered_pred, pr_y_test2_suffered_pred]))

        print(f"Linear Regression MSE (Goals Scored): {lr_mse_scored:.2f}, MAE (Goals Scored): {lr_mae_scored:.2f}")
        print(f"Poisson Regression MSE (Goals Scored): {pr_mse_scored:.2f}, MAE (Goals Scored): {pr_mae_scored:.2f}")
        print(f"Linear Regression MSE (Goals suffered): {lr_mse_suffered:.2f}, MAE (Goals Suffered): {lr_mae_suffered:.2f}")
        print(f"Poisson Regression MSE (Goals suffered): {pr_mse_suffered:.2f}, MAE (Goals Suffered): {pr_mae_suffered:.2f}")
    
    except:
        print ("None values present")
        
    ####################################
    # Averaging goals for each tier
    team1_avg = team1_df.groupby('tier').mean().reset_index()
    team2_avg = team2_df.groupby('tier').mean().reset_index()

    team1_avg = team1_avg.dropna(subset=['goals_scored', 'goals_suffered'])
    team2_avg = team2_avg.dropna(subset=['goals_scored', 'goals_suffered'])
    
    
    # Prepare the training data
    X_team1 = team1_avg[['tier']]
    y_team1_scored = team1_avg['goals_scored']
    y_team1_suffered = team1_avg['goals_suffered']

    X_team2 = team2_avg[['tier']]
    y_team2_scored = team2_avg['goals_scored']
    y_team2_suffered = team2_avg['goals_suffered']

    # Initialize and train the models
    model_scored_team1 = LinearRegression().fit(X_team1, y_team1_scored)
    model_suffered_team1 = LinearRegression().fit(X_team1, y_team1_suffered)

    model_scored_team2 = LinearRegression().fit(X_team2, y_team2_scored)
    model_suffered_team2 = LinearRegression().fit(X_team2, y_team2_suffered)
    
    pr_scored_team1 = PoissonRegressor().fit(X_team1, y_team1_scored)
    pr_suffered_team1 = PoissonRegressor().fit(X_team1, y_team1_suffered)

    pr_scored_team2 = PoissonRegressor().fit(X_team2, y_team2_scored)
    pr_suffered_team2 = PoissonRegressor().fit(X_team2, y_team2_suffered)
    

    # Predict the next game between team1 (tier 13) and team2 (tier 15)
    next_game_team1_tier = team_2[2]
    next_game_team2_tier = team_1[2]

    team1_goals_predicted_ln = model_scored_team1.predict([[next_game_team1_tier]])[0]
    team1_suffered_predicted_ln = model_suffered_team1.predict([[next_game_team1_tier]])[0]

    team2_goals_predicted_ln = model_scored_team2.predict([[next_game_team2_tier]])[0]
    team2_suffered_predicted_ln = model_suffered_team2.predict([[next_game_team2_tier]])[0]

    team1_goals_predicted_pr = pr_scored_team1.predict([[next_game_team1_tier]])[0]
    team1_suffered_predicted_pr = pr_suffered_team1.predict([[next_game_team1_tier]])[0]

    team2_goals_predicted_pr = pr_scored_team2.predict([[next_game_team2_tier]])[0]
    team2_suffered_predicted_pr = pr_suffered_team2.predict([[next_game_team2_tier]])[0]
    
    print(f"Predicted goals for linear model {team1_name}: {team1_goals_predicted_ln:.2f} scored, {team1_suffered_predicted_ln:.2f} suffered")
    print(f"Predicted goals for linear model {team2_name}: {team2_goals_predicted_ln:.2f} scored, {team2_suffered_predicted_ln:.2f} suffered")
    
    print(f"Predicted goals for poisson model {team1_name}: {team1_goals_predicted_pr:.2f} scored, {team1_suffered_predicted_pr:.2f} suffered")
    print(f"Predicted goals for poisson model {team2_name}: {team2_goals_predicted_pr:.2f} scored, {team2_suffered_predicted_pr:.2f} suffered")

    # Predict the final score by taking the average of the predictions
    final_team1_goals_ln = (team1_goals_predicted_ln + team2_suffered_predicted_ln) / 2
    final_team2_goals_ln = (team2_goals_predicted_ln + team1_suffered_predicted_ln) / 2
    
    final_team1_goals_pr = (team1_goals_predicted_pr + team2_suffered_predicted_pr) / 2
    final_team2_goals_pr = (team2_goals_predicted_pr + team1_suffered_predicted_pr) / 2

    print(f"Final predicted score linear model: {team1_name} {final_team1_goals_ln:.2f} - {final_team2_goals_ln:.2f} {team2_name}")
    
    print(f"Final predicted score poisson model: {team1_name} {final_team1_goals_pr:.2f} - {final_team2_goals_pr:.2f} {team2_name}")
    

    # Ensure the 'results_graph' directory exists
    if not os.path.exists('results_graph'):
        os.makedirs('results_graph')
    
    # Linear Regression for team1
    plt.figure(figsize=(7, 5))
    plt.scatter(team1_avg['tier'], team1_avg['goals_scored'], color='blue', label='Goals Scored')
    plt.scatter(team1_avg['tier'], team1_avg['goals_suffered'], color='red', label='Goals Suffered')
    plt.plot(team1_avg['tier'], model_scored_team1.predict(X_team1), color='blue', linestyle='--', label='Scored Regression')
    plt.plot(team1_avg['tier'], model_suffered_team1.predict(X_team1), color='red', linestyle='--', label='Suffered Regression')
    plt.title(f'{team1_name} Goals (Linear Regression)')
    plt.xlabel('Tier')
    plt.ylabel('Goals')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('results_graph', f'{team1_name}_linear_regression.png'))
    plt.close()
    
    # Linear Regression for team2
    plt.figure(figsize=(7, 5))
    plt.scatter(team2_avg['tier'], team2_avg['goals_scored'], color='blue', label='Goals Scored')
    plt.scatter(team2_avg['tier'], team2_avg['goals_suffered'], color='red', label='Goals Suffered')
    plt.plot(team2_avg['tier'], model_scored_team2.predict(X_team2), color='blue', linestyle='--', label='Scored Regression')
    plt.plot(team2_avg['tier'], model_suffered_team2.predict(X_team2), color='red', linestyle='--', label='Suffered Regression')
    plt.title(f'{team2_name} Goals (Linear Regression)')
    plt.xlabel('Tier')
    plt.ylabel('Goals')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('results_graph', f'{team2_name}_linear_regression.png'))
    plt.close()
    
    # Poisson Regression for team1
    plt.figure(figsize=(7, 5))
    plt.scatter(team1_avg['tier'], team1_avg['goals_scored'], color='blue', label='Goals Scored')
    plt.scatter(team1_avg['tier'], team1_avg['goals_suffered'], color='red', label='Goals Suffered')
    plt.plot(team1_avg['tier'], pr_scored_team1.predict(X_team1), color='blue', linestyle='--', label='Scored Regression')
    plt.plot(team1_avg['tier'], pr_suffered_team1.predict(X_team1), color='red', linestyle='--', label='Suffered Regression')
    plt.title(f'{team1_name} Goals (Poisson Regression)')
    plt.xlabel('Tier')
    plt.ylabel('Goals')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('results_graph', f'{team1_name}_poisson_regression.png'))
    plt.close()
    
    # Poisson Regression for team2
    plt.figure(figsize=(7, 5))
    plt.scatter(team2_avg['tier'], team2_avg['goals_scored'], color='blue', label='Goals Scored')
    plt.scatter(team2_avg['tier'], team2_avg['goals_suffered'], color='red', label='Goals Suffered')
    plt.plot(team2_avg['tier'], pr_scored_team2.predict(X_team2), color='blue', linestyle='--', label='Scored Regression')
    plt.plot(team2_avg['tier'], pr_suffered_team2.predict(X_team2), color='red', linestyle='--', label='Suffered Regression')
    plt.title(f'{team2_name} Goals (Poisson Regression)')
    plt.xlabel('Tier')
    plt.ylabel('Goals')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('results_graph', f'{team2_name}_poisson_regression.png'))
    plt.close()

    return (final_team1_goals_ln,final_team2_goals_ln,final_team1_goals_pr,final_team2_goals_pr,\
        team1_goals_predicted_ln,team1_suffered_predicted_ln,team2_goals_predicted_ln,team2_suffered_predicted_ln,\
        team1_goals_predicted_pr,team1_suffered_predicted_pr,team2_goals_predicted_pr,team2_suffered_predicted_pr,\
        lr_mse_scored,pr_mse_scored,lr_mse_suffered,pr_mse_suffered,lr_mae_scored,pr_mae_scored,lr_mae_suffered,pr_mae_suffered,\
        f'{team1_name}_linear_regression.png',f'{team2_name}_linear_regression.png',f'{team1_name}_poisson_regression.png',f'{team2_name}_poisson_regression.png')