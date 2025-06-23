from flask import Flask, request, jsonify
from flask import render_template, redirect, session
import joblib
import time
import pandas as pd
import numpy as np
import logging
from flask_cors import CORS, cross_origin
from flask_executor import Executor
import warnings
from azure.storage.blob import BlobServiceClient
warnings.filterwarnings('ignore')
#from prediction import get_powerplay_winner_adjusted

# Azure Blob Storage connection string
AZURE_STORAGE_CONNECTION_STRING = ""
CONTAINER_NAME = "data"
#EXCEL_FILE_NAME = "IPLDataChatbot.xlsx"
EXCEL_FILE_NAME_1 = "IPL_Data_with_First_Inns_Score.xlsx"

#data = pd.read_excel("IPL_Data_with_First_Inns_Score.xlsx")


#model = joblib.load('model151024.pkl')  # Replace 'your_model.pkl' with the path to your trained model file
model = joblib.load('model_22_03_ppw.pkl')
#model = joblib.load('model_01_06_25_ppw.pkl')
 
def download_excel_from_blob():
    # Create a BlobServiceClient object using the connection string
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)

    # Get a container client
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)

    # Download the Excel file from Blob Storage
    blob_client = container_client.get_blob_client(EXCEL_FILE_NAME_1)
    with open(EXCEL_FILE_NAME_1, "wb") as my_blob:
        download_stream = blob_client.download_blob()
        my_blob.write(download_stream.readall())

def read_excel_file():
    # Read the Excel file into a pandas DataFrame
    df = pd.read_excel(EXCEL_FILE_NAME_1)
    return df


download_excel_from_blob()
df = read_excel_file() # Whole Data 
data = read_excel_file() # Venue Data Avg
#print(df.head(1))


def preprocess_new_data(new_data):
    #predefined_cities = df['city'].unique()
    predefined_venues = df['venue'].unique()
    predefined_batting_teams = df['team_batting_first'].unique()
    predefined_bowling_teams = df['team_batting_second'].unique()
    predefined_toss_winner = df['toss_winner'].unique()
    predefined_toss_decision = df['toss_decision'].unique()
    #predefined_winner = df['winner']

    # Validate city and venue inputs
    # if new_data['city'].values[0] not in predefined_cities:
    #     raise ValueError("Invalid city. Please enter a city from the predefined list.")
    if new_data['venue'].values[0] not in predefined_venues:
        raise ValueError("Invalid venue. Please enter a venue from the predefined list.")
    if new_data['team_batting_first'].values[0] not in predefined_batting_teams:
        raise ValueError("Invalid Team. Please enter a team from the predefined list.")
    if new_data['team_batting_second'].values[0] not in predefined_bowling_teams:
        raise ValueError("Invalid Team. Please enter a team from the predefined list.")
    if new_data['toss_winner'].values[0] not in predefined_toss_winner:
        raise ValueError("Invalid Team. Please enter a team from the predefined list.")
    if new_data['toss_decision'].values[0] not in predefined_toss_decision:
        raise ValueError("Invalid Decision. Please enter a value from the predefined list.")

    # Calculate derived features
    new_data['pp_runrate_battingfirst'] = round(new_data['team_batting_first_ppscore'] / 6, 2)
    new_data['pp_runrate_battingsecond'] = round(new_data['team_batting_second_ppscore'] / 6, 2)
    new_data['pp_wicketfallrate_battingfirst'] = round(new_data['team_batting_first_ppwickets'] / 6, 2)
    new_data['pp_wicketfallrate_battingsecond'] = round(new_data['team_batting_second_ppwickets'] / 6, 2)
    new_data['reqrunrate_battingsecond'] = round((new_data['target']-new_data['team_batting_second_ppscore'])/14,2)

    venue_ppscore_avg = data.groupby('venue').agg( #df
        avg_ppscore_batting_first_venue=('team_batting_first_ppscore', 'mean'),
        avg_ppscore_batting_second_venue=('team_batting_second_ppscore', 'mean')
    ).reset_index()


    #print(venue_ppscore_avg)
    

    #
    # # Merge the average Powerplay scores back to the original DataFrame
    new_data = pd.merge(new_data, venue_ppscore_avg, on='venue', how='left')
    #print("New Data Venue:- ",new_data)
    #
    new_data['avg_ppscore_venue'] = round((new_data['avg_ppscore_batting_first_venue'] + new_data['avg_ppscore_batting_second_venue']) / 2,2)
    
    #print(new_data['avg_ppscore_venue'])
    # new_data['powerplay_winner'] = new_data.apply(get_powerplay_winner_adjusted, axis=1)

    # venue = pd.DataFrame(venue_ppscore_avg)
    # venue.to_excel("venue_avg.xlsx")
    # print("Venue Data Saved!")

    new_data.drop(["avg_ppscore_batting_first_venue", "avg_ppscore_batting_second_venue"],axis=1, inplace=True)
    
    #new_data['powerplay_winner_match_win'] = new_data['powerplay_winner'].reindex(data.index) == data['winner']

    #new_data.to_excel("TestEx.xlsx", index=False)
    return new_data

# Prepare a new data point
# new_data_point = {
#     'city': 'Bangalore',
#     'team_batting_first': 'Royal Challengers Bangalore',
#     'team_batting_first_ppscore': 61,
#     'team_batting_first_ppwickets': 1,
#     'target': 183,
#     'team_batting_second': 'Kolkata Knight Riders',
#     'team_batting_second_ppscore': 85,
#     'team_batting_second_ppwickets': 0,
#     'toss_winner': 'Kolkata Knight Riders',
#     'toss_decision': 'bowl',
#     'venue': 'M Chinnaswamy Stadium'
# }
#
# # Convert the new data point into a DataFrame
# new_data_df = pd.DataFrame([new_data_point])
#
# # Preprocess the new data point
# new_data_processed = preprocess_new_data(new_data_df)
#
# # Use the trained model to make predictions
# prediction = model.predict_proba(new_data_processed)
#
# print("Predicted winner:", prediction)

# Load the trained model

app = Flask(__name__, template_folder='templates')
executor = Executor(app)

@app.route('/', methods=['GET'])  # route to display the home page
@cross_origin()
def homepage():
    return render_template("index.html")

@app.route('/loading', methods= ['GET', 'POST'])
@cross_origin()
def loading():
    if request.method == 'POST':
        try:
            input_data = {
                #'city': request.form['city'],
                'team_batting_first': request.form['team_batting_first'],
                'team_batting_first_ppscore': float(request.form['team_batting_first_ppscore']),
                'team_batting_first_ppwickets': int(request.form['team_batting_first_ppwickets']),
                'target': int(request.form['target']),
                'team_batting_second': request.form['team_batting_second'],
                'team_batting_second_ppscore': float(request.form['team_batting_second_ppscore']),
                'team_batting_second_ppwickets': int(request.form['team_batting_second_ppwickets']),
                'toss_winner': request.form['toss_winner'],
                'toss_decision': request.form['toss_decision'],
                'venue': request.form['venue']
            }
            print("Input Data", input_data)
            new_data_processed = preprocess_new_data(pd.DataFrame([input_data]))
            print("******&&&&&&&&&&&&&&&&&",new_data_processed)
            # Use the trained model to make predictions
            prediction = model.predict_proba(new_data_processed)
            print("Prediction", prediction)
            # Redirect to results page with prediction data
            team_batting_first = request.form.get('team_batting_first')
            team_batting_second = request.form.get('team_batting_second')

            return render_template("results.html", input_data=input_data, result_data=prediction,
                                   team_batting_first = team_batting_first, team_batting_second = team_batting_second)
        except Exception as e:
            return render_template("error.html", message="An error occurred: {}".format(str(e)))

    return render_template("loading.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5080, debug=True)