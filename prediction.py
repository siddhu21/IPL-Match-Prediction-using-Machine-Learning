import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from scipy.stats import uniform
import seaborn as sns
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
from xgboost import XGBClassifier,XGBRFClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import scipy.stats as stats
import warnings
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


warnings.filterwarnings("ignore")
from sklearn import metrics
import xgboost as xgb  

#from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
#from sklearn.linear_model import LogisticRegression
#from lightgbm import LGBMClassifier
#from sklearn.linear_model import LogisticRegression
#from sklearn.tree import DecisionTreeClassifier

df = pd.read_excel("IPL_Data_with_First_Inns_Score.xlsx")

#df = pd.read_excel("IPLDataChatbot.xlsx")
df.drop(['id', 'season', 'date','result', 'win_by_runs', 'win_by_wickets','dl_applied', 'player_of_match', 'city'],axis=1, inplace = True)
#print(df.shape)

df['toss_decision'] = df['toss_decision'].str.replace("Bowl", 'bowl')
df['toss_decision'] = df['toss_decision'].str.replace("Bat", 'bat')

#Define Consistent Teams

const_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
              'Mumbai Indians', 'Punjab Kings', 'Royal Challengers Bangalore',
              'Delhi Capitals', 'Sunrisers Hyderabad', 'Gujarat Titans', 'Lucknow Super Giants']

#print(f'Before Removing Inconsistent Teams : {df.shape}')
df = df[(df['team_batting_first'].isin(const_teams)) & (df['team_batting_second'].isin(const_teams))]
#print(f'After Removing Irrelevant Columns : {df.shape}')
#print(f"Consistent Teams : \n{df['team_batting_first'].unique()}")

# Defining Powerplay Runrate
df['pp_runrate_battingfirst'] = round(df['team_batting_first_ppscore']/6,2)
df['pp_runrate_battingsecond'] = round(df['team_batting_second_ppscore']/6,2)

# Defining Powerplay WicketFallrate
df['pp_wicketfallrate_battingfirst'] = round(df['team_batting_first_ppwickets']/6,2)
df['pp_wicketfallrate_battingsecond'] = round(df['team_batting_second_ppwickets']/6,2)

# Defining Required Runrate
df['reqrunrate_battingsecond'] = round((df['target']-df['team_batting_second_ppscore'])/14,2)

# Average Powerplay Score based on venue

venue_ppscore_avg = df.groupby('venue').agg(
    avg_ppscore_batting_first_venue=('team_batting_first_ppscore', 'mean'),
    avg_ppscore_batting_second_venue=('team_batting_second_ppscore', 'mean')
).reset_index()

# Merge the average Powerplay scores back to the original DataFrame
df = pd.merge(df, venue_ppscore_avg, on='venue', how='left')

df['avg_ppscore_venue'] = round((df['avg_ppscore_batting_first_venue'] + df['avg_ppscore_batting_second_venue']) / 2,2)

df.drop(["avg_ppscore_batting_first_venue", "avg_ppscore_batting_second_venue"],axis=1, inplace=True)


#print("Columns in DataFrame:", df.columns)

df.winner[df.winner == df.team_batting_first] = 0
df.winner[df.winner == df.team_batting_second] = 1  #Chasing Team

df['winner'] = df['winner'].astype('int64')

#print("Winner Disttribution:", df['winner'].value_counts(normalize=True))

#filterdata = df.to_excel("filterdata.xlsx")

#Model Building

# Separate features and target variable
X = df.drop("winner", axis=1)
y = df["winner"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# from sklearn.model_selection import TimeSeriesSplit

# tscv = TimeSeriesSplit(n_splits=10)  # 5 different train-test splits

# for train_index, test_index in tscv.split(X):
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Now train your LightGBM model on X_train, y_train


print("X_train Shape", X_train.shape)
print("X_test Shape", X_test.shape)
print(X_train.head(1))
print(X_train.columns)

#print("Avg. PP Score by Venue", X[X['avg_ppscore_venue']][0])

# Define numerical and categorical features
numerical_features = ['team_batting_first_ppscore', 'team_batting_first_ppwickets', 'target',
                      'team_batting_second_ppscore', 'team_batting_second_ppwickets',
                      'pp_runrate_battingfirst', 'pp_runrate_battingsecond',
                      'pp_wicketfallrate_battingfirst', 'pp_wicketfallrate_battingsecond',
                      'reqrunrate_battingsecond', 'avg_ppscore_venue']

#city
categorical_features = ['team_batting_first', 'team_batting_second', 'toss_winner', 'toss_decision',
                        'venue']

#Create preprocessing pipelines for numerical and categorical data
numerical_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing pipelines
preprocessor = ColumnTransformer([
    ('numerical', numerical_pipeline, numerical_features),
    ('categorical', categorical_pipeline, categorical_features)
    
])

#print(preprocessor)

model = Pipeline([
    ('preprocessor', preprocessor),

    ('classifier', XGBClassifier(n_estimators = 100, #202
                                 random_state =42,
                                 max_depth = 4,
                                 colsample_bytree = 0.5621780831931538,
                                 learning_rate=0.15639878836228102,
                                 subsample = 0.9592488919658672,
                                 min_child_weight = 5,
                                 gamma = 4.75357153204958,
                                 reg_alpha = 0.44583275285359114,
                                 reg_lambda = 0.09997491581800289,
                                 objective='binary:logistic'))
])

#
# ('classifier', XGBClassifier(n_estimators = 100, #202
#                                  random_state =42,
#                                  max_depth = 4,
#                                  colsample_bytree = 0.5621780831931538,
#                                  learning_rate=0.15639878836228102,
#                                  subsample = 0.9592488919658672,
#                                  min_child_weight = 5,
#                                  gamma = 4.75357153204958,
#                                  reg_alpha = 0.44583275285359114,
#                                  reg_lambda = 0.09997491581800289,
#                                  objective='binary:logistic'))

# model = Pipeline([
#     ('preprocessor', preprocessor),
#   ('classifier', SVC(kernel='linear', random_state=42, gamma = 0.017587828927856153, C=6.785643618422824))
# ])

#Train the model

model.fit(X_train, y_train)

# #Make predictions
predictions = model.predict(X_test)

# Generate confusion matrix
cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:\n", cm)

#Visualize confusion matrix

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap='Blues')

plt.show()


# Evaluate the model
accuracy = round(accuracy_score(y_test, predictions),2)
print("Accuracy:", accuracy)

f1 = round(f1_score(y_test, predictions),2)
print("f1 Score:", f1)

recall = round(recall_score(y_test, predictions),2)
print("recall", recall)

class_rep = classification_report(y_test, predictions)
print("Classification Report", class_rep)

#pickle.dump(model, open('model_08_03_pp%.pkl', 'wb'))
#pickle.dump(model, open('model_22_03_ppw.pkl', 'wb'))
#pickle.dump(model, open('model_01_06_25_ppw.pkl', 'wb'))
#print("Pickle File Saved!")

######################
