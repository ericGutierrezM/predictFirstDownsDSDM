import firstDown
import pandas as pd
import joblib

# Load data
pbp, players = firstDown.load_data.nfl_data()

dataset = firstDown.feature_engineering.build_features.get_positions(pbp, players)
dataset = firstDown.preprocessing.clean.drop_penalties(dataset, penalty_col='first_down_penalty')
dataset = firstDown.preprocessing.clean.drop_control_rows(dataset, control_col='desc')

dataset = firstDown.feature_engineering.build_features.inertia(dataset)
dataset = firstDown.feature_engineering.build_features.play_type(dataset)
dataset = firstDown.feature_engineering.build_features.defense_rush(dataset)
dataset = firstDown.feature_engineering.build_features.defense_pass(dataset)
dataset = firstDown.feature_engineering.build_features.defense_scramble(dataset)

dataset = dataset[['first_down','ydstogo','down','inertia','score_differential',
                   'play_category','rush_pos','pass_pos','rec_pos',
                   'def_vs_rush','def_vs_pass','def_vs_qb_scramble',
                   'shotgun','wp','temp','wind','roof','surface',
                   'location','half_seconds_remaining','game_half','yardline_100']]

y = dataset['first_down']
X = dataset.drop('first_down', axis=1)

# Handle NaNs
X, wind_value = firstDown.preprocessing.deal_nan.replace_nan(X, 'wind', method='num')
X, temp_value = firstDown.preprocessing.deal_nan.replace_nan(X, 'temp', method='median')

# Scaling
num_cols = ['ydstogo','inertia','score_differential','def_vs_rush',
            'def_vs_pass','def_vs_qb_scramble','wp','temp','wind',
            'half_seconds_remaining','yardline_100']

X, get_scaler = firstDown.preprocessing.scale.scaler(X, num_cols=num_cols)

# Encoding
one_hot_cols = ['down','play_category','rush_pos','pass_pos','rec_pos',
                'roof','surface','location','game_half']

encoder = firstDown.feature_engineering.encode.one_hot(X, one_hot_cols)
X = firstDown.feature_engineering.encode.one_hot_transform(X, one_hot_cols, encoder)

# Train model (do hyperparam tuning here if you want)
clf = firstDown.train.models.get_model()
clf.fit(X, y)

# Save EVERYTHING needed for inference
joblib.dump({
    "model": clf,
    "scaler": get_scaler,
    "encoder": encoder,
    "num_cols": num_cols,
    "one_hot_cols": one_hot_cols,
    "wind_value": wind_value,
    "temp_value": temp_value,
}, "first_down_model.pkl")

print("Model saved!")