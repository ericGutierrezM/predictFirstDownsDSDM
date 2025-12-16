from flask import Flask, request, jsonify, send_file
import firstDown
import pandas as pd
import joblib

pipeline = joblib.load("../first_down_model.pkl")
model = pipeline["model"]
scaler = pipeline["scaler"]
encoder = pipeline["encoder"]
num_cols = pipeline["num_cols"]
one_hot_cols = pipeline["one_hot_cols"]
wind_value = pipeline.get("wind_value", None)
temp_value = pipeline.get("temp_value", None)


app = Flask(__name__)

@app.route('/')
def index():
    return send_file('../docs/index.html')

@app.route('/header.html')
def header():
    return send_file('../docs/header.html')

@app.route('/footer.html')
def footer():
    return send_file('../docs/footer.html')

@app.route('/pages/documentation.md')
def docs_md():
    return send_file('../docs/pages/documentation.md')

@app.route('/img/daniel_campos.jpg')
def img_daniel():
    return send_file('../docs/img/daniel_campos.jpg')

@app.route('/img/eric_gutierrez.jpg')
def img_eric():
    return send_file('../docs/img/eric_gutierrez.jpg')

@app.route('/img/field.jpg')
def img_field():
    return send_file('../docs/img/field.jpg')

@app.route('/api.html')
def api():
    return send_file('../docs/api.html')

@app.route('/playground.html')
def playground():
    return send_file('../docs/playground.html')

@app.route('/documentation.html')
def documentation():
    return send_file('../docs/documentation.html')


'''def predict():
    data = request.get_json()
    X = pd.DataFrame([data])

    # Handle NaNs using TRAIN values
    X, _ = firstDown.preprocessing.deal_nan.replace_nan(
        X, 'wind', method='num', num=wind_value
    )
    X, _ = firstDown.preprocessing.deal_nan.replace_nan(
        X, 'temp', method='num', num=temp_value
    )

    # Scale
    X = firstDown.preprocessing.scale.scaler_transform(
        X, num_cols=num_cols, scaler=scaler
    )

    # Encode
    X = firstDown.feature_engineering.encode.one_hot_transform(
        X, cols=one_hot_cols, encoder=encoder
    )

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    return jsonify({
        "prediction": y_pred.tolist(),
        "probability": y_prob.tolist()
    })'''

@app.route('/api/predict', methods=['POST']) 
@app.route('/api/predict', methods=['POST']) 
def predict():
    try:
        data = request.get_json()
        if isinstance(data, dict):
            data = [data]
        
        X = pd.DataFrame(data)

        # Fill numeric NaNs
        for col in num_cols:
            if col not in X.columns:
                X[col] = 0
        X[num_cols] = X[num_cols].fillna(0)

        # Fill categorical NaNs and ensure string type
        for col in one_hot_cols:
            if col not in X.columns or pd.isna(X[col].iloc[0]):
                X[col] = pipeline[f"{col}_mode"]  # must exist in training
            X[col] = X[col].astype(str)

        # Scale numeric columns
        X_scaled = scaler.transform(X[num_cols])
        X[num_cols] = X_scaled

        # One-hot encode categorical columns safely
        encoded = encoder.transform(X[one_hot_cols])
        encoded_df = pd.DataFrame(
            encoded,
            columns=encoder.get_feature_names_out(one_hot_cols),
            index=X.index
        )
        X = pd.concat([X.drop(one_hot_cols, axis=1), encoded_df], axis=1)

        # Predict
        y_prob = model.predict_proba(X)[:, 1]


        return jsonify({"probability": y_prob.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)