import os
import pickle
import numpy as np
import keras
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load model safely
MODEL_PATH = "ANN_model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("Model loaded successfully via pickle.")
    except Exception as e:
        print(f"Error loading model via pickle: {e}")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ANN Model Predictor</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        body {
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            background: linear-gradient(135deg, #0f172a, #1e1b4b, #311042);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
            padding: 20px;
        }
        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.07);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 20px;
            padding: 30px;
            width: 100%;
            max-width: 650px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            color: #ffffff;
        }
        h2 {
            text-align: center;
            margin-bottom: 24px;
            font-weight: 600;
            letter-spacing: 0.5px;
            color: #f1f5f9;
        }
        .input-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .input-group {
            display: flex;
            flex-direction: column;
        }
        .input-group label {
            font-size: 0.82rem;
            margin-bottom: 6px;
            color: #cbd5e1;
            font-weight: 500;
        }
        .input-group input {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            padding: 10px 12px;
            color: #ffffff;
            outline: none;
            transition: all 0.3s ease;
            font-size: 0.95rem;
        }
        .input-group input:focus {
            border-color: #a855f7;
            box-shadow: 0 0 10px rgba(168, 85, 247, 0.4);
            background: rgba(255, 255, 255, 0.15);
        }
        .btn-submit {
            width: 100%;
            padding: 12px;
            border: none;
            border-radius: 10px;
            background: linear-gradient(90deg, #6366f1, #a855f7);
            color: #ffffff;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(168, 85, 247, 0.4);
        }
        .result-box {
            margin-top: 20px;
            padding: 15px;
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            text-align: center;
            display: none;
        }
        .result-title {
            font-size: 0.9rem;
            color: #94a3b8;
        }
        .result-value {
            font-size: 1.4rem;
            font-weight: bold;
            margin-top: 5px;
            color: #38bdf8;
        }
    </style>
</head>
<body>
    <div class="glass-card">
        <h2>ANN Prediction Interface</h2>
        <form id="predictionForm">
            <div class="input-grid">
                {% for i in range(1, 11) %}
                <div class="input-group">
                    <label for="f{{i}}">Feature {{i}}</label>
                    <input type="number" step="any" id="f{{i}}" name="f{{i}}" required placeholder="0.0">
                </div>
                {% endfor %}
            </div>
            <button type="submit" class="btn-submit">Predict Result</button>
        </form>

        <div id="resultBox" class="result-box">
            <div class="result-title">Prediction Result</div>
            <div id="resultValue" class="result-value">---</div>
        </div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const inputs = [];
            for (let i = 1; i <= 10; i++) {
                inputs.push(parseFloat(document.getElementById(`f${i}`).value));
            }

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ features: inputs })
                });
                const data = await response.json();
                
                const resultBox = document.getElementById('resultBox');
                const resultValue = document.getElementById('resultValue');
                
                if (data.error) {
                    resultValue.innerText = data.error;
                    resultValue.style.color = '#f87171';
                } else {
                    resultValue.innerText = `Probability: ${data.probability} (${data.class})`;
                    resultValue.style.color = '#38bdf8';
                }
                resultBox.style.display = 'block';
            } catch (err) {
                alert('Error processing request.');
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model pickle file not loaded correctly.'}), 500

    try:
        data = request.get_json()
        features = data.get('features')
        
        if not features or len(features) != 10:
            return jsonify({'error': 'Input array must contain exactly 10 values.'}), 400

        input_data = np.array([features], dtype=np.float32)
        prediction = model.predict(input_data)
        prob = float(prediction[0][0])
        predicted_class = "Class 1" if prob >= 0.5 else "Class 0"

        return jsonify({
            'probability': round(prob, 4),
            'class': predicted_class
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)




import os
import pickle
import numpy as np
import keras  # Must be imported before unpickling Keras models
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load model
MODEL_PATH = "ANN_model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Model load error: {e}")
