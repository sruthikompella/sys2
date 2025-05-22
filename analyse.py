from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

app = Flask(__name__)

HAZARD_THRESHOLDS = {
    'drought': {'precip_percentile': 20, 'min_duration': 5},
    'heavy_rain': {'precip_percentile': 80, 'min_days': 2}
}

# Fake sample data
data = {
    'date': pd.to_datetime(pd.date_range(start='2023-01-01', periods=100)),
    'precipitation': np.random.rand(100) * 10
}
df = pd.DataFrame(data)

def detect_hazards(df, hazard_type):
    hazards = []
    if hazard_type == 'drought':
        threshold = np.percentile(df['precipitation'], HAZARD_THRESHOLDS['drought']['precip_percentile'])
        below = df['precipitation'] < threshold
        groups = below.ne(below.shift()).cumsum()
        counts = groups.where(below).value_counts()
        for group, days in counts.items():
            if days >= HAZARD_THRESHOLDS['drought']['min_duration']:
                group_dates = df[groups == group]['date']
                hazards.append({
                    'type': 'drought',
                    'start_date': group_dates.min().strftime('%Y-%m-%d'),
                    'end_date': group_dates.max().strftime('%Y-%m-%d'),
                    'duration': int(days),
                    'intensity': float(df[groups == group]['precipitation'].mean())
                })

    elif hazard_type == 'heavy_rain':
        threshold = np.percentile(df['precipitation'], HAZARD_THRESHOLDS['heavy_rain']['precip_percentile'])
        above = df['precipitation'] > threshold
        groups = above.ne(above.shift()).cumsum()
        counts = groups.where(above).value_counts()
        for group, days in counts.items():
            if days >= HAZARD_THRESHOLDS['heavy_rain']['min_days']:
                group_dates = df[groups == group]['date']
                hazards.append({
                    'type': 'heavy_rain',
                    'start_date': group_dates.min().strftime('%Y-%m-%d'),
                    'end_date': group_dates.max().strftime('%Y-%m-%d'),
                    'duration': int(days),
                    'intensity': float(df[groups == group]['precipitation'].mean())
                })
    return hazards

@app.route("/", methods=["POST"])
def analyze():
    body = request.json
    hazard_type = body.get("hazard_type", "heavy_rain")
    results = detect_hazards(df, hazard_type)
    return jsonify(results)

if __name__ == "__main__":
    app.run()
