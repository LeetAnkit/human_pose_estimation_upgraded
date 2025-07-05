# FILE: firebase_helper.py

import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
from datetime import datetime

# Initialize only once
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase_key.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

def save_workout_to_firebase(reps, accuracy, exercise="Squats"):
    data = {
        'timestamp': datetime.now(),
        'exercise': exercise,
        'reps': reps,
        'accuracy': accuracy
    }
    db.collection("workouts").add(data)

def get_all_workouts():
    docs = db.collection("workouts").order_by("timestamp").stream()
    data = []
    for doc in docs:
        d = doc.to_dict()
        d['timestamp'] = d['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        data.append(d)
    return pd.DataFrame(data)
