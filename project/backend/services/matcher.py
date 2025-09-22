import pandas as pd

# Load datasets
symptom_disease = pd.read_csv("data/symptom_disease.csv")
symptom_desc = pd.read_csv("data/symptom_Description.csv")
symptom_prec = pd.read_csv("data/symptom_precaution.csv")

def match_symptoms(symptoms: list):
    """
    Placeholder logic:
    Matches first disease found with symptom overlap.
    """
    for _, row in symptom_disease.iterrows():
        disease_symptoms = row["Symptoms"].split(",")
        if any(sym in disease_symptoms for sym in symptoms):
            return row["Disease"], 0.75  # dummy confidence
    return "Unknown", 0.0

def get_description(disease: str):
    row = symptom_desc[symptom_desc["Disease"] == disease]
    if not row.empty:
        return row["Description"].values[0]
    return "No description available."

def get_precautions(disease: str):
    row = symptom_prec[symptom_prec["Disease"] == disease]
    if not row.empty:
        return [row[f"Precaution_{i}"].values[0] for i in range(1, 5)]
    return []
