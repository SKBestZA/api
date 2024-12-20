from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Union
import numpy as np
import joblib
app = FastAPI()

class HealthFeatures(BaseModel):
        features: Dict[str, Union[int, float]]

class PredictionResponse(BaseModel):
        prediction: str

@app.post("/predict", response_model=PredictionResponse)
async def predict_alzheimers(data: HealthFeatures):
        try:
            # Validate required features
            required_features = {
                "Age", "Gender", "Ethnicity", "EducationLevel", "BMI",
                "Smoking", "AlcoholConsumption", "PhysicalActivity", "DietQuality",
                "SleepQuality", "FamilyHistoryAlzheimers", "CardiovascularDisease",
                "Diabetes", "Depression", "HeadInjury", "Hypertension",
                "SystolicBP", "DiastolicBP", "CholesterolTotal", "CholesterolLDL",
                "CholesterolHDL", "CholesterolTriglycerides", "MMSE",
                "FunctionalAssessment", "MemoryComplaints", "BehavioralProblems",
                "ADL", "Confusion", "Disorientation", "PersonalityChanges",
                "DifficultyCompletingTasks","Forgetfulness"
            }

            if not all(feature in data.features for feature in required_features):
                missing_features = required_features - set(data.features.keys())
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required features: {', '.join(missing_features)}"
                )
            features_array = np.array([[data.features[feature] for feature in required_features]])
            pca=joblib.load('pca_model.pkl')
            data=pca.transform(features_array)
            model=joblib.load('rfmodel.joblib')
            print(data)

            # Here you would implement your actual prediction logic
            # This is a placeholder for demonstration
            # Replace with your actual machine learning model prediction
            
            prediction =model.predict(data) # Example probability
            print(prediction[0])
            # Determine risk level based on prediction
            if(prediction[0]==1):
                  predict="YES"
            elif (prediction==0):
                  predict="NO"
            else : predict ="eror"
            return PredictionResponse(
                prediction=predict
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
    