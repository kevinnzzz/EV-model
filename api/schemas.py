from typing import List

from pydantic import BaseModel, Field


class InputData(BaseModel):
    Age: float = Field(ge=17, le=100)
    Annual_Income_USD: float = Field(ge=0)
    Daily_Commute_km: float = Field(ge=0)
    Number_of_Cars_Owned: int = Field(ge=0)
    Charging_Stations_Near_Home: int = Field(ge=0)
    Charging_Stations_Near_Work: int = Field(ge=0)
    Environmental_Concern_Level: int = Field(ge=1, le=5)
    Gender: str
    City_Type: str
    Current_Car_Type: str
    Home_Charging_Possible: int = Field(ge=0, le=1)
    Subsidy_Available: int = Field(ge=0, le=1)
    Range_Anxiety_Level: int

    model_config = {
        "json_schema_extra": {
            "example": {
                "Age": 34, "Annual_Income_USD": 52000, "Daily_Commute_km": 25,
                "Number_of_Cars_Owned": 1, "Charging_Stations_Near_Home": 2,
                "Charging_Stations_Near_Work": 1, "Environmental_Concern_Level": 4,
                "Gender": "Male", "City_Type": "Urban", "Current_Car_Type": "Petrol",
                "Home_Charging_Possible": 1, "Subsidy_Available": 1,
                "Range_Anxiety_Level": 3,
            }
        }
    }


class BatchIn(BaseModel):
    items: List[InputData] = Field(min_length=1, max_length=1000)


class PredictionOut(BaseModel):
    probability: float
    prediction: int      # 1 jika probability >= threshold
    threshold: float