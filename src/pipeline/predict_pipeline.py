import sys
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.preprocessing import StandardScaler
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:

         model_path="artifacts/model.pkl"
         preprocessor_path="artifacts/preprocessor.pkl"
         model=load_object(file_path=model_path)
         preprocessor=load_object(file_path=preprocessor_path)
         data_scaled=preprocessor.transform(features)
         prediction= model.predict(data_scaled)
         return prediction
        except Exception as e:
            raise CustomException(e,sys)




# Area_Type', 'Area_Locality', 'City',
    #    'Furnishing_Status', 'Tenant_Preferred'

class CustomData:
   def __init__(self, BHK,  Size,  Area_Type,  City,Furnishing_Status, Tenant_Preferred, Bathroom):
        self.BHK = BHK
        
        self.Size = Size
        
        self.Area_Type = Area_Type
      
        self.City = City
        self.Furnishing_Status = Furnishing_Status
        self.Tenant_Preferred = Tenant_Preferred
        self.Bathroom = Bathroom
   def get_data_as_data_frame(self):
        try:
            input_data={
                "BHk":[self.BHK],
                "Size":[self.Size],
                "AreaType":[self.Area_Type],
                "City":[self.City],
                "FurnishingStatus":[self.Furnishing_Status],
                "TenantPreferred":[self.Tenant_Preferred],
                "Bathroom":[self.Bathroom]
                 }

            return pd.DataFrame(input_data)


           
        except Exception as e:
            raise CustomException(e,sys)




