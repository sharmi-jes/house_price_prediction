import pandas as pd
import numpy as np
import sys
import os
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_training import ModelTrainer
from src.components.model_training import ModelTrainerConfig

@dataclass
class DataIngestionconfig:
    raw_data_path:str=os.path.join("artifacts/raw.csv")
    train_data_path:str=os.path.join("artifacts/train.csv")
    test_data_path:str=os.path.join("artifacts/test.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionconfig()

    def initiate_data_ingestion(self):
        try:

         df=pd.read_csv(r"D:\RESUME ML PROJECTS\HOUSE_PRICES\notebooks\house_price\House_Rent_Dataset.csv")

         logging.info("create a directory for train trst raw data")

         os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
         df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)


         logging.info("split the dtaa as train and test")

         train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

         logging.info("passed the data train inot the path")
         train_set.to_csv(self.ingestion_config.train_data_path)
         test_set.to_csv(self.ingestion_config.test_data_path)

         return (
            self.ingestion_config.train_data_path,
            self.ingestion_config.test_data_path
         )
        except Exception as e:
           raise CustomException(e,sys)
        

if __name__=="__main__":
   obj=DataIngestion()
   train_data,test_data=obj.initiate_data_ingestion()

   transformation=DataTransformation()
   train_arr,test_arr,_=transformation.intiate_data_transformation(train_data,test_data)

   trainer=ModelTrainer()
   print(trainer.initiate_model_trianer(train_arr,test_arr))

         

