import sys
import os
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts/preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
        logging.info("DataTransformationConfig class created to take inputs.")

    def get_data_transformation(self):
        logging.info("This function is responsible for transforming the data.")
        
        numerical_cols = ["BHK", "Bathroom"]
        categorical_cols = ['Area Type', 'City', 'Furnishing Status', 'Tenant Preferred']

        # Creating numerical pipeline
        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]
        )

        # Creating categorical pipeline
        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one", OneHotEncoder())
            ]
        )

        # Combining both pipelines using ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ("num_pipeline", num_pipeline, numerical_cols),
                ("cat_pipeline", cat_pipeline, categorical_cols)
            ]
        )

        return preprocessor

    def intiate_data_transformation(self, train_path, test_path):
        try:
            logging.info(f"Reading train data from: {train_path}")
            train_df = pd.read_csv(train_path)
            logging.info(f"Reading test data from: {test_path}")
            test_df = pd.read_csv(test_path)

            target_col = ["Rent"]
            preprocessor_obj = self.get_data_transformation()

            # Splitting data into input and target variables
            input_train = train_df.drop(columns=target_col)
            input_target = train_df[target_col]

            input_test = test_df.drop(columns=target_col)
            input_test_target = test_df[target_col]

            logging.info("Applying preprocessor to train and test data")
            # Fit and transform the training data, and transform the test data
            input_preprocessor_train = preprocessor_obj.fit_transform(input_train)
            input_preprocessor_test = preprocessor_obj.transform(input_test)

            logging.info("Combining input and target data for train and test")
            train_array = np.c_[
                input_preprocessor_train, np.array(input_target),
            ]

            test_array = np.c_[
                input_preprocessor_test, np.array(input_test_target)
            ]

            # Saving the preprocessor object
            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            logging.info("Returning the train and test arrays and the preprocessor file path")
            return (
                train_array,
                test_array,
                self.transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e,sys)
