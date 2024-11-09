import sys
import os
from src.exception import CustomException
from src.logger import logging

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor

from sklearn.metrics import r2_score

from dataclasses import dataclass
from src.utils import save_object,evaluate_model
import numpy as np

@dataclass
class ModelTrainerConfig:
    model_file_path:str=os.path.join("artifacts/model.pkl")

class ModelTrainer:
    def __init__(self):
        self.trainer_config=ModelTrainerConfig()

    def initiate_model_trianer(self,train_array,test_array):
        try:
          logging.info("split the train and test data as x_train,y_train,x_test,y_test")
       

          x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
        
          logging.info("by using the models we can train the data")
          models={
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                 "ridge":Ridge(),
                 "Lasso":Lasso(),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
          
          params={
             "Random Forest":{
                 'n_estimators':[100,200,300,400,500],
                  "criterion":['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
              'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
             },
             "Decision Tree":{
                    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 4, 8],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_leaf_nodes': [None, 20, 50, 100]

             },
             "Gradient Boosting":{
                 'n_estimators': [100, 200, 300, 400],                # Number of boosting stages
    'learning_rate': [0.01, 0.05, 0.1, 0.2],              # Step size at each iteration
    'max_depth': [3, 5, 7, 10],                            # Maximum depth of the individual trees
    'min_samples_split': [2, 5, 10],                        # Minimum samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],                          # Minimum samples required to be at a leaf node
    'subsample': [0.8, 0.9, 1.0],                          # Fraction of samples used for fitting each tree
    'criterion': ['friedman_mse', 'squared_error'],         # Loss function to optimize
    'max_features': ['auto', 'sqrt', 'log2', None]
             },
             "Linear":{
                
             },
             "Ridge":{
                'alpha': [0.1, 1, 10, 100, 1000],          # Regularization strength
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg'],  # Solver for optimization
    'max_iter': [1000, 2000, 3000],            # Maximum number of iterations
    'tol': [1e-4, 1e-3, 1e-2]
             },
             "Lasso":{
                'alpha': [0.1, 1, 10, 100, 1000],          # Regularization strength
    'selection': ['cyclic', 'random'],         # Feature selection strategy
    'max_iter': [1000, 2000, 3000],            # Maximum number of iterations
    'tol': [1e-4, 1e-3, 1e-2]  
             },
             "AdaBoost Regressor":{
                'n_estimators': [50, 100, 200],                 # Number of boosting stages
    'learning_rate': [0.01, 0.1, 1.0, 1.5],         # Weighting of each classifier
    'loss': ['linear', 'square', 'exponential'],     # Loss function to minimize
    'base_estimator': [DecisionTreeRegressor(max_depth=3), DecisionTreeRegressor(max_depth=5)] 
             }
            
          }
         



        
          # print(type(models))
          # print(type(params))
          model_report:dict=evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,params=params)
          logging.info("we get the best score")

          best_score=max(sorted(model_report.values()))

          best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_score)
            ]

          best_model=models[best_model_name]

          if best_score<0.4:
            raise CustomException("Not found a model",sys)

          logging.info(f"best model found on train and test data")
          
          logging.info("save the file and obj model")
          save_object(
            file_path=self.trainer_config.model_file_path,
            obj=best_model
        )
          logging.info("preict the r2_score")
          predicted=best_model.predict(x_test)
          r2_square = r2_score(y_test, predicted)
          return r2_square
        except Exception as e:
           raise CustomException(e,sys)


