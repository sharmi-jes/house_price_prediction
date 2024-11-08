import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from src.logger import logging



def error_message_detail(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_lines()
    error_message="Error occured in pytohn script should file name [{0}] file no [{1}] error is [{2}]".format(
    file_name,exc_tb.tb_lineno,str(error)
    )

    return error_message



class CustomException(Exception):
    def __init__(self,error_message,error_detail):
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detail=error_detail)


    def str(self):
        return self.error_message
    


if __name__=="__main__":
    try:
        a=1/0;
    except Exception as e:
        logging.info("divisible by zero")
        raise CustomException(e,sys)

