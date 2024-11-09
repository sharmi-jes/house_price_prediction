import sys
from src.logger import logging

def error_message_detail(error, error_detail):
    exc_type, exc_obj, exc_tb = error_detail.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        error_message = "Error occurred in python script name [{0}] line no[{1}] error message [{2}]".format(
            file_name, exc_tb.tb_lineno, str(error)
        )
    else:
        error_message = f"Error occurred: {str(error)}"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        # Generate detailed error message
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message

if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as e:
        logging.info(f"Error occurred: {e}")  # Log the exception message
        raise CustomException(e, sys)  # Pass sys to capture traceback details
