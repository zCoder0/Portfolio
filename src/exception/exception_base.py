import sys
from src.logger import logging

class ProjectException(Exception):
    
    def __init__(self,error_message,erroe_details:sys):
        self.error_message = error_message
        
        _,_,exc_tb = erroe_details.exc_info()
        
        self.lineno = exc_tb.tb_lineno
        self.filename = exc_tb.tb_frame.f_code.co_filename
        
        logging.logging.info(self.__str__())
    def __str__(self):
        return f"Error occured in python script name at {self.filename} on line number {self.lineno} error messgae {self.error_message}"
    