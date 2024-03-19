import os
import numpy as np

def print_log(args, string):
    """
    Printing log information
    打印日志信息
    """
    log = open(args.log_file, 'a')
    log.write(string + '\n')
    log.flush()
    print(string)
