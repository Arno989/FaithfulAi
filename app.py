import logging
import sys
import os
from threading import Thread

# from tkinter import *


PROJECT_ROOT = os.path.dirname(os.path.abspath("./"))

print(PROJECT_ROOT)

try:
    exec(open("./ML/Test-2.py").read())
    # while True:
    #     try:
    #         pass
    #     except Exception as ex:
    #         pass

except KeyboardInterrupt:
    pass
