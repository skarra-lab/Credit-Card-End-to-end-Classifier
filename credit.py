import pandas as pd
import numpy as np
import sqlite3 

from sklearn import preprocesssing 
import matplotlib.pyplot as plt

#Extract data
df = pd.read_csv("creditcard.csv")
print(df.head())