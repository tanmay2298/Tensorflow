import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

diabetes = pd.read_csv('pima-indians-diabetes.csv')
print diabetes.head()

print diabetes.columns