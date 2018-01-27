import numpy as np
from sklearn.preprocessing import MinMaxScaler

data = np.random.randint(0, 100, (10, 2))
print data

scaler_model = MinMaxScaler()
print type(scaler_model)

scaler_model.fit(data)
print scaler_model

print scaler_model.transform(data)

# OR
print "Other way"
print scaler_model.fit_transform(data)

from sklearn.model_selection import train_test_split