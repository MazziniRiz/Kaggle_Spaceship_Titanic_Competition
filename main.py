import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import OrdinalEncoder

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')



