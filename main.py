import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

X = train_data.copy()
y = test_data.copy()

#Breaking Down Complex Features:
#both "PassengerId" and "Cabin varaibles" have data that can be broken down

#Feature correction (fc):

X[['Group','Ranking']] = (X['PassengerId'].str.split("_", expand=True))
y[['Group','Ranking']] = (y['PassengerId'].str.split("_", expand=True))

X[['Deck', 'Num', 'Side']] = (X['Cabin'].str.split("/", expand=True))
y[['Deck', 'Num', 'Side']] = (y['Cabin'].str.split("/", expand=True))

X = X.drop(['PassengerId', 'Cabin', 'Name'], axis=1)
y = y.drop(['PassengerId', 'Cabin', 'Name'], axis=1)

X['Group'] = X['Group'].astype(int)
y['Group'] = y['Group'].astype(int)

#Applying Ordinal Encoding
X_label = X.copy()
y_label = y.copy()
object_cols = [cols for cols in X.columns if X[cols].dtypes == 'object']

ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_label[object_cols] = ordinal_encoder.fit_transform(X[object_cols])
y_label[object_cols] = ordinal_encoder.transform(y[object_cols])

#Applying Simple Imputer
my_imputer = SimpleImputer(strategy='mean')
imputed_X = pd.DataFrame(my_imputer.fit_transform(X_label))
imputed_y = pd.DataFrame(my_imputer.transform(y_label))

imputed_X.columns = X_label.columns
imputed_y.columns = y_label.columns
