import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import matplotlib as plt

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

X_train = train_data.drop(columns=['Transported'])
y_train = train_data['Transported']
X_test = test_data.copy()

#Breaking Down Complex Features:
#both "PassengerId" and "Cabin varaibles" have data that can be broken down

#Feature correction (fc):

#PassengerId can be split into Group and Ranking
X_train[['Group','Ranking']] = (X_train['PassengerId'].str.split("_", expand=True))
X_test[['Group','Ranking']] = (X_test['PassengerId'].str.split("_", expand=True))

#Cabin can be split into Deck, Num, and Side
X_train[['Deck', 'Num', 'Side']] = (X_train['Cabin'].str.split("/", expand=True))
X_test[['Deck', 'Num', 'Side']] = (X_test['Cabin'].str.split("/", expand=True))

#drop the originals for their more readable counterparts
X_train = X_train.drop(['PassengerId', 'Cabin', 'Name'], axis=1)
X_test = X_test.drop(['PassengerId', 'Cabin', 'Name'], axis=1)

#was type == object, converted to type == int for better use
X_train['Group'] = X_train['Group'].astype(int)
X_test['Group'] = X_test['Group'].astype(int)

#Applying Ordinal Encoding
label_X_train = X_train.copy()
label_X_test = X_test.copy()
object_cols = [cols for cols in X_train.columns if X_train[cols].dtypes == 'object']

ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
label_X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
label_X_test[object_cols] = ordinal_encoder.transform(X_test[object_cols])

#Applying Simple Imputer
my_imputer = SimpleImputer(strategy='mean')
imputed_X_train= pd.DataFrame(my_imputer.fit_transform(label_X_train))
imputed_X_test = pd.DataFrame(my_imputer.transform(label_X_test))

imputed_X_train.columns = label_X_train.columns
imputed_X_test.columns = label_X_test.columns

#Testing to see which feature is relevant via. MI scores

discrete_features = X_train.dtypes == int

def make_mi_scores(X,y,discrete_features):
    mi_scores = mutual_info_regression(imputed_X_train,y_train,discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name='MI Scores', index = imputed_X_train.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

mi_scores = make_mi_scores(imputed_X_train, y_train, discrete_features)

#Based on Observation ['Group','CryoSleep'] have the most effect as features
# ['Spa','RoomService','VRDeck','ShoppingMall','FoodCourt'] have some influence

#Unsure about the label encoding on ['Group'], maybe fix it using a k-means cluster or a PCA?

# Test 1: Using only ['Group','CryoSleep']

# Test2: Using ['Group','CryoSleep'] and ['Spa','RoomService','VRDeck','ShoppingMall','FoodCourt']