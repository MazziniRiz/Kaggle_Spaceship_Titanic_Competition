import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib as plt
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

X = train_data.drop(columns=['Transported'])
y = train_data['Transported']
X_test = test_data.copy()

#Breaking Down Complex Features:
#both "PassengerId" and "Cabin varaibles" have data that can be broken down

#Feature correction (fc):

#PassengerId can be split into Group and Ranking
X[['Group','Ranking']] = (X['PassengerId'].str.split("_", expand=True))
X_test[['Group','Ranking']] = (X_test['PassengerId'].str.split("_", expand=True))

#Cabin can be split into Deck, Num, and Side
X[['Deck', 'Num', 'Side']] = (X['Cabin'].str.split("/", expand=True))
X_test[['Deck', 'Num', 'Side']] = (X_test['Cabin'].str.split("/", expand=True))

#drop the originals for their more readable counterparts
X = X.drop(['PassengerId', 'Cabin', 'Name'], axis=1)
X_test = X_test.drop(['PassengerId', 'Cabin', 'Name'], axis=1)

#was type == object, converted to type == int for better use
X['Group'] = X['Group'].astype(int)
X_test['Group'] = X_test['Group'].astype(int)

X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2,random_state=0)

#Applying Ordinal Encoding
label_X_train = X_train.copy()
label_X_val = X_val.copy()
label_X_test = X_test.copy()
object_cols = [cols for cols in X_train.columns if X_train[cols].dtypes == 'object']

ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
label_X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
label_X_val[object_cols] = ordinal_encoder.transform(X_val[object_cols])
label_X_test[object_cols] = ordinal_encoder.transform(X_test[object_cols])

# Test1: Applying Simple Imputer
my_imputer = SimpleImputer(strategy='mean')
imputed_X_train= pd.DataFrame(my_imputer.fit_transform(label_X_train))
imputed_X_val = pd.DataFrame(my_imputer.transform(label_X_val))
imputed_X_test = pd.DataFrame(my_imputer.transform(label_X_test))

imputed_X_train.columns = label_X_train.columns
imputed_X_val.columns = label_X_val.columns
imputed_X_test.columns = label_X_test.columns

#Testing to see which feature is relevant via. MI scores

discrete_features = X_train.dtypes == int

def make_mi_scores(X,y,discrete_features):
    mi_scores = mutual_info_regression(imputed_X_train,y_train,discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name='MI Scores', index = imputed_X_train.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

mi_scores = make_mi_scores(imputed_X_train, y_train, discrete_features)

#Testing using PCA to find relevant features (INCREASED MAE AND DECREASED ACCURACY)

'''''
#Standardize
X_scaled = (imputed_X_train - imputed_X_train.mean(axis=0))/imputed_X_train.std(axis=0)

#Creating Principal Components
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
X_pca = pd.DataFrame(X_pca, columns=component_names)

#Wrap loadings up in a dataframe

loadings = pd.DataFrame(pca.components_.T, columns=component_names, index=X.columns)

imputed_X_train = imputed_X_train.join(X_pca)

mi_scores_pca = make_mi_scores(imputed_X_train, y_train, discrete_features=False)
'''

#Based on Observation ['Group','CryoSleep'] have the most effect as features
# ['Spa','RoomService','VRDeck','ShoppingMall','FoodCourt'] have some influence

#Unsure about the label encoding on ['Group'], maybe fix it using a k-means cluster or a PCA?

# Test 1: Using only ['Group','CryoSleep'] 

'''''
X_features = imputed_X_train[['Group', 'CryoSleep']]
X_val_features = imputed_X_val[['Group', 'CryoSleep']]
my_model = XGBRegressor(n=500)
my_model.fit(X_features,y_train, verbose = False)

y_pred = my_model.predict(X_val_features)

mean_absolute_error(y_pred, y_val)
'''

#MAE = 0.388, Accuracy = 1-0.388 = 62% (Not Bad, can be better) (SimpleImputer)


# Test2: Using ['Group','CryoSleep'] and ['Spa','RoomService','VRDeck','ShoppingMall','FoodCourt'] 


X_features = imputed_X_train[['Group', 'CryoSleep', 'Spa','RoomService','VRDeck','ShoppingMall','FoodCourt']]
X_val_features = imputed_X_val[['Group', 'CryoSleep','Spa','RoomService','VRDeck','ShoppingMall','FoodCourt']]
my_model = XGBClassifier(n=300)
my_model.fit(X_features,y_train, verbose = False)

y_pred = my_model.predict(X_val_features)

mean_absolute_error(y_pred, y_val)

# MAE = 0.312, Accuracy = 68.7% (SimpleImputer)

#Test3: Use all features 
'''''
my_model = XGBClassifier(n=300)
my_model.fit(imputed_X_train,y_train, verbose = False)

y_pred = my_model.predict(imputed_X_val)

mean_absolute_error(y_pred, y_val)
'''''
#MAE = 0.282, Accuracy = 71.2% (SimpleImputer)

#Test4: Getting rid of features with 0 MI score 
'''''
X_features = imputed_X_train[['Group', 'CryoSleep', 'Spa','RoomService','VRDeck','ShoppingMall','FoodCourt', 'HomePlanet', 'Deck', 'Age', 'Side', 'Num', 'Destination']]
X_val_features = imputed_X_val[['Group', 'CryoSleep','Spa','RoomService','VRDeck','ShoppingMall','FoodCourt', 'HomePlanet', 'Deck', 'Age', 'Side', 'Num', 'Destination']]
my_model = XGBClassifier(n=300)
my_model.fit(X_features,y_train, verbose = False)

y_pred = my_model.predict(X_val_features)

'''
#MAE = 0.279, Accuracy = 72.1% (SimpleImputer)

#Next Approach maybe use k-means and PCA to test features, Optimize the preprocessing from SimpleImputer to something else.

X_features_test = imputed_X_test[['Group', 'CryoSleep', 'Spa','RoomService','VRDeck','ShoppingMall','FoodCourt']]
y_pred_test = my_model.predict(X_features_test)

y_pred_bool = [bool(x) for x in y_pred_test]

output = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Transported': y_pred_bool
})

output.to_csv('Submission4.csv', index=False)

#Changed Regressor to Classifier, Realized my dumb mistake, final score = 79% grading on the model

