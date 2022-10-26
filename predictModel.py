## Loading libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import datetime as dt
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from mlxtend.classifier import StackingCVClassifier
from sklearn.ensemble import StackingClassifier
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

final_data_events = pd.read_csv('https://fmartin1.s3.amazonaws.com/final_data_events.csv')
final_data_events = final_data_events[(final_data_events.longitude>1) & (final_data_events.latitude>1)]

## For better performance and run time constraints reduced the dataset

train_sample = final_data_events[final_data_events['train_test_flag'] == 'train'].sample(100000)
test_sample = final_data_events[final_data_events['train_test_flag'] == 'test'].sample(20000)

## Since gender: Male is biased - Adding few more female for better training
female_df = final_data_events[final_data_events.gender == 'F'].sample(10000)
## Concat all dataframes
final=pd.concat([train_sample, test_sample])
category_columns = final.select_dtypes(include='object').columns.to_list()
category_columns.remove('train_test_flag')

df_classes = []
def labelEncode_saveClass(x):
  le = LabelEncoder()
  colTransformed = le.fit_transform(x)
  df_classes.append(le.classes_)
  return colTransformed

def findCategoryAndItsClass(column, value):
  colIndex = category_columns.index(column)
  classValue = np.where(df_classes[colIndex] == value)
  if len(classValue[0]) > 0:
    return classValue[0][0]
  else:
    return -1

final[category_columns] = final[category_columns].apply(lambda col: labelEncode_saveClass(col))  

test_data = final.drop_duplicates(['device_id'])
test_data.drop(['train_test_flag', 'is_installed', 'is_active'], axis=1, inplace=True)
pickle.dump(test_data, open('test_data.pkl', 'wb'))

pickle.dump(df_classes, open('df_classes.pkl', 'wb'))
pickle.dump(category_columns, open('category_columns.pkl', 'wb'))

scenario1_train = final[final['train_test_flag'] == 'train']
scenario1_test = final[final['train_test_flag'] == 'test']

scenario1_train.drop(['train_test_flag', 'device_id', 'is_installed', 'is_active'], axis=1, inplace=True)
scenario1_test.drop(['train_test_flag', 'device_id', 'is_installed', 'is_active'], axis=1, inplace=True)


### Gender Prediction

X_train = scenario1_train.drop('gender', axis=1)
y_train = scenario1_train.gender
X_test = scenario1_test.drop('gender', axis=1)
y_test = scenario1_test.gender

scaler = StandardScaler()
scaled_df = scaler.fit_transform(X_train)
x_train = pd.DataFrame(scaled_df)
scaled_df = scaler.fit_transform(X_test)
x_test = pd.DataFrame(scaled_df)

clf1 = LogisticRegression()
clf2 = RandomForestClassifier(random_state=1, n_estimators=100, min_samples_leaf = 50, min_samples_split=12, max_depth=3)
xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bytree=0.6,
              gamma=0.5, learning_rate=0.05, max_depth=3,
              min_child_weight=10, missing=None, n_estimators=100, n_jobs=1,
              subsample=0.8, verbosity=1)

# Below code is not really needed - still kept it for me know the scores

genderModel = StackingCVClassifier(classifiers=[clf1, clf2], meta_classifier=xgb, use_probas=True, cv=5)
genderModel.fit(x_train.values, y_train.values)

for clf, label in zip([clf1, clf2, genderModel], 
                      ['lr', 
                       'Random Forest', 
                       'Stacking CV Classifier']):

    scores = model_selection.cross_val_score(clf, x_train.values, y_train.values, cv=5, scoring='accuracy')
    print("Gender Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

pickle.dump(genderModel, open('genderModel.pkl', 'wb'))

### Age Prediction

X_train2 = scenario1_train.drop('age_group', axis=1)
y_train2 = scenario1_train.age_group
X_test2 = scenario1_test.drop('age_group', axis=1)
y_test2 = scenario1_test.age_group

scaler = StandardScaler()
scaled_df = scaler.fit_transform(X_train2)
x_train2 = pd.DataFrame(scaled_df)
scaled_df = scaler.fit_transform(X_test2)
x_test = pd.DataFrame(scaled_df)


ageModel = StackingCVClassifier(classifiers=[clf1, clf2], meta_classifier=xgb, use_probas=True, cv=5)
ageModel.fit(x_train2.values, y_train2.values)

pickle.dump(ageModel, open('ageModel.pkl', 'wb'))

# Below code is not really needed - still kept it for me know the scores

for clf, label in zip([clf1, clf2, ageModel], 
                      ['lr', 
                       'Random Forest', 
                       'Stacking CV Classifier']):

    scores = model_selection.cross_val_score(clf, x_train2.values, y_train2.values, cv=5, scoring='accuracy')
    print("Age Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
