import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from flask import jsonify

app = Flask(__name__, static_folder='client/age_gender_prediction/build', static_url_path='/')

genderModel = pickle.load(open('./genderModel.pkl', 'rb'))
ageModel = pickle.load(open('./ageModel.pkl', 'rb'))

category_columns = pickle.load(open('./category_columns.pkl', 'rb'))
df_classes = pickle.load(open('./df_classes.pkl', 'rb'))
test_data = pickle.load(open('./test_data.pkl', 'rb'))


### This method is to find the respective class value

def findCategoryAndItsClass(column, value):
    colIndex = category_columns.index(column)
    classValue = np.where(df_classes[colIndex] == value)
    if len(classValue[0]) > 0:
        return classValue[0][0]
    else:
        return -1

### Entry route

@app.route('/')
def home():
    return app.send_static_file('index.html')

### Endpoint to load predict data with probability mapped for both age/gender

@app.route('/predict', methods=['GET'])
def predict():
    ### Get 50 sample device ids
    sample50 = test_data.sample(50).reset_index(drop=True)

    ### Predict gender - pass sample without device id and gender
    genderPrediction = genderModel.predict(sample50.drop(['gender', 'device_id'], axis=1))
    genderProb =  genderModel.predict_proba(sample50.drop(['gender', 'device_id'],axis=1))

    # Creating prediciton and probability as dataframe 
    genderProb_df = pd.DataFrame(data=genderProb, columns=['female_probability', 'male_probability'])
    genderPred_df = pd.DataFrame(data=genderPrediction, columns=['predicted_gender'])
    data = sample50.merge(genderProb_df, left_index=True, right_index=True)
    data = data.merge(genderPred_df, left_index=True, right_index=True)

    # Age predict - pass sample without device id and age_group
    ageProb = ageModel.predict_proba(sample50.drop(['age_group', 'device_id'], axis=1))
    agePred = ageModel.predict(sample50.drop(['age_group', 'device_id'], axis=1))   

    # Creating age prediciton and probability as dataframe
    ageProb_df = pd.DataFrame(data=ageProb, columns=['0-24_Prob', '24-32_Prob', '32_plus_Prob'])
    agePred_df = pd.DataFrame(data=agePred, columns=['predicted_age'])
    data = data.merge(ageProb_df, left_index=True, right_index=True)
    data = data.merge(agePred_df, left_index=True, right_index=True)

    # Converting encoded class value to actual value - the reason we have df_classes
    data['gender'] = data['gender'].apply(lambda x: df_classes[0][x])
    data['predicted_gender'] = data['predicted_gender'].apply(lambda x: df_classes[0][x])
    data['phone_brand'] = data['phone_brand'].apply(lambda x: df_classes[2][x])
    data['device_model'] = data['device_model'].apply(lambda x: df_classes[3][x])
    data['age_group'] = data['age_group'].apply(lambda x: df_classes[4][x])
    data['predicted_age'] = data['predicted_age'].apply(lambda x: df_classes[4][x])
    data['GENERIC_CATEGORY'] = data['GENERIC_CATEGORY'].apply(lambda x: df_classes[5][x])

    # return it as json - frontend will use this endpoint to render the table
    return data.to_json(orient='records')

### endpoint to predict gender

@app.route('/genderPredict', methods=['POST'])
def genderPredict():
    # retrieve post data
    req_data = request.get_json()
    event_id = req_data['event_id']
    hour_of_day = req_data['hour_of_day']
    day_of_week = req_data['day_of_week']
    longitude = req_data['longitude']
    latitude = req_data['latitude']
    # convert actual to class value
    phone_brand = findCategoryAndItsClass('phone_brand', req_data['phone_brand'])
    device_model = findCategoryAndItsClass('device_model', req_data['device_model'])

    app_id = req_data['app_id']
    age_group = findCategoryAndItsClass('age_group', req_data['age_group'])
    GENERIC_CATEGORY = findCategoryAndItsClass('GENERIC_CATEGORY', req_data['GENERIC_CATEGORY'])

    prediction = genderModel.predict(
        [[event_id, longitude, latitude, phone_brand, device_model, app_id, age_group, hour_of_day, day_of_week, GENERIC_CATEGORY]])

    genderIndex = category_columns.index('gender')   
    predictedGender = df_classes[genderIndex][prediction[0]]

    # return the predicted info
    return '''Predicted gender is: {}'''.format(predictedGender)

#### endpoint to predict age    

@app.route('/agePredict', methods=['POST'])
def agePredict():
    req_data = request.get_json()
    print(req_data)
    event_id = req_data['event_id']
    hour_of_day = req_data['hour_of_day']
    day_of_week = req_data['day_of_week']
    longitude = req_data['longitude']
    latitude = req_data['latitude']
    # convert actual to class value
    phone_brand = findCategoryAndItsClass('phone_brand', req_data['phone_brand'])
    device_model = findCategoryAndItsClass('device_model', req_data['device_model'])

    app_id = req_data['app_id']
    gender = findCategoryAndItsClass('gender', req_data['gender'])
    GENERIC_CATEGORY = findCategoryAndItsClass('GENERIC_CATEGORY', req_data['GENERIC_CATEGORY'])

    prediction = ageModel.predict(
        [[gender, event_id, longitude, latitude, phone_brand, device_model, app_id, hour_of_day, day_of_week, GENERIC_CATEGORY]])

    ageGroupIndex = 4  
    predictedAgeGroup = df_classes[ageGroupIndex][prediction[0]]
   
    # return the predicted info
    return '''Predicted age group is: {}'''.format(predictedAgeGroup)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")