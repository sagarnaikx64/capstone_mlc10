# README #

This application is to predict the age/gender on telecom dataset.

# Backend #
* Python does the predicting work
* We have `predictModel.py` which will do the modelling and save respective model into pickle file
* `genderModel.pkl` will be used to predict gender
* `ageModel.pkl` will be used to predict age
* `df_classes.pkl` - this pickle holds the information of label encoded respective classes. Helpful to retransform and find the actual value.
* `test_data.pkl` - this pickle holds the sample 50 test data.
* `app.py` - This is main flask app that initiate and create endpoints

# Frontend #
* This has been build using reactjs and material ui framework


### How do I get set up? ###

* This whole app can be dockerised
* Before doing that we need build our frontend (we can enhance it to separate frontend docker)
* `cd /client/age_gender_prediction` - and run `npm install` to download frontend dependencies
*  run `npm run build` - frontend will be compiled and packed into build folder
* Once this is done we can run docker from the root
* `Dockerfile` has to configuration to create docker images
* run `sudo docker build -t age_gender_prediction .` to create docker image
* run `docker images -a` to verify docker image
* run `sudo docker run -p 5000:5000 age_gender_prediction` to run the docker
* if you are running on local - verify it on browser via `localhost:5000`
# capstone_mlc10
