# Disaster Response Messaging Pipeline
> A project on Machine Learning pipelines as part of the Udacity Data Scientist Nanodegree programme.

A web app to share the results of an ETL and Machine Learning pipeline.  Using real message data from
provided by FigureEight from various disaster response efforts over the past few years.  This machine learning
model uses the NLTK library for natural language processing and categorises each message to support emergency
response organisations in finding the most important communications amongst thousands of messages.



### How to run

How to run the python scripts and web app:

### Instructions:
1. Run the following commands in the project's web_app directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's web_app directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/




## Features

* Web app
* ETL pipeline
* Machine learning pipeline



## Files included

data
* process_data.py
	 This file contains the ETL pipeline.  It reads data from two csv files (disaster messages and disaster categories), cleans and merges this data and stores it in a sqlite database ready to be consumed by the machine learning pipeline to train the model.
* disaster_messages.csv
	 This file is a csv file of data provided by FigureEight.  It contains 26,248 rows and 4 columns.
     The columns are 'id', 'message', 'original' and 'genre' and all relate to messages received during real life disasters.
     The 'message' column is what we use from this file to train the machine learning model.
* disaster_categories.csv
	 This file is the second csv data file provided by FigureEight.  It also has 26,248 rows corresponding to the 'disaster_messages.csv' file - each linked with a message id.  
     This file has 2 columns which are 'id' and 'categories'.  
     The 'categories' column provides a labelled dataset for training showing how each message would have been manually categorised (after the event).  
     The categories column is actually made up of a string of 36 possible categorisation labels and each has been tagged as a 1 or 0 to denote if applicable 
     (e.g. request-1 means the message has been labelled as a request; request-0 means it has not been labelled as a request).
     Worthy of note is that a small number of the related columns have been tagged as a "2".  The 'process_data.py' file cleans this up later so that only 1 or 0 is used for training the model.
* DisasterResponse.db
	 This file is the sqlite database set up with the cleaned data at the end of the ETL pipeline, to be used at the start of the ML pipeline.
     It contains only one table 'message_cats' and is accessed using the sqlalchemy library.
     
models
* train_classifier.py
	 This file contains the machine learning pipeline.
     It reads in X and y data, along with column_names from the sqlite database, applies some nltk transformations and trains a machine learning model using GridSearchCV.
     Finally it stores the best performing model parameters in a pickle file called 'classifier.pkl'.
* classifier.pkl
	 This file is the pickled model ready to be used by the web application.
     
app
* run.py
	 The python file used to run the Flask application - instructions are included at the top of this README file.
     
app/templates
* go.html
	 The html template that is rendered with the results, when a user enters a text string into the "message to classify" bar.
     Presents a list of the 36 categories and highlights any that the model is predicting as related to the message entered.
* master.html
	 The base html template and the landing page.
     Displays the bar ready to accept the user's first message to test, along with a visualisation of the overall message data split by genre.


Input screen
![](https://github.com/MikeDurrant/DisasterResponse/blob/6aaf563dddb777818fabdcda38674cbd79f962a2/static/MessageEnterPage.PNG)


Output screen
![](https://github.com/MikeDurrant/DisasterResponse/blob/a386a079d1697683a455144076f41bf99470a7d9/static/ClassifierOutputPage.PNG)



## Note on imbalance

Many of the response columns are highly imbalanced with only a small number of positive responses.
In future iterations of this project this imbalance could be addressed for further improvement of the model scoring.


## Links


- Project web page: to be added (currently in process of setting up to deploy to Heroku, but not a requirement for project submission).



## Licensing

The code in this project is licensed under MIT license.