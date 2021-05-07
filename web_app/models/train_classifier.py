# import libraries
import sys
import pandas as pd
import re
import nltk
import pickle

nltk.download(['punkt','wordnet','stopwords'])

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report



def load_data(database_filepath):
    """
    INPUT
        database filepath for sqlite database containing cleaned message data
    OUTPUT
        1 - an array X - containing the message column ready for natural language processing and an input to the ml models
        2 - a dataframe y - containing the 36 response columns for each message
        3 - a list category_names to label the model evaluation later
        
    SUMMARY
        This function reads in the entire message_cats table and creates the X input array, the y dataframe and the category_names list to be used later in the ml models        
    
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('SELECT * FROM message_cats', engine)

    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = list(y.columns)
    
    return X, y, category_names


def tokenize(text):
    """
    INPUT
        a string of text
    OUTPUT
        a processed and cleaned list of tokens
        
    SUMMARY
        This function prepares a 'message' text for ml models by the following steps
            - convert all to lower case
            - create a list of tokens (word tokenize)
            - remove stopwords from tokens
            - lemmatize the words to common dictionary stems using WordNetLemmatizer
            - outputs a clean list of tokens called clean_tokens
    """
    text = re.sub(r'[^a-zA-Z0-9]', " ", text.lower())
    
    tokens = word_tokenize(text)
    
    words = [w for w in tokens if w not in stopwords.words("english")]
    
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in words:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)
                      
    return clean_tokens


def build_model():
    """
    Build a gridsearch pipeline using two transformers (CountVectorizer and TfidfTransformer) and a knn classifier.
    GridSearchCV used to ascertain best features for final model
    OUTPUT
        A CV object to be saved as a pickle file as the final model
    """
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {"clf__estimator__min_samples_split": [2, 4],
                #'clf__estimator__n_estimators': [10, 20],
              #"clf__min_samples_leaf": [1, 2],
              #"clf__criterion": ["gini", "entropy"]
             }
        
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    A function to output a score for each column prediction from the model.
    Printing the score for the precision, the recall and the overall F1 score for each.
    """
    
    # predict y multioutputs using model
    y_pred = model.predict(X_test)
    
    # iterate through y_pred prediction columns and score against y_test using classification_report to give F1 score, precision and recall scores
    n=0
    for col in Y_test.columns:
        print("Column_tested:{}".format(Y_test.columns[n]))
        print("Result:")
        print(classification_report(Y_test.iloc[:,n], y_pred[:,n]))
        n+=1
    
    
    
    


def save_model(model, model_filepath):
    """
    Save down pickle file of the model
    """
    
    with open(model_filepath, 'wb') as pklfile:
        pickle.dump(model, pklfile)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()