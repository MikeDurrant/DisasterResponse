import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    INPUTS 
        filepath to messages csv file
        filepath to categories csv file
        
    OUTPUTS
        new merged dataframe
        
    SUMMARY
        loads data from two csv files with a common id into separate dataframes, merges those into one and outputs as a single dataframe        
    
    """
    
    # read in messages csv
    messages = pd.read_csv(messages_filepath)
    
    # read in categories csv
    categories = pd.read_csv(categories_filepath)
    
    # merge the two dataframes
    df = messages.merge(categories, on='id')
    
    return df



def clean_data(df):
    """
    INPUT
        merged dataframe from the load_data function
        
    OUTPUT
        single dataframe cleaned and transformed and ready to save into a database for ML pipeline
        
    SUMMARY
        splits category column into 36 new columns and holds in new temporary dataframe
        renames new df columns with top row of data
        cleans these 36 column names to remove final two characters from string (e.g. "-0" or "-1")
        converts category values from text to a number - 1 or 0
        drops the category column from the input merged dataframe
        concatenates the temporary 36 column dataframe to the main df dataframe
        drops duplicates from the dataframe
        outputs the cleaned dataframe ready for ml pipeline        
        
    """
    
    # split the single categories column into new temporary dataframe with 36 columns
    categories = df.categories.str.split(";", expand=True)
    
    # select the first row of the categories df
    row = categories.head(1)
    
    # extract a list of the column names
    category_colnames = list(row.loc[0].values)
    
    # clean the list by removing last two characters of string
    for col in range(len(category_colnames)):
        category_colnames[col] = category_colnames[col][:-2]
    
    # rename the columns using this list
    categories.columns = category_colnames
    
    # set the values in the columns to be the last character of the string
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
        
        # convert to numeric
        categories[column] = pd.to_numeric(categories[column])
        
    # there are some values that are not 1 or 0 (e.g. 2) so this converts all to 1 and 0
    categories = categories.astype(bool).astype(int)
        
    # drop original categories column from merged df
    df = df.drop(['categories'], axis=1)
    
    # concatenate original merged df with the new 'categories' df
    df = pd.concat([df, categories], axis=1)
    
    # remove duplicates from this df
    df = df.drop_duplicates()
    
    return df
    
    


def save_data(df, database_filename):
    """
    INPUT
        cleaned dataframe
        
    OUTPUT
        updated database with set filename ready for ML pipeline
    """
    
    # instantiate sqlalchemy engine
    engine = create_engine('sqlite:///'+database_filename)
    
    # save dataframe as a table called 'message_cats'
    df.to_sql('message_cats', engine, index=False, if_exists='replace')
    
    pass
    

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()