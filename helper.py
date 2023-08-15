import pandas as pd
from unidecode import unidecode
import numpy as np
import re
from typing import List

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from langdetect import detect

import pickle

from collections import Counter

import re
import nltk
from nltk.corpus import stopwords

def clip_outliers(df, col: str):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    upper_threshold = q3 + (1.5 * (q3 - q1))
    df[col] = df[col].clip(upper=upper_threshold)

def impute_knn(df, subset: str, text_columns: List):
    #divide data into 2
    data_complete = df.dropna(subset=subset).reset_index(drop = True)
    data_missing = df[df[subset].isnull()].reset_index(drop = True)
    #data_missing[subset] = data_missing.astype('str')
    
    #create tfidf vectorizer, join all the text in the text_columns into one string
    vectorizer = TfidfVectorizer()
    text_data = data_complete[text_columns].astype('str').apply(lambda x: ' '.join(x), axis=1)
    #create tfidf_matrix
    tfidf_matrix = vectorizer.fit_transform(text_data)
    
    #fit the model
    nn_model = NearestNeighbors(n_neighbors=3)  # Choose an appropriate value
    nn_model.fit(tfidf_matrix)
    for idx, row in data_missing.iterrows():
        #convert text_columns 
        row[text_columns] = row[text_columns].astype('str')
        #transform the text_columns in the data_missing into vector representation
        text_representation = vectorizer.transform([' '.join(row[text_columns])])
        #distance calculation
        _, indices = nn_model.kneighbors(text_representation)
        #get indices from the current row from data_complete
        neighbor_models = data_complete.loc[indices[0], subset]
        #calculate most frequent value
        imputed_data = neighbor_models.mode()[0]
        #impute the data
        data_missing.loc[idx, subset] = imputed_data
    
    data_complete = pd.DataFrame(np.concatenate((data_complete, data_missing), axis=0), columns = df.columns)
    return data_complete

def preprocess_df(df):
    df.drop(['Unnamed: 0', 'hull_shape'], axis = 1, inplace = True)
    df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
    
    #clip price and length
    cols = ['price', 'length']
    for col in cols:
        clip_outliers(df, col)
    
    #data imputation
    df_model = impute_knn(df, 'model', ['country', 'manufacturer', 'offerer', 'category', 'hull_material'])
    df_fuel = impute_knn(df_model, 'fuel_type', ['country', 'manufacturer', 'offerer', 'category', 'hull_material', 'model'])
    df_raw = df.copy()
    df = df_fuel
    
    df['word_count'] = df['description'].apply(lambda x: len(re.findall(r'\w+', x)))
    df = df[df['word_count'] > 20]
    df.drop(['word_count'], axis=1, inplace=True)
    
    df['description_language'] = df['description'].apply(lambda x:detect(x))
    df = df[df['description_language'] == 'en']
    df.drop(['description_language'], axis=1, inplace=True)
    
    df['price'] = df['price'].astype('float')
    df['year'] = df['year'].astype('float')
    df['length'] = df['length'].astype('float')
    
    df.drop(['name', 'location', 'offerer', 'id', 'manufacturer'], axis=1, inplace=True)
    
    return df

REPLACE_BY_SPACE_RE = re.compile("[/(){}\[\]\|@,;!]")
BAD_SYMBOLS_RE = re.compile("[^0-9a-z #+_]")
STOPWORDS_nlp = set(stopwords.words('english'))

#Custom Stoplist
stoplist = ["i","project","living","home",'apartment',"pune","me","my","myself","we","our","ours","ourselves","you","you're","you've","you'll","you'd","your",
            "yours","yourself","yourselves","he","him","his","himself","she","she's","her","hers","herself","it",
            "it's","its","itself","they","them","their","theirs","themselves","what","which","who","whom","this","that","that'll",
            "these","those","am","is","are","was","were","be","been","being","have","has","had","having","do","does","did",
            "doing","a","an","the","and","but","if","or","because","as","until","while","of","at","by","for","with","about",
            "against","between","into","through","during","before","after","above","below","to","from","up","down","in","out",
            "on","off","over","under","again","further","then","once","here","there","when","where","why","all","any",
            "both","each","few","more","most","other","some","such","no","nor","not","only","own","same","so","than","too",
            "very","s","t","can","will","just","don","don't","should","should've","now","d","ll","m","o","re","ve","y","ain",
            "aren","couldn","didn","doesn","hadn","hasn",
            "haven","isn","ma","mightn","mustn","needn","shan","shan't",
            "shouldn","wasn","weren","won","rt","rt","qt","for",
            "the","with","in","of","and","its","it","this","i","have","has","would","could","you","a","an",
            "be","am","can","edushopper","will","to","on","is","by","ive","im","your","we","are","at","as","any","ebay","thank","hello","know",
            "need","want","look","hi","sorry","http", "https","body","dear","hello","hi","thanks","sir","tomorrow","sent","send","see","there","welcome","what","well","us"]

STOPWORDS_nlp.update(stoplist)
    
def text_prepare(text: str):
    """
        text: a string
        
        return: modified initial string
    """
    
    text = text.replace("\d+"," ") # removing digits
    text = re.sub(r"(?:\@|https?\://)\S+", "", text) #removing mentions and urls
    text = text.lower() # lowercase text
    text =  re.sub('[0-9]+', '', text)
    text = REPLACE_BY_SPACE_RE.sub(" ", text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub(" ", text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join([word for word in text.split() if word not in STOPWORDS_nlp]) # delete stopwors from text
    text = text.strip()
    return text

def pos_counter(x: str,pos: str):
    """
    Returns the count for the given parts of speech tag
    
    NN - Noun
    VB - Verb
    JJ - Adjective
    RB - Adverb
    """
    tokens = nltk.word_tokenize(x.lower())
    tokens = [word for word in tokens if word not in STOPWORDS_nlp]
    text = nltk.Text(tokens)
    tags = nltk.pos_tag(text)
    counts = Counter(tag for word,tag in tags)
    return counts[pos]

def create_features(df):
    #prepare text data
    df['description'] = df['description'].astype('str').apply(text_prepare)
    
    #pos counter
    df['noun_counts'] = df['description'].apply(lambda x: pos_counter(x,'NN'))
    df['verb_counts'] = df['description'].apply(lambda x: (pos_counter(x,'VB')+pos_counter(x,'RB')))
    df['adjective_counts'] = df['description'].apply(lambda x: pos_counter(x,'JJ'))
    
    pos_cols = ['noun_counts', 'verb_counts', 'adjective_counts']
    for col in pos_cols:
        df[col] = df[col].astype('float')
    
    #load tfidf_vectorizer, transform the description column
    fileName = 'model/tfidf_vectorizer.pkl'
    with open(fileName,'rb') as f:
        tfidf_object = pickle.load(f)
        
    tfidf_matrix = tfidf_object.transform(df['description'])
    
    #get ngrams
    feature_names = tfidf_object.get_feature_names_out()
    tfidf_array = tfidf_matrix.toarray()
    tfidf_df = pd.DataFrame(tfidf_array, columns=feature_names)
    df = pd.concat([df.reset_index(drop=True),tfidf_df.reset_index(drop=True)],axis=1)
    
    for col in feature_names:
        df[col] = df[col].astype('float')
    
    #price-based features
    fileName = 'model/price_by_hull_material.pkl'
    with open(fileName,'rb') as f:
        price_by_hull_material = pickle.load(f)
    
    average_value_hull_material = sum(price_by_hull_material.values()) / len(price_by_hull_material)
    df['avg_price_by_hull_material'] = df['hull_material'].apply(lambda x: price_by_hull_material.get(x, average_value_hull_material))
    df['avg_price_by_hull_material'] = df['avg_price_by_hull_material'].astype('float')
    
    fileName = 'model/price_by_fuel_type.pkl'
    with open(fileName,'rb') as f:
        price_by_fuel_type = pickle.load(f)
    
    average_value_fuel_type = sum(price_by_fuel_type.values()) / len(price_by_fuel_type)
    df['avg_price_by_fuel_type'] = df['fuel_type'].apply(lambda x: price_by_fuel_type.get(x, average_value_fuel_type))
    df['avg_price_by_fuel_type'] = df['avg_price_by_fuel_type'].astype('float')
    
    fileName = 'model/price_by_category.pkl'
    with open(fileName,'rb') as f:
        price_by_category = pickle.load(f)
    
    average_value_category = sum(price_by_category.values()) / len(price_by_category)
    df['avg_price_by_category'] = df['category'].apply(lambda x: price_by_category.get(x, average_value_category))
    df['avg_price_by_category'] = df['avg_price_by_category'].astype('float')
    
    '''
    #one hot encode
    
    df = pd.get_dummies(df, columns=['category'], prefix='category', drop_first=True)
    column_names = df.columns
    category_columns = [col for col in column_names if col.startswith('category')]
    for col in category_columns:
        df[col] = df[col].astype('float')
    
    df = pd.get_dummies(df, columns=['fuel_type'], prefix='fuel', drop_first=True)
    column_names = df.columns
    fuel_columns = [col for col in column_names if col.startswith('fuel')]
    for col in fuel_columns:
        df[col] = df[col].astype('float')
    
    df = pd.get_dummies(df, columns=['hull_material'], prefix='hull', drop_first=True)
    column_names = df.columns
    hull_columns = [col for col in column_names if col.startswith('hull')]
    for col in hull_columns:
        df[col] = df[col].astype('float')
    
    df = pd.get_dummies(df, columns=['country'], prefix='country', drop_first=True)
    column_names = df.columns
    country_columns = [col for col in column_names if col.startswith('country')]
    for col in country_columns:
        df[col] = df[col].astype('float')
    
    #one_hot_cols = category_columns + fuel_columns + hull_columns + country_columns
    '''
    df.drop(['description'], axis=1,inplace=True)
    
    df = df.applymap(lambda x: x if not isinstance(x, str) or not has_non_ascii(x) else x.encode('ascii', 'ignore').decode('ascii'))
    string_categorical_cols = ['model', 'category', 'hull_material', 'country', 'fuel_type']
    for col in string_categorical_cols:
        df[col] = clean_columns(df[col].tolist())
    
    #cleaning column names
    df.columns = clean_columns(df.columns)
    column_names = df.columns
    
    '''
    category_columns = [col for col in column_names if col.startswith('category')]
    fuel_columns = [col for col in column_names if col.startswith('fuel')]
    hull_columns = [col for col in column_names if col.startswith('hull')]
    country_columns = [col for col in column_names if col.startswith('country')]
    
    
    one_hot_cols = category_columns + fuel_columns + hull_columns + country_columns
    '''

    return df, column_names

def clean_columns(column_list: List):
    all_cols = column_list
    
    modified_list = []

    for item in all_cols:
        item = str(item).lower()
        modified_item = re.sub(r'[^a-zA-Z0-9]', '_', item)
        modified_list.append(modified_item)
    
    final_list = []
    
    for i in modified_list:
        cleaned_column_name = re.sub(r'_+', '_', i)
        final_list.append(cleaned_column_name)
    
    final_list = [col.strip('_') for col in final_list]
        
    return final_list

def has_non_ascii(s):
    for char in s:
        if ord(char) > 127:
            return True
    return False