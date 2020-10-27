import os
import argparse
import joblib

# import libraries for manipulating/controlling/viewing data
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
import csv

# import libraries to evaluate data/summarize data
from scipy import stats
from sklearn import metrics
import numpy as np
import math

# import libraries for feature and model selection
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.decomposition import PCA
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.impute import SimpleImputer

# import ML models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb

# import libraries for hyper-parameter optimization
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate


# =============================================== FUNCTIONS USED FOR PREPROCESSING
def fix_cats_to_numerics(df_cat_num):
    """
    :param df_cat_num: pandas data frame
    :return: using a list of columns names, changes certain categorical features in df to numerics
    """

    # opens file that contains list of name of columns that are categorical but should be features
    open_file = open('../tracking lists/cat_to_num_list.pkl', 'rb')
    cat_to_num_cols = pickle.load(open_file)
    open_file.close()

    # changes strings integers w/ % symbol to float integers
    for col in cat_to_num_cols:
        if col in df_cat_num.columns:
            df_cat_num[col] = df_cat_num[col].str.rstrip('%').astype('float')

    # since percentages where the only features that needed to be converted to floats, no further steps needed
    # besides returning df
    return df_cat_num


def fix_dates(df_dates):
    """
    :param df_dates: pandas dataframe
    :return: converts string dates into a specific date time int --> in this case, the number of days from '10/23/2020'
    """

    # opens file containing the name of columns that have data values
    open_file = open('../tracking lists/date_list.pkl', 'rb')
    date_cols = pickle.load(open_file)
    open_file.close()

    # created today's date column used for comparison --> can change to something else but I just used today's date
    # converted string to date time obj
    df_dates['today_date'] = '10/23/2020'
    df_dates['today_date'] = pd.to_datetime(df_dates['today_date'], format='%m/%d/%Y')

    # loops through list of date columns
    for col in date_cols:
        # first converts date string to readable date format --> then does calculations into a new column
        if col in df_dates.columns:
            df_dates[col] = pd.to_datetime(df_dates[col], format='%m/%d/%Y')
            df_dates[f'{col}_since_10/23/2020'] = (df_dates['today_date'] - df_dates[col]).dt.days
            # drops date column since no longer needed after being adjusted into new column
            df_dates = df_dates.drop([col], axis=1)

    # column no longer needed, so dropped
    df_dates = df_dates.drop(['today_date'], axis=1)
    return df_dates


def fix_baths(df_bath):
    # not in use at the momment
    """
    :param df_bath: takes dataframe
    :return: fixes the bathrooms_text string feature (which contains an int and bathroom type string) and converts
    to 2 separate features --> then drops bathroom_text column since no longer needed
    """

    df_bath['bathrooms_text'] = df_bath['bathrooms_text'].str.split(' ', n=1)

    # bathroom_num is feature containing number of bathrooms, bathroom_type is feature containing type of bathrooms
    # note: for bathroom num
    df_bath['bathroom_num'] = df_bath['bathrooms_text'].str[0]
    df_bath['bathroom_type'] = df_bath['bathrooms_text'].str[1]

    # df_bath['bathroom_num'] = df_bath['bathroom_num'].astype(float)

    df_bath = df_bath.drop(['bathrooms_text'], axis=1)
    return df_bath


def expand_features(df_expand):
    """
    :param df: pass in training df and test df
    - opens expand_feature list to use as columns to get list of features that have records with list of items in 
    them --> makes them into individual binary features
    :return: creates new binary columns on data based on tags found --> uses those columns to make new binary
    features for  test data. Also returns list of tags for the new columns added
    """

    # Assumption: tags for host verification and amenities are in a least in the 'host_verifications' and 'amenities'
    # columns respectively. Assume original database had a column for each tag rather than compiling them into a list.
    # This function transforms those tags back into own individual columns. It does this for the entire
    # df (so it will be doing it for the test data as well). I do not consider this data leakage under
    # the assumption the original database had features for all those tags, I'm just converting it back to that state
    import ast

    # open expand_feature.pkl to get list of features names that I marked for expands
    # this list contains the name of features that have records with lists of stings/tags
    open_file = open('../tracking lists/expand_feature.pkl', 'rb')
    expand_feat_cols = pickle.load(open_file)
    open_file.close()

    all_tags = []
    # loops through each feature from expand_feat_cols
    for engi_col in expand_feat_cols:
        # creates a set that will contain the name of all the unique tags found in column
        # these will the be new binary features added later
        if engi_col in df_expand.columns.tolist():
            new_cols = set({})
            for row in df_expand[engi_col]:
                for r in ast.literal_eval(row):
                    new_cols.add(r)

            new_cols = list(new_cols)
            new_cols.sort()  # had to sort or else order of columns would be random
            # new_cols.append(engi_col + '_rare_tags') --> may implement later for columns handling rare tags

            # loops through the new_cols list (i.e. name of new binary features)
            for check in new_cols:
                # creates new binary feature based on appearance of tag in training data[engi_col] column
                df_expand[engi_col + '_' + check] = 0
                df_expand.loc[df_expand[engi_col].str.contains(check), engi_col + '_' + check] = 1

                # adds newly added column to all tags list
                all_tags.append(str(engi_col + '_' + check))

        # drops column since no longer needed
        if engi_col in df_expand.columns:
            df_expand = df_expand.drop([engi_col], axis=1)
    return df_expand, all_tags


def col_majority_null(df_null):
    """
    :param df_null: takes in dataframe
    :return: returns list of all columns that have a majority null value
    """
    null_list = []
    for col in df_null.columns:
        null_ratio = (df_null[col].isnull().sum() / len(df_null)) * 100
        if null_ratio >= 50:
            null_list.append(col)
    return null_list


def low_varience(df_var):
    """
    :param df_var: dataframe with only numeric values
    :return: returns list of columns with 0 variance
    """

    drop_col = []
    for col in df_var.columns:
        varience = df_var[col].var()
        if varience == 0:
            drop_col.append(col)
    return drop_col


def preprocess_model(fold, remove_outliers=False, model='NONE'):
    """
    :param fold: the fold position to train on
    :param remove_outliers: flag to remove outliers during cleaning phase
    :param model:
    :return:
    """
    df = pd.read_csv('..\input\price_airbnb_train.csv')

    # ======================================================= DATA CLEANING PHASE

    # opens up a list of saved columns (combined_drop_list) which has a bunch of columns no longer needed
    # drops those columns from df
    open_file = open('../tracking lists\combined_drop_list.pkl', 'rb')
    drop_cols = pickle.load(open_file)
    open_file.close()
    df = df.drop(drop_cols, axis=1)

    # these values are considered numerics by pandas --> changed them to no longer be numerics
    # setting them to categorical here gives an error so I chose string instead
    df['host_id'] = df['host_id'].astype(str)
    df['latitude'] = df['latitude'].astype(str)
    df['longitude'] = df['longitude'].astype(str)

    # convert categorical features to numerics
    df = fix_cats_to_numerics(df)

    # change dates columns to be number of days since 10/23/2020 (chosen date arbitrary, just used date at the time)
    df = fix_dates(df)

    # fix bathroom columns (UNUSED --> some bugs so just treating bathrooms_text as categorical for now)
    # df = fix_baths(df)

    # expand feature columns
    df, tags_list = expand_features(df)

    # ======================================================= FEATURE ENGINEERING PHASE

    # list of columns I decided to drop because too collinear w/ other features or not needed
    drop_test = ['host_id', 'host_name', 'host_location', 'host_neighbourhood', 'host_listings_count',
                 'neighbourhood', 'latitude', 'longitude', 'minimum_nights', 'maximum_nights', 'minimum_minimum_nights',
                 'maximum_minimum_nights', 'minimum_maximum_nights', 'first_review_since_10/23/2020',
                 'neighbourhood_cleansed', 'calculated_host_listings_count_shared_rooms',
                 'maximum_maximum_nights', 'availability_30', 'availability_60', 'availability_90',
                 'number_of_reviews_ltm', 'number_of_reviews_l30d', 'review_scores_accuracy',
                 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication',
                 'review_scores_location', 'review_scores_value', 'license', 'calculated_host_listings_count',
                 'calculated_host_listings_count_entire_homes', 'calculated_host_listings_count_private_rooms',
                 'reviews_per_month']

    df = df.drop(drop_test, axis=1)

    # remove outliers if remove_outliers flag is True
    if remove_outliers:
        # NOTE: Not sure to remove outliers from entire DF or only for train --> removing from entire DF for now
        # sets quantile range for outlier detection
        Q1 = df.quantile(0.15)
        Q3 = df.quantile(0.85)
        IQR = Q3 - Q1

        # selects all numeric features to check for outliers and then removes outliers
        # NOTE: tags from tag_list are used as binary variables but pandas sees them as int64,
        # so exclude them before outlier elimination
        df_holder = df.drop(tags_list, axis=1).reset_index(drop=True)
        num_feat = list(df_holder.select_dtypes(include=['float64', 'int64']).columns)
        df = df[~((df[num_feat] < (Q1 - 1.5 * IQR)) | (df[num_feat] > (Q3 + 1.5 * IQR))).any(axis=1)].reset_index(
            drop=True)

    # get train / validation sets and drops kfold since not needed anymore
    df_train = df[df['kfold'] != fold].reset_index(drop=True)
    df_valid = df[df['kfold'] == fold].reset_index(drop=True)
    df_train = df_train.drop(['kfold'], axis=1)
    df_valid = df_valid.drop(['kfold'], axis=1)

    # get numeric and categorical features
    # do not include tag list for numerics or categories since already converted to binary data
    num_feat = list(df_train.drop(tags_list, axis=1).select_dtypes(include=['float64', 'int64']).columns)
    cat_feat = list(df_train.select_dtypes(exclude=['float64', 'int64']).columns)

    # remove columns w/ majority null values
    mostly_null_list = col_majority_null(df_train)
    df_train = df_train.drop(mostly_null_list, axis=1)
    df_valid = df_valid.drop(mostly_null_list, axis=1)

    # set categorical columns w/ some nulls to 'NONE'
    df_train[cat_feat] = df_train[cat_feat].fillna('NONE')
    df_valid[cat_feat] = df_valid[cat_feat].fillna('NONE')

    # converts at cat variables to categories (just in case they weren't)
    # df_train[cat_feat] = df_train[cat_feat].astype('category')
    # df_valid[cat_feat] = df_valid[cat_feat].astype('category')
    # remove columns w/ very low std --> may implement later

    # set numeric columns w/ some nulls to average of training
    # NOTE: My impute null values in the future
    df_train[num_feat] = df_train[num_feat].fillna(df_train[num_feat].mean())
    df_valid[num_feat] = df_valid[num_feat].fillna(df_train[num_feat].mean())

    # normalize numerics --> doesn't use target value b/c I don't want to normalize it
    scaler = MinMaxScaler()
    num_feat.remove('price')
    df_train[num_feat] = scaler.fit_transform(df_train[num_feat])
    df_valid[num_feat] = scaler.fit_transform(df_valid[num_feat])
    num_feat.append('price')

    # get ordinal categorical features
    lbl_encode_feat = ['host_response_time']

    # get nominal categorical features i.e. all features in cat_feat that aren't ordinal categories
    oh_encode_feat = [cat for cat in cat_feat if cat not in lbl_encode_feat]

    # label encode ordinal features
    lbl_encoder = LabelEncoder()
    for feat in lbl_encode_feat:
        df_train[feat] = lbl_encoder.fit_transform(df_train[feat])
        df_valid[feat] = lbl_encoder.fit_transform(df_valid[feat])

    # one-hot encode nominal features
    # this implementation could be cause data leakage?
    df_train_ohe = pd.get_dummies(df_train[oh_encode_feat], drop_first=True)
    df_valid_ohe = pd.get_dummies(df_valid[oh_encode_feat], drop_first=True)

    # makes sure hot encoded valid and train have same number of columns
    missing_cols = set(list(df_train_ohe.columns) + list(df_valid_ohe.columns))

    for col in missing_cols:
        if col not in df_train_ohe.columns:
            df_train_ohe[col] = 0
        if col not in df_valid_ohe.columns:
            df_valid_ohe[col] = 0

    # sorts in columns so they stay consistent per each trial (b/c sets don't keep a certain order for missing_cols)
    df_train_ohe = df_train_ohe.sort_index(axis=1)
    df_valid_ohe = df_valid_ohe.sort_index(axis=1)

    # drop oh_encode_feat columns --> add replace new one-hot encoded columns
    df_train = df_train.drop(oh_encode_feat, axis=1)
    df_valid = df_valid.drop(oh_encode_feat, axis=1)
    df_train = pd.concat([df_train, df_train_ohe], axis=1)
    df_valid = pd.concat([df_valid, df_valid_ohe], axis=1)

    # drop all columns w/ 0 varience since they are constants
    no_var = low_varience(df_train)
    df_train = df_train.drop(no_var, axis=1)
    df_valid = df_valid.drop(no_var, axis=1)
    return df_train, df_valid


# =============================================== FEATURE SELECTION
def simple_ln_reg(X, y, sel_method=None, fit_intercept=True, normalize=False):
    # Linear Regression
    ln_reg = LinearRegression(fit_intercept=True, normalize=False)
    para = f'fit_intercept = {fit_intercept}, \n normalize = {normalize}'

    cv_scores = cross_validate(ln_reg, X, y, scoring=['r2', 'neg_root_mean_squared_error'])

    r2 = cv_scores['test_r2'].mean()
    rmse = cv_scores['test_neg_root_mean_squared_error'].mean()
    entry = {'Model Name': 'Linear Regression', 'Selection': sel_method, 'Parameters': para, 'R2 Score': r2,
             'RMSE Score': rmse}

    score_sheet = pd.read_csv('../records/Score Sheet.csv')
    score_sheet = score_sheet.append(entry, ignore_index=True)
    score_sheet.to_csv('../records/Score Sheet.csv', index=False, line_terminator='\n')


def simple_lasso(X, y, sel_method=None, alpha=1.0, normalize=False):
    # Lasso Regression
    lasso = Lasso(alpha=alpha, normalize=normalize)
    para = f'alpha = {alpha}, \n normalize = {normalize}'

    cv_scores = cross_validate(lasso, X, y, scoring=['r2', 'neg_root_mean_squared_error'])

    r2 = cv_scores['test_r2'].mean()
    rmse = cv_scores['test_neg_root_mean_squared_error'].mean()
    entry = {'Model Name': 'Lasso Regression', 'Selection': sel_method, 'Parameters': para, 'R2 Score': r2,
             'RMSE Score': rmse}

    score_sheet = pd.read_csv('../records/Score Sheet.csv')

    score_sheet = score_sheet.append(entry, ignore_index=True)
    score_sheet.to_csv('../records/Score Sheet.csv', index=False, line_terminator='\n')


def simple_svm(X, y, sel_method=None, C=1, gamma='scale'):
    # SVM
    svr = SVR(C=C, gamma=gamma)
    para = f'C = {C}, \n gamma = {gamma}'

    cv_scores = cross_validate(svr, X, y, scoring=['r2', 'neg_root_mean_squared_error'])
    r2 = cv_scores['test_r2'].mean()
    rmse = cv_scores['test_neg_root_mean_squared_error'].mean()

    entry = {'Model Name': 'SVR', 'Selection': sel_method, 'Parameters': para, 'R2 Score': r2,
             'RMSE Score': rmse}

    score_sheet = pd.read_csv('../records/Score Sheet.csv')

    score_sheet = score_sheet.append(entry, ignore_index=True)
    score_sheet.to_csv('../records/Score Sheet.csv', index=False, line_terminator='\n')


def simple_knn(X, y, sel_method=None, n_neighbors=5, p=2):
    # KNN
    knnr = KNeighborsRegressor(n_neighbors=n_neighbors, p=p)

    para = f'n_neighbors = {n_neighbors}, \n p = {p}'

    cv_scores = cross_validate(knnr, X, y, scoring=['r2', 'neg_root_mean_squared_error'])
    r2 = cv_scores['test_r2'].mean()
    rmse = cv_scores['test_neg_root_mean_squared_error'].mean()

    entry = {'Model Name': 'KNN', 'Selection': sel_method, 'Parameters': para, 'R2 Score': r2,
             'RMSE Score': rmse}

    score_sheet = pd.read_csv('../records/Score Sheet.csv')

    score_sheet = score_sheet.append(entry, ignore_index=True)
    score_sheet.to_csv('../records/Score Sheet.csv', index=False, line_terminator='\n')


def simple_tree(X, y, sel_method=None, criterion='mse', max_depth=None, random_state=0):
    # Decision Tree
    tree = DecisionTreeRegressor(criterion='mse', max_depth=None, random_state=0)

    para = f'criterion = {criterion}, \n max_depth = {max_depth}'

    cv_scores = cross_validate(tree, X, y, scoring=['r2', 'neg_root_mean_squared_error'])
    r2 = cv_scores['test_r2'].mean()
    rmse = cv_scores['test_neg_root_mean_squared_error'].mean()

    entry = {'Model Name': 'Tree', 'Selection': sel_method, 'Parameters': para, 'R2 Score': r2,
             'RMSE Score': rmse}

    score_sheet = pd.read_csv('../records/Score Sheet.csv')

    score_sheet = score_sheet.append(entry, ignore_index=True)
    score_sheet.to_csv('../records/Score Sheet.csv', index=False, line_terminator='\n')


def simple_forest(X, y, sel_method=None, n_estimators=100, max_depth=None, min_impurity_decrease=0,
                  min_samples_leaf=1, max_features='auto'):
    # Random Forest
    rforest = RandomForestRegressor(random_state=0, n_estimators=n_estimators, max_depth=max_depth,
                                    min_impurity_decrease=min_impurity_decrease, min_samples_leaf=min_samples_leaf,
                                    max_features=max_features, criterion='mse')

    para = f'n_estimators = {n_estimators}, \n max_depth = {max_depth}, ' \
           f'min_impurity_decrease = {min_impurity_decrease}, \n min_samples_leaf = {min_samples_leaf}, ' \
           f'max_features = {max_features}'

    cv_scores = cross_validate(rforest, X, y, scoring=['r2', 'neg_root_mean_squared_error'])
    r2 = cv_scores['test_r2'].mean()
    rmse = cv_scores['test_neg_root_mean_squared_error'].mean()

    entry = {'Model Name': 'Random Forest', 'Selection': sel_method, 'Parameters': para, 'R2 Score': r2,
             'RMSE Score': rmse}

    score_sheet = pd.read_csv('../records/Score Sheet.csv')

    score_sheet = score_sheet.append(entry, ignore_index=True)
    score_sheet.to_csv('../records/Score Sheet.csv', index=False, line_terminator='\n')


def simple_xgb(X, y, sel_method=None):
    # XGBoost
    xg = xgb.XGBRegressor()

    para = f'default'

    cv_scores = cross_validate(xg, X, y, scoring=['r2', 'neg_root_mean_squared_error'])
    r2 = cv_scores['test_r2'].mean()
    rmse = cv_scores['test_neg_root_mean_squared_error'].mean()

    entry = {'Model Name': 'XGBoost', 'Selection': sel_method, 'Parameters': para, 'R2 Score': r2,
             'RMSE Score': rmse}

    score_sheet = pd.read_csv('../records/Score Sheet.csv')

    score_sheet = score_sheet.append(entry, ignore_index=True)
    score_sheet.to_csv('../records/Score Sheet.csv', index=False, line_terminator='\n')


def ML_feature_selection(df_train, df_valid, method='sel', model_name='ln_reg', score_type='r2',
                         min=5, max=15, run_baseline=False):
    """
    :param X_train: X training data
    :param y_train: y training target
    :param X_train: X validation data
    :param y_train: y validation target
    :param method: type of feature selection to perform:
        - 'forward': runs forward selection
        - 'backward': runs backward elimination
        - 'step': runs step-wise selection
        - 'sel': uses the model_name variable selected for feature selection --> default
        - 'rfe': uses recursive method for feature selection
    :param model_name: select type of model to use evaluation for
        - 'ln_reg': uses linear regression to evaluation features --> Default
        - 'rnd_for_reg': uses random forest regression
        - 'dec_tree': uses decision tree regression
        - 'grd_boost': uses gradient boost regression
    :param score_type: select scoring method for picking best features
        - 'r2': uses r2 to score features --> Default
    :param min: determines --> used as custom parameter for some of the ML Models
    :param max: determines --> used as custom parameter for some of the ML Models
    :param run_baseline: run selected feature again various base ML model
    :return: returns numpy arraus of df_train, df_valid (X_train_sfs, y_train, X_valid_sfs, y_valid) with the best
    features selected
    """

    # selects model selection type and evaluation method based on inputs
    select_model = {
        'ln_reg': LinearRegression(normalize=True, fit_intercept=True),
        'rnd_for_reg': RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1, max_depth=5),
        'dec_tree': DecisionTreeRegressor(random_state=0, max_depth=5),
        'grd_boost': GradientBoostingRegressor(n_estimators=100, random_state=0, max_depth=5),
        'lasso': Lasso(random_state=0, normalize=True),
        'ridge': Ridge(random_state=0, normalize=True)
    }
    model = select_model[model_name]
    select_method = {
        'forward': SFS(
            model, k_features=(min, max), forward=True, floating=False, scoring=score_type, cv=0
        ),
        'backward': SFS(
            model, k_features=(min, max), forward=False, floating=False, scoring=score_type, cv=0
        ),
        'step': SFS(
            model, k_features=(min, max), forward=True, floating=True, cv=0
        ),
        'sel': SelectFromModel(
            model,
            max_features=max
        ),
        'rfe': RFE(
            model, n_features_to_select=max
        )
    }
    sfs = select_method[method]

    df_train_x = df_train.drop(['price'], axis=1)
    df_train_y = df_train['price']

    df_valid_x = df_valid.drop(['price'], axis=1)
    df_valid_y = df_valid['price']

    # fit model
    sfs.fit(df_train_x, df_train_y)

    X_train_sfs = sfs.transform(df_train_x)
    y_train = df_train_y.to_numpy()
    X_valid_sfs = sfs.transform(df_valid_x)
    y_valid = df_valid_y.to_numpy()

    method = method + ' ' + model_name

    # run through basic ml regression models (w/o parameter tuning) to get baseline results for each selection version
    # used to get idea of how well each selection performs
    # saves results to csv
    if run_baseline:
        simple_ln_reg(X_train_sfs, y_train, method)
        simple_lasso(X_train_sfs, y_train, method)
        simple_svm(X_train_sfs, y_train, method)
        simple_knn(X_train_sfs, y_train, method)
        simple_tree(X_train_sfs, y_train, method)
        simple_forest(X_train_sfs, y_train, method)
        simple_xgb(X_train_sfs, y_train, method)

    return X_train_sfs, y_train, X_valid_sfs, y_valid


# =============================================== TUNE ML MODELS
def train_tune_model(X_train, y_train, model_name='ln_reg', search_type='GS'):
    # sets x and y inputs for train, valid sets to put into model
    if model_name == 'ln_reg':
        clf = LinearRegression()
        param_grid = {
            'normalize': [True, False],
            'fit_intercept': [True, False],
        }
    elif model_name == 'knn':
        clf = KNeighborsRegressor()
        param_grid = {
            'n_neighbors': range(3, 16),
            'p': [2, 3]
        }
    elif model_name == 'ridge':
        clf = Ridge()
        param_grid = {
            'alpha': [0.02, 0.024, 0.025, 0.026, 0.03]

        }
    elif model_name == 'lasso':
        clf = Lasso()
        param_grid = {
            'alpha': [200, 230, 250, 265, 270, 275, 290, 300, 500],
            'normalize': [True, False],
        }
    elif model_name == 'tree':
        clf = DecisionTreeRegressor(random_state=0)
        param_grid = {
            'splitter': ['best', 'random'],
            'max_depth': np.arange(1, 31),
        }
    elif model_name == 'rn_for_reg':
        clf = RandomForestRegressor(random_state=0, n_jobs=-1)
        param_grid = {
            'n_estimators': np.arange(100, 1500, 100),
            'max_depth': np.arange(1, 31),
        }
    elif model_name == 'xgb':
        clf = xgb.XGBRegressor()
        param_grid = {
            'min_child_weight': [1, 5, 10],
            'gamma': [0.5, 1, 1.5, 2, 5],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'max_depth': [3, 4, 5]
        }

    if search_type == 'GS':
        model = GridSearchCV(
            estimator=clf,
            param_grid=param_grid,
            scoring='r2',
            verbose=10,
            n_jobs=1,
            cv=5
        )

    model.fit(X_train, y_train)

    r2 = model.best_score_
    params = []
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        params.append([param_name, best_parameters[param_name]])

    entry = {
        'Model Name': model_name,
        'Parameters': params,
        'R2 Score': r2
    }
    score_sheet = pd.read_csv('../records/Model Score Sheet.csv')
    score_sheet = score_sheet.append(entry, ignore_index=True)
    score_sheet.to_csv('../records/Model Score Sheet.csv', index=False, line_terminator='\n')

    # FUTURE: Implement Pipeline possible?


# NOTE: For hyper-parameter use grid search and randomized search w/ pipelines

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 100)

    print('START')
    # preprocess data --> get train and validation sets
    df_train, df_valid = preprocess_model(0, remove_outliers=True)


    # flag to first run some tests to determine which model selection method to use
    # also used to get a baseline score for all regression models
    run_basic_tests = False
    if run_basic_tests:
        selection_list = [
            'ln_reg',
            'dec_tree',
            'lasso',
            'ridge'
        ]
        for method in selection_list:
            ML_feature_selection(df_train, df_valid, method='rfe', model_name=method, show_plot=True)

    X_train, y_train, X_valid, y_valid = ML_feature_selection(df_train, df_valid, method='sel',
                                                              model_name='dec_tree', max=15)
    train_tune_model(X_train, y_train, model_name='xgb', search_type='GS')
    print('DONE')
