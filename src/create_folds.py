import pandas as pd
from sklearn import model_selection

def kfold_val(df, k):
    '''
    :param df: pandas data frame
    :param k: k value for number of folds
    :return: randomizes data and creates a kfold column --> used for  kfold cross validation
    This split is used for creating a split to predict prices
    '''
    # sets the kfold column and randomizes data
    df['kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)

    # conducts cross validation
    kf = model_selection.KFold(n_splits=k)
    for f, (t, v) in enumerate(kf.split(X=df)):
        df.loc[v, 'kfold'] = f

    df.to_csv(f'..\input\price_{k}fold.csv', index=False)


def strat_kfold_val(df, k):
    '''
    :param df: pandas data frame
    :param k: k value for number of folds
    :return: randomizes data and creates a kfold column --> used for stratified kfold cross validation
    This split is used for creating a split to predict review scores
    '''

    # we create a new column called kfold and fill it with -1
    df["kfold"] = -1

    # randmoize rows of data
    df = df.sample(frac=1).reset_index(drop=True)

    # set target y value
    y = df['review_scores_rating']

    # initialize stratified 5 fold
    kf = model_selection.StratifiedKFold(n_splits=k)

    # fill kfold columns with values
    # this kf needs X and y vals (y val to maintain class ratio in folds)
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f

    # save to new csv
    df.to_csv('strat_train_folds.csv', index=False)

def split_train_test(df, target):
    '''
    :param df: pandas dataframe
    :param target: uses target value for future prediction as csv name
    :return: takes on of the kfolds and turns it into a holdout set for model performance. Creates a train and test csv
    '''

    test = df[df['kfold'] == 5].reset_index(drop=True)
    train = df[df['kfold'] != 5].reset_index(drop=True)

    train.to_csv(f'..\input\{target}_airbnb_train.csv', index=False)
    test.to_csv(f'..\input\{target}_airbnb_holdout_test.csv', index=False)

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)

    df = pd.read_csv('..\input\data.csv')
    kfold_val(df, 6)

    # creates a hold out set and splits the data into 2 csv
    df = pd.read_csv('..\input\price_6fold.csv')
    split_train_test(df, 'price')


