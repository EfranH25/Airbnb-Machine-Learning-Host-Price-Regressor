import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle


def visualize_numeric(df, target):
    """
    :param df: takes in single column dataframe of type numeric (float, int)
    :param target: name of the column the dataframe is looking at --> used for file name
    :return: Creates a distribution, histogram, box plot of numeric column. Also creates version w/o outliers
    """
    target_df = df
    target_df_no_outliers = target_df[target_df.between(target_df.quantile(.15), target_df.quantile(.85))]

    fig, axes = plt.subplots(1, 3, sharex=True, figsize=(10, 5))
    fig.suptitle(f'{target} Plots')
    sns.distplot(target_df, ax=axes[0], hist=False)
    axes[0].set_title('Distrubtion')
    sns.histplot(target_df, ax=axes[1], bins=15)
    axes[1].set_title('Histogram')
    sns.boxplot(target_df, ax=axes[2])
    axes[2].set_title('Boxplot')
    fig.savefig(f'..\\figures\{target} figures.png')

    fig, axes = plt.subplots(1, 3, sharex=True, figsize=(10, 5))
    fig.suptitle(f'{target} Plots without Outliers')
    sns.distplot(target_df_no_outliers, ax=axes[0], hist=False)
    axes[0].set_title('Distrubtion')
    sns.histplot(target_df_no_outliers, ax=axes[1], bins=15)
    axes[1].set_title('Histogram')
    sns.boxplot(target_df_no_outliers, ax=axes[2])
    axes[2].set_title('Boxplot')
    plt.show()
    fig.savefig(f'..\\figures\{target} figures without outliers.png')

    print('Models Created')


def explore_numerics(df):
    """
    :param df: takes in a dataframe with only numeric features
    :return:
        - loops throw each column and provides view of top 10 rows, number of nulls, descriptive stats
        - asks if you want to add column to list of columns to drop later (drop_list)
        - asks if you want to add column to list of columns with low std (low_std)
            - if no:
                - asks if you want to create visualizations for numeric data
        - saves drop_list and low_std for later use on dropping certain features of feature engineering
    """

    drop_list = []
    low_std = []
    for col in df.columns:
        print(col)
        print(df[col].head(10))
        print(df[col].isnull().value_counts())
        print(df[col].describe())

        action_input = input()
        if action_input == 'd':
            drop_list.append(col)
        if action_input == 's':
            low_std.append(col)
        if action_input == 'v':
            visualize_numeric(df[col], col)
        if action_input == 'x':
            break
        print('+++++++++++++++++++++++++++++++++++++++++++++')

    print(drop_list)
    file_name = '../tracking lists/no longer needed/drop_list.pkl'
    open_file = open(file_name, 'wb')
    pickle.dump(drop_list, open_file)
    open_file.close()

    print(low_std)
    file_name = '../tracking lists/low_std.pkl'
    open_file = open(file_name, 'wb')
    pickle.dump(low_std, open_file)
    open_file.close()

def check_corr(df, columns):
    """
    :param df: data frame
    :param columns: numeric columns to view
    :return: creates heatmap of all numeric variables
    """

    file_name = '../tracking lists/no longer needed/drop_list.pkl'
    open_file = open(file_name, 'rb')
    drop_list = pickle.load(open_file)
    open_file.close()

    print(drop_list)

    df_num = df[columns].drop(
        drop_list,
        axis=1)

    print(df_num.corr())
    plt.figure(figsize=(15, 10))
    sns.heatmap(df_num.corr(), annot=False, linewidths=.5, cmap='coolwarm').get_figure()
    plt.show()


def explore_cat(df):
    """
    :param df: pandas dataframe with only categorical/obj features selected
    :return: outputs column information and checks if column is
        - categorical: then changes type to category, makes visualization and saves it
        - not important: adds it to cat_drop_list to drop later
        - numeric: adds to cat_to_num_list to convert to numeric later
        - date: adds to date_list to convert to date later
        - list of items: adds to expand_feature to convert to other categorical features later
    """
    cat_drop_list = []
    cat_to_num_list = []
    date_list = []
    expand_feature = []

    for col in df.columns:
        print(df[col].describe())
        print(df[col].value_counts(normalize=False))
        print(df[col].value_counts(normalize=True))

        action_input = input()
        if action_input == 'v':
            df[col] = df[col].astype('category')
            visualize_cat(df[col], col)
        if action_input == 'd':
            cat_drop_list.append(col)
        if action_input == 'e':
            expand_feature.append(col)
        if action_input == 'c':
            cat_to_num_list.append(col)
        if action_input == 't':
            date_list.append(col)
        if action_input == 'x':
            break
        print('+++++++++++++++++++++++++++++++++++++++++++++')

    print(cat_drop_list)
    file_name = '../tracking lists/no longer needed/cat_drop_list.pkl'
    open_file = open(file_name, 'wb')
    pickle.dump(cat_drop_list, open_file)
    open_file.close()

    print(cat_to_num_list)
    file_name = '../tracking lists/cat_to_num_list.pkl'
    open_file = open(file_name, 'wb')
    pickle.dump(cat_to_num_list, open_file)
    open_file.close()

    print(date_list)
    file_name = '../tracking lists/date_list.pkl'
    open_file = open(file_name, 'wb')
    pickle.dump(date_list, open_file)
    open_file.close()

    print(expand_feature)
    file_name = '../tracking lists/expand_feature.pkl'
    open_file = open(file_name, 'wb')
    pickle.dump(expand_feature, open_file)
    open_file.close()


def visualize_cat(df, target):
    """
    :param df: takes in dataframe w/ single categorical feature
    :param target: name of the column, used name figure
    :return: creates visualization of data
    """
    target_df = df

    plt.figure(figsize=(10, 5))
    fig = sns.barplot(target_df.value_counts().index, target_df.value_counts())
    plt.show()

    fig.get_figure().savefig(f'../figures/categorical/{target} figure.png')


def open_saved_list(file_name):
    """
    :param file_name: file path of desired list to retrieve
    :return: returns on the the saved lists (should be in tracking lists folder)
    """
    open_file = open(file_name, 'rb')
    loaded_list = pickle.load(open_file)
    open_file.close()
    return loaded_list


def select_final_features():
    """
    :return: opens up the drop_list (from explore numerics) and  cat_drop_list (from explore cat) and combines them to
    one list and saves it --> used to drop unneeded columns (decided on after run explore num and explore cat)
    """
    drop_numerics = open_saved_list('../tracking lists/no longer needed/drop_list.pkl')
    drop_cats = open_saved_list('../tracking lists/no longer needed/cat_drop_list.pkl')

    file_name = '../tracking lists/combined_drop_list.pkl'

    open_file = open(file_name, 'wb')
    pickle.dump(drop_numerics + drop_cats, open_file)
    open_file.close()


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 100)

    df = pd.read_csv('..\input\price_airbnb_train.csv')

    # visualize distribution of target variable
    # visualize_numeric(df['price'], 'price')

    num_features = df.select_dtypes(include=['float64', 'int64']).columns
    cat_feat = df.select_dtypes(exclude=['float64', 'int64']).columns

    # explore_numerics(df[num_features])
    # check_nulls_cols(df, num_features)
    # check_corr(df, num_features)

    # explore_cat(df[cat_feat])

    # select_final_features()
