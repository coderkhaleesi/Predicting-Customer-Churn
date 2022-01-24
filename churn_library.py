# library doc string
"""
    This package is for predicting credit card customers most likely to churn.

    ________________________________________________________________________________

    *import_data - imports the customer data
    *perform_eda - for performing exploratory data analysis
    *encoder_helper - for encoding categorical data
    *perform_feature_engineering - for turning raw data into features as input to the model
    *classification_report_image - produces classification report and saves it
    *feature_importance_plots - produces and stores feature importance plots
    *train_models - trains and saves models

"""

import os

# import libraries
import joblib

# data manipulation imports
import pandas as pd
import numpy as np

# Imports for preprocessing
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

# Imports for modeling
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# imports for metrics
from sklearn.metrics import plot_roc_curve, classification_report

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df_data: pandas dataframe
    '''
    df_data = pd.read_csv(pth)
    return df_data


def perform_eda(df_data):
    '''
    perform eda on df_data and save figures to images folder

    input:
            df_data: pandas dataframe

    output:
            None
    '''

    # check and create directories
    if not os.path.exists(os.path.join(os.getcwd(), 'images')):
        os.makedirs('images')

    if not os.path.exists(os.path.join(os.getcwd(), 'images', 'eda')):
        os.makedirs(os.path.join('images', 'eda'))

    # plot init
    plt.figure(figsize=(20, 10))

    # Churn hist plot and save
    df_data['Churn'].hist()
    plt.savefig(os.path.join('images', 'eda', 'churn_fig.png'))

    # Customer age hist plot and save
    df_data['Customer_Age'].hist()
    plt.savefig(os.path.join('images', 'eda', 'age_fig.png'))

    # Marital status bar plot and save
    df_data.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(os.path.join('images', 'eda', 'marital_status_fig.png'))

    # Total transaction count  plot and save
    sns.distplot(df_data['Total_Trans_Ct'])
    plt.savefig(os.path.join('images', 'eda', 'total_trans_ct_fig.png'))

    # Correlation heatmap plot and save
    sns.heatmap(df_data.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(os.path.join('images', 'eda', 'heatmap_fig.png'))


def encoder_helper(df_data, category_lst, response=None):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df_data: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be
                                used for naming variables or index y column]

    output:
            df_data: pandas dataframe with new columns for categorical columns
    '''


    for category in category_lst:
        category_df_data_list = []
        groups = df_data.groupby(category).mean()['Churn']

        for val in df_data[category]:
            category_df_data_list.append(groups.loc[val])

        new_category_name = category + "_" + "Churn"

        df_data[new_category_name] = category_df_data_list

    return df_data


def perform_feature_engineering(df_data, response=None):
    '''
    helper function to perform feature engineering and return a new dataframe that is
    ready for analysis

    input:
              df_data: pandas dataframe
              response: string of response name [
                        optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    X = pd.DataFrame()
    X[keep_cols] = df_data[keep_cols]
    y = df_data['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    plt.figure()
    plt.rc('figure', figsize=(8, 8))
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
        'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(
        0.01, 0.6, str('Random Forest Test (below) Random Forest Train (above)'), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
        'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')

    if not os.path.exists(os.path.join(os.getcwd(), 'images', 'results')):
        os.makedirs(os.path.join('images', 'results'))
    plt.savefig(os.path.join('images', 'results', 'rf_results.png'))
    plt.close()

    plt.figure()
    plt.rc('figure', figsize=(8, 8))
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
        'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(
        0.01, 0.6, str('Logistic Regression Test (below) Logistic Regression Train (above)'), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
        'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')

    plt.savefig(os.path.join('images', 'results', 'lr_results.png'))
    plt.close()


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    plt.bar(range(X_data.shape[1]), importances[indices])
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    plt.savefig(output_pth)


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    # fitting two classifiers - random forest and logistic regression
    random_forest_classifier = RandomForestClassifier(random_state=42)
    logistic_reg_classifier = LogisticRegression()

    cross_val_random_forest = GridSearchCV(estimator=random_forest_classifier,
                                           param_grid=param_grid, cv=5)

    # fit classifiers on training data
    cross_val_random_forest.fit(X_train, y_train)
    logistic_reg_classifier.fit(X_train, y_train)

    # store each classification report
    y_train_preds_rf = cross_val_random_forest.best_estimator_.predict(X_train)
    y_test_preds_rf = cross_val_random_forest.best_estimator_.predict(X_test)

    # store each classifier test predictions to create reports
    y_train_preds_lr = logistic_reg_classifier.predict(X_train)
    y_test_preds_lr = logistic_reg_classifier.predict(X_test)

    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)

    # plotting roc curves
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(
        cross_val_random_forest.best_estimator_,
        X_test,
        y_test,
        ax=ax,
        alpha=0.8)

    if not os.path.exists(os.path.join(os.getcwd(), 'images', 'results')):
        os.makedirs(os.path.join('images', 'results'))

    plt.savefig(os.path.join('images', 'results', 'random_forest_roc.png'))

    lr_plot = plot_roc_curve(logistic_reg_classifier, X_test, y_test)
    lr_plot.plot(ax=ax, alpha=0.8)
    plt.savefig(os.path.join('images', 'results', 'logistic_reg_roc.png'))

    # create "models" directory if it does not exist
    if not os.path.exists(os.path.join(os.getcwd(), 'models')):
        os.makedirs(os.path.join('models'))

    # save the models
    joblib.dump(
        cross_val_random_forest.best_estimator_,
        os.path.join(
            'models',
            'random_forest_model.pkl'))
    joblib.dump(
        logistic_reg_classifier,
        os.path.join(
            'models',
            'logistic_model.pkl'))

    # plot feature importance plots for random forest classifier
    feature_importance_plot(
        cross_val_random_forest.best_estimator_,
        X_train,
        os.path.join(
            'images',
            'results',
            'feat_importances.png'))


if __name__ == "__main__":
    data = import_data(os.path.join("data", "bank_data.csv"))
    data['Churn'] = data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    perform_eda(data)

    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    data = encoder_helper(data, cat_columns)

    X_train_data, X_test_data, y_train_data, y_test_data = perform_feature_engineering(
        data)

    train_models(X_train_data, X_test_data, y_train_data, y_test_data)
