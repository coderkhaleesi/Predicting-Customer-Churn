import os
import logging
import churn_library as cl

import pytest

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


@pytest.fixture
def import_data():
    data = cl.import_data(os.path.join("data", "bank_data.csv"))
    return data

@pytest.fixture
def perform_eda(import_data):
    data = import_data
    data['Churn'] = data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    cl.perform_eda(data)
    return data

@pytest.fixture
def encoder_helper(perform_eda):
    data = perform_eda

    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']

    data = cl.encoder_helper(data, cat_columns)
    return data

@pytest.fixture
def perform_feature_engineering(encoder_helper):
    data = encoder_helper

    return cl.perform_feature_engineering(data)

@pytest.fixture
def train_models(perform_feature_engineering):
    X_train, X_test, y_train, y_test = perform_feature_engineering
    cl.train_models(X_train, X_test, y_train, y_test)



def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df_data = import_data
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df_data.shape[0] > 0
        assert df_data.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err

    return df_data


def test_eda(perform_eda):
    '''
    test perform eda function
    '''

    path = os.path.join("images", "eda")
    # Checking if the list is empty or not
    try:
        # Getting the list of directories
        dirs = os.listdir(path)
        assert len(dirs) > 0
        logging.info("Testing perform_eda: SUCCESS")

    except AssertionError as err:
        logging.error("Testing eda results exist: no eda results found")
        raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    
    df_data = encoder_helper

    cat_columns = [
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    try:
        for col in cat_columns:
            assert col in df_data.columns
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error("Testing cols: Cannot find columns")
        return err

    return df_data


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''

    X_train, X_test, y_train, y_test = perform_feature_engineering

    try:
        assert X_train.shape[1] == 19
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing cols: cannot find all features")
        return err


def test_train_models(train_models):
    '''
    test train_models
    '''
    path = "./images/results/"
    try:
        # Getting the list of directories
        dir_val = os.listdir(path)
        assert len(dir_val) > 0
    except FileNotFoundError as err:
        logging.error("Testing train_models: Results image files not found")
        raise err

    path = "./models/"
    try:
        # Getting the list of directories
        dir_val = os.listdir(path)
        assert len(dir_val) > 0
        logging.info("Testing train_models: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing train_models: Model files not found")
        raise err


if __name__ == "__main__":
    
    test_eda(cl.perform_eda, data)

    data = test_encoder_helper(cl.encoder_helper, data)

    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = test_perform_feature_engineering(
        cl.perform_feature_engineering, data)

    test_train_models(cl.train_models, X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
