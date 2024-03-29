# Predict Customer Churn

This is a classification task to predict customer attrition. The data is taken from Kaggle [here](https://www.kaggle.com/sakshigoyal7/credit-card-customers).

This is an imbalanced dataset. It was fun interpreting the accuracy, precision, and recall numbers.

## Running the code

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the following in a new environment.

```bash
conda create --name churn
pip install pandas numpy sklearn matplotlib pylint autopep8 pytest seaborn
```

## Usage

```bash
python3 churn_script_logging_and_tests.py

python3 -m pytest churn_script_logging_and_tests.py

```

![pytest_result](images/pytest.png)

## Results

### Exploratory Data Analysis

Age of customers
![EDA_age](images/eda/age_fig.png)

Churn ratio
![EDA_churn](images/eda/churn_fig.png)

Correlation of features
![EDA_heatmap](images/eda/heatmap_fig.png)

Marital Status ratio
![EDA_marital_status](images/eda/marital_status_fig.png)

Total transactions per customer
![EDA_total_transations](images/eda/total_trans_ct_fig.png)

### Results

Random Forest Results
![RF_Results](images/results/rf_results.png)

Logistic Regression Results
![RF_Results](images/results/lr_results.png)

It is interesting to note that even if on average, logistic regression performs well, it performs particularly bad for class 1 i.e. for attritioned customers. We can see that the recall is particularly bad for class 1. This means that there is a very high number of False Negatives (As we can also see that precision is much better for class 1). The algorithm is misclassifying a lot of this class as the other class.

To read more about interpreting precision, recall, read [this](https://medium.com/data-science-in-your-pocket/calculating-precision-recall-for-multi-class-classification-9055931ee229). 

Feature importance for Random Forest Classifier
![RF_Features](images/results/feat_importances.png)


## Learnings
I learned how to properly use fixtures.

Some advice [here](https://www.analyticsvidhya.com/blog/2022/01/writing-test-cases-for-machine-learning/), [here](https://www.seanh.cc/2017/02/12/advanced-fixtures/#:~:text=A%20fixture%20can%20use%20multiple,fixture%20value%20that%20it%20returns.), [here](https://madewithml.com/courses/mlops/testing/), and [here](https://towardsdatascience.com/make-your-python-tests-efficient-with-pytest-fixtures-3d7a1892265f)

More [here](https://www.youtube.com/watch?v=ErS0PPfLFLI&ab_channel=PyCharmbyJetBrains), and [here](https://docs.pytest.org/en/6.2.x/parametrize.html).

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)