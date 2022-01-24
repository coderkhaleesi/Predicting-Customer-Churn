# Predict Customer Churn

This is a classification task to predict customer attrition. The data is taken from Kaggle [here] (https://www.kaggle.com/sakshigoyal7/credit-card-customers).

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

## Results

![EDA_age](images/eda/age_fig.png)
![EDA_churn](images/eda/churn_fig.png)
![EDA_heatmap](images/eda/heatmap_fig.png)

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)