# Decision Stump
A practice of ridge regression, which is a regularized version of linear regression.

## How to run

Start the program
```
python RidgeRegression.py <TrainFile> <TestFile> <Lambda>
```
```
python RidgeRegressionWithValidation.py <TrainFile> <TestFile> <Lambda> <ValidationSize>
```
```
python RidgeRegressionWithVfold.py <TrainFile> <TestFile> <Lambda> <VFold>
```

### RidgeRegression.py

Run ridge regression on the train dataset and output the error for train dataset and error for test dataset.

### RidgeRegressionWithValidation.py

Same as RidgeRegression.py, but cut out a validation dataset from the train dataset first, and run ridge regression on the rest of the dataset, then output the error for train dataset, validation dataset, and test dataset.

### RidgeRegressionWithVfold.py

Run Vfold on the train dataset, which cuts the train dataset into V folds, use one fold as the validation dataset, and run ridge regression on the rest of train dataset, also record the error of the current validation dataset, repeat this for all folds. Output the average error of all validation datasets. This could give us a good view of how ridge regression actually works on the full dataset, while also being practical (compared to leave one out validation).

## Built With

* Python 3.6.0 :: Anaconda custom (64-bit)

## Authors

* **SaKaTetsu** - *Initial work* - [SaKaTetsu](https://github.com/SaKaTetsu)