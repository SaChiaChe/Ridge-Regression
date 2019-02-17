import sys
import numpy as np
from func.ReadData import *
from func.Calculations import *
from func.Validation import *

if len(sys.argv) != 5:
	print("Format: python RidgeRegressionWithValidation.py <TrainFile> <TestFile> <Lambda> <ValidationSize>")
	exit(0)

# Read Data
TrainFile, TestFile = sys.argv[1], sys.argv[2]
Train_X, Train_Y = ReadData(TrainFile)
Test_X, Test_Y = ReadData(TestFile)

# Cut Validation set
ValidSize = int((1 - float(sys.argv[4])) * len(Train_X))
[Train_X, Train_Y], [Valid_X, Valid_Y] = CutValidation(Train_X, Train_Y, ValidSize)

# Run Ridge Regression on Train dataset
Lambda = float(sys.argv[3])
W_REG = CalPseudoinverse(Train_X, Lambda) @ Train_X.T @ Train_Y

# Calculate error
E_train = CalError(W_REG, Train_X, Train_Y)
E_valid = CalError(W_REG, Valid_X, Valid_Y)
E_out = CalError(W_REG, Test_X, Test_Y)

print("E_train:", E_train)
print("E_valid:", E_valid)
print("E_out", E_out)