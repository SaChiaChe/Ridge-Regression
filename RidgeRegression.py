import sys
import numpy as np
from func.ReadData import *
from func.Calculations import *

if len(sys.argv) != 4:
	print("Format: python RidgeRegression.py <TrainFile> <TestFile> <Lambda>")
	exit(0)

# Read Data
TrainFile, TestFile = sys.argv[1], sys.argv[2]
Train_X, Train_Y = ReadData(TrainFile)
Test_X, Test_Y = ReadData(TestFile)

# Run Ridge Regression on Train dataset
Lambda = float(sys.argv[3])
W_REG = CalPseudoinverse(Train_X, Lambda) @ Train_X.T @ Train_Y

# Calculate error
E_in = CalError(W_REG, Train_X, Train_Y)
E_out = CalError(W_REG, Test_X, Test_Y)

print("E_in:", E_in)
print("E_out", E_out)