import sys
import numpy as np
from func.ReadData import *
from func.Calculations import *
from func.Validation import *

if len(sys.argv) != 5:
	print("Format: python RidgeRegression.py <TrainFile> <TestFile> <Lambda> <VFold>")
	exit(0)

# Read Data
TrainFile, TestFile = sys.argv[1], sys.argv[2]
Train_X, Train_Y = ReadData(TrainFile)
Test_X, Test_Y = ReadData(TestFile)

# Cut Validation set
VFold = int(sys.argv[4])
Fold_X, Fold_Y = Vfold(Train_X, Train_Y, VFold)

# Run Ridge Regression while leave one fold out
Lambda = float(sys.argv[3])
TrackError = []
for i in range(VFold):
	Train_X = CombineFolds(Fold_X, i)
	Train_Y = CombineFolds(Fold_Y, i)
	W_REG = CalPseudoinverse(Train_X, Lambda) @ Train_X.T @ Train_Y
	# Calculate error
	E_cv_i = CalError(W_REG, Fold_X[i], Fold_Y[i])
	TrackError.append(E_cv_i)

print("E_cv:", sum(TrackError) / VFold)