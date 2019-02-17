def CutValidation(X, Y, N):
	# Ideally, the validation set should be splitted randomly
	# But for this problem, we split it in a fixed way as the following:
	# the first N samples will be the training set and the rest will be the validation set
	Train_X, Valid_X = X[:N], X[N:]
	Train_Y, Valid_Y = Y[:N], Y[N:]
	return [Train_X, Train_Y], [Valid_X, Valid_Y]

def Vfold(X, Y, V):
	Size = int(len(X) / V)
	Offset = 0
	Fold_X, Fold_Y = [], []
	for i in range(V):
		if i != V-1:
			Fold_X.append(X[Offset:Offset+Size])
			Fold_Y.append(Y[Offset:Offset+Size])
		else:
			Fold_X.append(X[Offset:])
			Fold_Y.append(Y[Offset:])
		Offset += Size
	return Fold_X, Fold_Y

def CombineFolds(Folds, I):
	import numpy as np
	V = len(Folds)
	temp = np.array([])
	for i in range(V):
		if i != I:
			if not len(temp):
				temp = Folds[i]
			else:
				temp = np.concatenate((temp, Folds[i]))
	return temp