from sklearn.model_selection import StratifiedKFold
import numpy as np

# feat_vecs is list of feature vectors from our labeled data
# labels is list of outputs for those feature vectors
# n_folds is number of times we want to perform cross validation
# returns a tuple of (xtrain, ytrain, xtest, ytest)
# xtrain, xtest are numpy arrays of lists, where each list is a feature vector
# ytrain, ytest are numpy arrays of classes
def cross_validate(feat_vecs, labels, n_folds):
	skf = StratifiedKFold(n_splits=n_folds, random_state=None, shuffle=False)
	min_loss, max_loss = float('inf'), -float('inf')
	min_accuracy, max_accuracy = float('inf'), -float('inf')
	loss_sum = 0
	accuracy_sum = 0
	# iterate through n_folds number of train/test partitions
	for train_indices, test_indices in skf.split(feat_vecs, labels):
        sz = len(train_indices)
        xtrain, ytrain, xtest, ytest = np.zeros(sz, dtype=object), np.zeros(sz, dtype=object), np.zeros(sz, dtype=object), np.zeros(sz, dtype=object)
        for i, train_index in enumerate(train_indices):
            xtrain[i] = feat_vecs[train_index]
            ytrain[i] = labels[train_index]
        for i, test_index in enumerate(test_indices):
            xtest[i] = feat_vecs[test_index]
            ytest[i] = labels[test_index]
        # train on xtrain, ytrain (copied from neuralnet.py)
        history = model.fit(xtrain, ytrain, epochs=200, batch_size=batch_sz)
        # test on xtest, ytest (copied from neuralnet.py)
        score = model.evaluate(xtest, ytest, batch_size=batch_sz)
        # update metrics
        loss_sum += score[0]
        accuracy_sum += score[1]
        if score[0] < min_loss: min_loss = score[0]
    	elif score[0] > max_loss: max_loss = score[0]
        if score[1] < min_accuracy: min_accuracy = score[1]
    	elif score[1] > max_accuracy: max_accuracy = score[1]

    avg_loss = float(loss_sum)/n_folds
    avg_accuracy = float(accuracy_sum)/n_folds
    final_tuple = (avg_loss, min_loss, max_loss, avg_accuracy, min_accuracy, max_accuracy)
    print ("average loss: %f\nmin: %f\tmax: %f\n\naverage accuracy: %f\nmin: %f\tmax: %f\t\n" % final_tuple)
    return final_tuple

