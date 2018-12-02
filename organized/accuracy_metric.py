# micro recall 


#X = np.array(new_ypred)
#X = dataset[:,0:4].astype(float)

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

# first, add threshold to calculate 0/1 values
def change_by_threshold(threshold, values_vector):
  new_values_vector = [] 
  for x in values_vector:
    actual = [] 
    for y in x: 
      y = 1 if y > threshold else 0 
      actual.append(y)
    new_values_vector.append(actual)
  return new_values_vector


# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(23):
    precision[i], recall[i], _ = precision_recall_curve(ytest[:, i],
                                                        ypred[:, i])
    average_precision[i] = average_precision_score(ytest[:, i], ypred[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(ytest.ravel(),
    ypred.ravel())
average_precision["micro"] = average_precision_score(ytest, ypred,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))


## TO DO: get average micro recall
