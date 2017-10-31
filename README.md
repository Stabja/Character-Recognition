# Character-Recognition
Character Recognition from A-J in Python

### Import the Libraries

	from __future__ import print_function
	import matplotlib.pyplot as plt
	import numpy as np
	import os
	import sys
	import tarfile
	from IPython.display import display, Image
	from scipy import ndimage
	from sklearn.linear_model import LogisticRegression
	from six.moves.urllib.request import urlretrieve
	from six.moves import cPickle as pickle

### Set the path to 'train_folders' and 'test_folders' as

	train_folders = ['notMNIST_large\\A', 'notMNIST_large\\B', 'notMNIST_large\\C', 'notMNIST_large\\D', 'notMNIST_large\\E', 	'notMNIST_large\\F', 'notMNIST_large\\G', 'notMNIST_large\\H', 'notMNIST_large\\I', 'notMNIST_large\\J']
	test_folders = ['notMNIST_small\\A', 'notMNIST_small\\B', 'notMNIST_small\\C', 'notMNIST_small\\D', 'notMNIST_small\\E', 'notMNIST_small\\F', 'notMNIST_small\\G', 'notMNIST_small\\H', 'notMNIST_small\\I', 'notMNIST_small\\J']

### Set the number of files

	train_datasets = maybe_pickle(train_folders, 1800)
	test_datasets = maybe_pickle(test_folders, 48)

### Generate the dataset by calling 'merge_datasets()'

### Generate the pickle file for the entire dataset by executing the code below it

	statinfo = os.stat(pickle_file)

### Convert 'train_dataset' and 'test_dataset' from 3D to 2D so as to fit in the LogisticRegressor

	nsamples, x_train, y_train = train_dataset.shape
	d2_train_dataset = train_dataset.reshape((nsamples,x_train*y_train))
	label_samples, x_test, y_test = test_dataset.shape
	d2_test_dataset = test_dataset.reshape((label_samples,x_test*y_test))

### Train the Classifier with scikitlearn's LogisticRegressor

	classifier = LogisticRegression(random_state = 0)
	classifier.fit(d2_train_dataset, train_labels)

### Predict the characters for test dataset

	pred_labels = classifier.predict(d2_test_dataset)
	
### Plot the graph of 'pred_labels' vs 'test_labels'

	k = list(range(100))
	plt.plot(k, pred_labels)
	plt.scatter(k, test_labels, color='green')
	plt.show()
