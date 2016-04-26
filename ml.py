import graphlab as gl
import math

target_feature = "response" # sample target feature
data_source = 'balanced_response_prediction_task.csv' # sample data source file
def get_data(source, dropna = False):
	# Load the dataset
	_data = gl.SFrame(source)
	if(dropna):
		_columns = get_column_names(_data)
		return _data.dropna(_columns)
	else:
		return _data

def get_auc(dependent, independent, train, test):
	# Build a logistic regression model using the train data
	model = gl.logistic_classifier.create(train, target = dependent, features = independent, validation_set = None, feature_rescaling = True)
	return model.evaluate(test, 'auc')["auc"]

def get_coefficients(dependent, independent, train, test):
	model = gl.logistic_classifier.create(train, target = dependent, features = independent, validation_set = None, feature_rescaling = True)
	return model.get("coefficients")

def split_data(data, partitions):
	one, two = data.random_split(partitions[0])
	two, three = two.random_split(partitions[1] / (partitions[1] + partitions[2]))
	return one, two, three

def get_column_names(data):
	return data[0].keys()

def mutual_information(a, b):
	sf = gl.SFrame({'a' : a, 'b' : b})
	joined = sf.groupby(['a', 'b'], {'a_b_count' : gl.aggregate.COUNT()}).join(sf.groupby('a', {'a_count' : gl.aggregate.COUNT()})).join(sf.groupby('b', {'b_count' : gl.aggregate.COUNT()}))

	joined['p(a)'] = joined['a_count'] / a.size()
	joined['p(b)'] = joined['b_count'] / a.size()
	joined['p(a,b)'] = joined['a_b_count'] / a.size()
	pSum = 0
	for i, x in enumerate(joined):
		if x['p(a,b)'] == 0:
			pSum += 0
		else:
			pSum += x['p(a,b)'] * (math.log(x['p(a,b)']) - (math.log(x['p(a)']) + math.log(x['p(b)'])))
	return pSum

#### Sample data code ####
data = get_data(data_source, dropna = True)
train_data, test_data, validation_data = split_data(data, [0.7, 0.2, 0.1])
#### End sample data code ####

# strategy can be greedy, logit or mui
# only auc is currently supported as a metric
def fs_test(train_data, test_data, strategy, metric = "auc", params = False):
	if strategy == "greedy":
		#### Greedy sampling sample code ####
		# params should be [n] where n is the amount of runs to make over the data
		if params != False: # lol
			runs = params[0]
		else:
			runs = 30 # default value
		aucsgreedy = [{}] * 115
		columns = []

		# iterate over the data
		for iteration in range(0, runs):
			auc_index = iteration + 1
			auc_max = 0
			columns_max = []
			for column in get_column_names(data):
				if column == "response" or column in columns:
					continue # do nothing if we've already chosen the
							 # column or if it's the target feature
				else:
					columns_temp = columns[:]
					columns_temp.append(column)
					curr_auc = get_auc("response", columns_temp, train_data, test_data)
					if curr_auc > auc_max:
						auc_max = curr_auc
						columns_max = columns_temp
			columns = columns_temp
			curr_elem = {}
			curr_elem["auc"] = auc_max
			curr_elem["features"] = columns_temp[:]
			curr_elem["amt"] = auc_index
			aucsgreedy[auc_index] = curr_elem

		aucs = [elem for elem in aucsgreedy if elem != {}] # remove all the null elements

		all_models = aucs[:] # just in case
		#### End greedy sampling sample code ####
	elif strategy == "logit":
		#### Logistic coefficient filtering sample code ####
		columns = get_column_names(data)
		columns_wo_target = columns[:]
		columns_wo_target.remove(target_feature)
		coefficients = get_coefficients(target_feature, columns_wo_target, train_data, test_data)
		models = []
		removed = [target_feature] # columns that we want to remove; this includes the target feature (obviously)

		def smallest_coefficient(sf):
			_columns = {}
			for _c in sf:
				if _c["name"] not in _columns:
					_columns[_c["name"]] = 0
				else:
					_columns[_c["name"]] += _c["value"]
			smallest = ["", 1000000]
			for key, value in _columns.items():
				if smallest[1] > abs(value) and key != "(intercept)":
					smallest = [key, value]
			return smallest
		# some features will not have coefficients in the model, due to them being completely irrelevant to the target feature
		# as a result, we have to remove these features from our list
		for c in columns:
			flag = False
			for d in coefficients:
				if d["name"] == c:
					flag = True
			if flag != True:
				removed.append(c)

		# now, we go through, feature by feature, and remove the feature with the lowest coefficient
		for c in columns:
			cols = columns[:] # create temporary set of features
			# remove all features we've decided to remove
			for r in removed:
				if r in cols:
					cols.remove(r)
			if cols != []:
				model = gl.logistic_classifier.create(train_data, target = target_feature, features = cols, validation_set = None)
				curr_auc = model.evaluate(test_data, 'auc')["auc"] # the auc of the current model
				models.append({"auc": curr_auc, "columns": cols})
				coeffs = model.get("coefficients")
				toRemove = smallest_coefficient(coeffs) # find the feature with the smallest coefficient...
				removed.append(toRemove[0]) # ...and add it to the removal array
			else:
				continue # it's a bad fix but it's better than nothing

		# this code just puts everything into a nice array form
		# the output looks like [{"auc": the model's auc, "features": the features used in the model, "amt": the amount of features (for easy graphing)}, {...}, ...]
		auccoeff = [{}] * 115
		for model in models:
			m = dict(model)
			features = m["columns"]
			index = len(features)
			auc = m["auc"]
			curr_elem = {}
			curr_elem["auc"] = auc
			curr_elem["features"] = features[:]
			curr_elem["amt"] = index
			auccoeff[index] = curr_elem

		aucs = [elem for elem in auccoeff if elem != {}]

		all_models = aucs[:] # just in case
		#### End logistic coefficient filtering sample code ####
	elif strategy == "mui":
		#### Mutual information filtering sample code ####
		columns = get_column_names(data)
		mui = {} # a dictionary mapping each feature to its mutual information with the target feature
		for column in columns:
			if column == target_feature:
				continue
			else:
				targetData = data[target_feature]
				colData = data[column]
				m = mutual_information(targetData, colData)
				mui[column] = m

		# function to find the feature with the smallest mutual information
		def smallest_mui(a):
			smallest_key = ""
			smallest_value = 1000
			for key, value in a.items():
				if value < smallest_value:
					smallest_value = value
					smallest_key = key
			return smallest_key

		mui_features = []
		for key, value in mui.items():
			mui_features.append(key)

		# putting everything into a nice array, same format as that of the logistic sampling method
		aucsmui = [{}] * 115
		curr = dict(mui)
		for key, value in mui.items(): # go through the features, each time removing the feature with the smallest mutual information
			aucs_index = len(curr)
			smallest = smallest_mui(curr)
			curr_auc = get_auc(target_feature, mui_features, train_data, test_data) # get the auc of the current model
			curr_elem = {}
			curr_elem["auc"] = curr_auc
			curr_elem["features"] = mui_features[:]
			curr_elem["amt"] = len(mui_features)
			aucsmui[aucs_index] = curr_elem
			mui_features.remove(smallest)
			del curr[smallest]

		aucs = [elem for elem in aucsmui if elem != {}]

		all_models = aucsmui[:] # just in case
		#### End mutual information filtering sample code ####
	# only return the model with the largest auc
	# I'm leaving the code above in just in case someone wants to modify what to return
	auc_max = 0;
	model_max = {}
	for model in all_models:
		try:
			curr_auc = model["auc"]
			if curr_auc > auc_max:
				auc_max = curr_auc
				model_max = dict(model)
		except Exception:
			continue # make it go away

	return model_max
	
#### IGNORE THIS ####
maxima = []
for iteration in range(0, 10):
	train_data, test_data, validation_data = split_data(data, [0.7, 0.2, 0.1])
	max = fs_test(train_data, test_data, "logit")
	maxima.append({"auc": max["auc"], "amt": max["amt"]})