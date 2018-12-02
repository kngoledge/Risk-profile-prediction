
"""
	proj_names is list of project names from COMPLAINTS
	wb_data is list of tuples where each element is tuple of (project name, [countries], [sectors])
	updates wb_data to remove instances of matching project names
"""
def remove_duplicate_projects(proj_names, wb_data):
	unmatched_data = []
	for name in proj_names:
		match = False
		for i, tup in enumerate(wb_data):
			if tup[0] == name:
				match = True
				break
		if match == False: unmatched_data.append(tup)
	return unmatched_data


"""
	Returns a shuffled combo of complaint_data and WB_data
	Complaint Data: list of ([countries], [sectors], [issues]) tuples
	WB Data: 		list of (proj name, [countries], [sectors]) tuples
"""
def combine_datasets(complaint_data, WB_data, numIssues):
	
	for i in range(len(WB_data)):
		noIssues = np.zeros(numIssues)
		noIssues[-1] = 1				# mark NONE as true
		complaint_data.append( (WB_data[i][1],WB_data[2],noIssues) )

	complaint_array = arr(complaint_data)
	np.random.shuffle(arr(complaint_array))

	return complaint_array

"""
	Replace section of code in util.py with this, to get new training/test data.
"""
def continuation(lol_temp)
	complaint_df = prepare_raw_data(filename)
	complaint_clean_df = prepare_clean_data(df)		#list of tuples
	unique_wb_data = remove_duplicate_projects(complaint_clean_df, wb_clean_df)
	total_dataset = combine_datasets(complaint_clean_df, unique_wb_data)


