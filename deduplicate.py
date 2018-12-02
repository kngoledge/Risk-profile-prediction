# proj_names is list of project names from COMPLAINTS
# wb_data is list of tuples where each element is tuple of (project name, [countries], [sectors])
# updates wb_data to remove instances of matching project names
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