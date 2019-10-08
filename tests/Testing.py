import Endpoint as e
import json
import pandas as pd
import numpy as np

binary_classifier = 'http://dev-int.ec2.rdc.com/level1/randomForestReview'
'''
* Could read in from a dataset? - both
* Turn char features into int - both
* Find sample space - both
* send URL 
* include threshold if model o/p is a value but is later made binary - both
'''
with open('p_batch.json') as p:
	json_data = json.load(p)

data = {'addressMatchScore':[],'surnameMatchScore':[],'matchSurnameCulture':[],'eventMapOne':[],'gridScore':[],'firmNbr':[],'givenNameMatchScore':[],'gmAutoAlert':[],'globalSearch':[],'matchScore':[],'batchTypeCd':[],'addressLineOneExactMatch':[],'sourceScore':[],'compositeMatchScore':[],'cvipValue':[],'alertScore':[],'eventScore':[],'provinceExactMatch':[],'nameMatchScore':[],'countryExactMatch':[],'dateOfBirthMatchScore':[],'matchGivenNameCulture':[],'cityExactMatch':[],'commonSurname':[],'marsStatus':[],'unitedStatesAddress':[]}
for i in json_data['level1ReviewRequests']:
	for key,value in data.items():
		if key in i['level1PersonFeatures']:
			data[key].append(i['level1PersonFeatures'][key])

d = pd.DataFrame.from_dict(data)

# Make categorical batchtypecd to int
# d['batchTypeCd'] = np.where(d['batchTypeCd']=='B', 1, 0)

obs = list(d.iloc[0])
o = {}
ignore = []
for n in range(len(list(d))):
	if type(obs[n]) == np.int64:
		o[list(d)[n]] = int(obs[n])
	else:
		o[list(d)[n]] = obs[n]
		ignore.append(n)
ignore.append(3)
ignore.remove(10)
# print(ignore)
# print([list(d)[i] for i in ignore])
observation_to_interpret = {'level1PersonFeatures':o}

sample_space = []
for i in list(d):
	l = []
	l.append(d[i].min())
	l.append(d[i].max())
	sample_space.append(l)

radius_eta = 20 #hyperparameters
observations_n = 10 #hyperparameters
threshold = 0.5

enemy_star = e.find_counterfactual(radius_eta,observations_n,binary_classifier,observation_to_interpret,sample_space,threshold,sorted(ignore))

print('Enemy:\t\t'+str(enemy_star))
print('Original:\t'+str(observation_to_interpret))

# curl --header "Content-Type: application/json" --request POST --data @p_single.json http://dev-int.ec2.rdc.com/level1/randomForestReview
# a = {"level1PersonFeatures": {"addressMatchScore": 0.0, "surnameMatchScore": 100.0, "matchSurnameCulture": 1.0, "eventMapOne": "9007199254740992", "gridScore": 90.0, "firmNbr": "BW10001P", "givenNameMatchScore": 94.0, "gmAutoAlert": 0.0, "globalSearch": 0.0, "matchScore": 92.0, "batchTypeCd": "B", "addressLineOneExactMatch": 1.0, "sourceScore": 100.0, "compositeMatchScore": 87.0, "cvipValue": 6.0, "alertScore": 49.0, "eventScore": 59.0, "provinceExactMatch": 1.0, "nameMatchScore": 97.0, "countryExactMatch": 1.0, "dateOfBirthMatchScore": 0.0, "matchGivenNameCulture": 1.0, "cityExactMatch": 1.0, "commonSurname": 1.0, "marsStatus": "NMH", "unitedStatesAddress": 0.0}, "correlationId": "56299745.211660137"}
# print(requests.post(binary_classifier, json=obs).json())

