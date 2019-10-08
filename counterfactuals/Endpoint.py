#!/usr/bin/env python
# coding: utf-8

import pickle
import numpy as np
import requests

def from_json(observation_to_interpret,sample_space,ignore):
    obs = [value for key,value in observation_to_interpret['level1PersonFeatures'].items()]
    put_back = []
    for i in range(len(ignore)):
        j = ignore[i]-i
        put_back.append(obs[j])
        del obs[j]
        del sample_space[j]
    # Map character value TODO: This doesnt always happen so make it conditional
    if 'B' in obs:
        sample_space[obs.index('B')][0] = 1
        sample_space[obs.index('B')][1] = 0
        char_index = obs.index('B')
        obs[obs.index('B')] = 1
    elif 'W' in obs:
        sample_space[obs.index('W')][0] = 1
        sample_space[obs.index('W')][1] = 0
        char_index = obs.index('W')
        obs[obs.index('W')] = 0
    return sample_space,obs,char_index,put_back

def to_json(char_index,random_point,ignore,observation_to_interpret,put_back):
    # Map character value TODO: This doesnt always happen so make it conditional
    if random_point[char_index]==1:
        random_point[char_index] = 'B'
    elif random_point[char_index]==0:
        random_point[char_index] = 'W'
    # make sure nothing is numpy int 64
    for i in range(len(random_point)):
        if type(random_point[i]) == np.int64:
            random_point[i] = int(random_point[i])
        elif type(random_point[i]) == np.float64:
            random_point[i] = float(random_point[i])
    # Add in ignored features
    counter = 0
    for i in ignore:
        random_point.insert(i,put_back[counter])
        counter+=1
    # Put in json format
    random_point_json = {'level1PersonFeatures':{}}
    counter = 0
    for key, value in observation_to_interpret['level1PersonFeatures'].items():
        random_point_json['level1PersonFeatures'][key] = random_point[counter]
        counter+=1
    return random_point_json

def check_if_in_SL(z,low,high,obs):
    diff = np.subtract(obs, z)
    norm_val = np.linalg.norm(diff)
    if low <= norm_val and norm_val <= high:
        return True
    else:
        return False

def call_generate(eta,low,high,obs,sample_space,char_index,ignore,observation_to_interpret, put_back):
    return generate(eta,low,high,obs,sample_space,char_index,ignore,observation_to_interpret, put_back)

def generate(eta,low,high,obs,sample_space,char_index,ignore,observation_to_interpret,put_back,distance_flag=False):
    random_vector = np.random.uniform(-1,1,len(obs))
    a = eta/np.sqrt(np.sum(np.square(random_vector)))
    b = [a*i for i in random_vector]
    random_point = [b[i]+obs[i] for i in range(len(b))]
    # Make sure int features are not floats
    for i in range(len(obs)):
        if isinstance(sample_space[i][1],int) or isinstance(sample_space[i][0],int):
            random_point[i] = int(random_point[i])
        # Normalize
        if random_point[i]>sample_space[i][1]:
            random_point[i] = sample_space[i][1]
        if random_point[i]<sample_space[i][0]:
            random_point[i] = sample_space[i][0]
    # Make sure the point lies in the spherical layer else calculate again
    if check_if_in_SL(random_point,low,high,obs):
        if distance_flag:
            d = [np.square(obs[i]-random_point[i]) for i in range(len(random_point))]
            print(np.sqrt(np.sum(d)))
        #convert back to json
        random_point_json = to_json(char_index,random_point,ignore,observation_to_interpret, put_back)
        return random_point_json
    else:
        return call_generate(eta,low,high,obs,sample_space,char_index,ignore,observation_to_interpret, put_back)
def make_z(eta,low,high,observations_n,obs,sample_space,char_index, ignore, observation_to_interpret, put_back):
    return [generate(eta,low,high,obs,sample_space,char_index,ignore, observation_to_interpret, put_back) for i in range(observations_n)]

def start_looking(z,eta,main_result,binary_classifier,threshold):
    for i in z:
        prediction = requests.post(binary_classifier, json=i).json()
        result = 1 if prediction['alertConfidence']>=threshold else 0
        if result!=main_result:
            return eta/2
    return

def find_eta(radius_eta,observations_n,obs,sample_space,main_result,binary_classifier,char_index, ignore, observation_to_interpret, put_back,threshold):
    eta = radius_eta
    z = make_z(eta,0,eta,observations_n,obs,sample_space,char_index, ignore, observation_to_interpret, put_back)
    while start_looking(z,eta,main_result,binary_classifier,threshold)!=None:
        eta = start_looking(z,eta,main_result,binary_classifier,threshold)
        z = make_z(eta,0,eta,observations_n,obs,sample_space,char_index, ignore, observation_to_interpret, put_back)
    return eta,z

def find_enemy(a0,a1,z,binary_classifier,observations_n,obs,sample_space,threshold,main_result,char_index, ignore, observation_to_interpret, put_back):
    eta = a0
    while 1:
        if not int(not(main_result)) in [1 if requests.post(binary_classifier, json=i).json()['alertConfidence']>=threshold else 0 for i in z]:
        # if not 1 in [binary_classifier.predict([i])[0] for i in z]:
            z = make_z(a1,a0,a1,observations_n,obs,sample_space,char_index, ignore, observation_to_interpret, put_back)
            a0=a1
            a1=a1+eta
        else:
            break
    # return z[[binary_classifier.predict([i])[0] for i in z].index(1)]
    return z[[1 if requests.post(binary_classifier, json=i).json()['alertConfidence']>=threshold else 0 for i in z].index(int(not(main_result)))]

def find_enemy_star(enemy,binary_classifier,main_result,observation_to_interpret,threshold):
    get_indices = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]

    enemy_prime = enemy

    # while binary_classifier.predict([enemy_prime])!= main_result:
    while (1 if requests.post(binary_classifier, json=enemy_prime).json()['alertConfidence']>=threshold else 0)!= main_result:
        enemy_star = enemy_prime
        to_minimize = []
        keys_list = []
        for key, value in enemy_prime['level1PersonFeatures'].items():
            if value!= observation_to_interpret['level1PersonFeatures'][key]:
                to_minimize.append(np.absolute(value-observation_to_interpret['level1PersonFeatures'][key]))
                keys_list.append(key)
        to_change = get_indices(min(to_minimize),to_minimize)
        to_change = [keys_list[i] for i in to_change]
        for i in to_change:
            enemy_prime['level1PersonFeatures'][i] = observation_to_interpret['level1PersonFeatures'][i]
    return enemy_star

def find_counterfactual(radius_eta,observations_n,binary_classifier,observation_to_interpret,sample_space,threshold,ignore):
    result = requests.post(binary_classifier, json=observation_to_interpret).json()
    main_result = 1 if result['alertConfidence']>=threshold else 0
    # Delete values we need to ignore
    sample_space_new, obs, char_index, put_back = from_json(observation_to_interpret,sample_space,ignore)
    ### Steps 1-5 Algorithm 1
    eta,z = find_eta(radius_eta,observations_n,obs,sample_space,main_result,binary_classifier, char_index, ignore, observation_to_interpret, put_back,threshold)
    ### Steps 6-12 Algorithm 1
    enemy = find_enemy(eta,2*eta,z,binary_classifier,observations_n,obs,sample_space,threshold,main_result,char_index, ignore, observation_to_interpret, put_back)
    ### Steps 1-7 Algorithm 2
    enemy_star = find_enemy_star(enemy,binary_classifier,main_result,observation_to_interpret,threshold)
    return enemy_star

if __name__ == '__main__':
    # <tt>observation_to_interpret</tt> is nothing but the point whose counterfactual we are trying to find. <tt>sample_space</tt> is the min and max limits of 
    # the features as described in the dataset.
    # We need to find the closest 'enemy' e i.e. an observation whose result is the opposite of the observation to interpret and is closest to it vector wise
    filename = 'binary_classifier.sav'
    binary_classifier = pickle.load(open(filename, 'rb'))
    observation_to_interpret = [ 1.  , 90.  , 62.  , 12.  , 43.  , 27.2 ,  0.58, 24.  ] #this is just the first row of the test set. its result is 0
    sample_space = [[0,17] , [0,199] , [0,122] , [0,99] , [0,846] , [0,67.1] , [0.08,2.42] , [21,81]]
    radius_eta = 5 #hyperparameters
    observations_n = 10 #hyperparameters

    enemy_star = find_counterfactual(radius_eta,observations_n,binary_classifier,observation_to_interpret,sample_space)

    print('Enemy:\t\t'+str(enemy_star))
    print('Original:\t'+str(observation_to_interpret))


# json_to_predict = observation_to_interpret
# observation_to_interpret = [value for key, value in json_to_predict['level1PersonFeatures'].items()]
# print(observation_to_interpret)

