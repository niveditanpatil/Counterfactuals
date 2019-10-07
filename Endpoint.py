#!/usr/bin/env python
# coding: utf-8

import pickle
import numpy as np

def check_if_in_SL(z,low,high,observation_to_interpret):
    diff = np.subtract(observation_to_interpret, z)
    norm_val = np.linalg.norm(diff)
    if low <= norm_val and norm_val <= high:
        return True
    else:
        return False

def call_generate(eta,low,high,observation_to_interpret,sample_space):
    return generate(eta,low,high,observation_to_interpret,sample_space)
def generate(eta,low,high,observation_to_interpret,sample_space,distance_flag=False):
    random_vector = np.random.uniform(0,1,len(observation_to_interpret))
    a = eta/np.sqrt(np.sum(np.square(random_vector)))
    b = [a*i for i in random_vector]
    random_point = [b[i]+observation_to_interpret[i] for i in range(len(b))]
    # Make sure int features are not floats
    for i in range(len(observation_to_interpret)):
        if isinstance(sample_space[i][1],int):
            random_point[i] = int(random_point[i])
        # Normalize
        if random_point[i]>sample_space[i][1]:
            random_point[i] = sample_space[i][1]
        if random_point[i]<sample_space[i][0]:
            random_point[i] = sample_space[i][0]
    # Make sure the point lies in the spherical layer else calculate again
    if check_if_in_SL(random_point,low,high,observation_to_interpret):
        if distance_flag:
            d = [np.square(observation_to_interpret[i]-random_point[i]) for i in range(len(random_point))]
            print(np.sqrt(np.sum(d)))
        return random_point
    else:
        return call_generate(eta,low,high,observation_to_interpret,sample_space)
def make_z(eta,low,high,observations_n,observation_to_interpret,sample_space):
    return [generate(eta,low,high,observation_to_interpret,sample_space) for i in range(observations_n)]
def start_looking(z,eta,main_result,binary_classifier):
    for i in z:
        if binary_classifier.predict([i])!=main_result:
            return eta/2
    return

def find_eta(radius_eta,observations_n,observation_to_interpret,sample_space,main_result,binary_classifier):
    eta = radius_eta
    z = make_z(eta,0,eta,observations_n,observation_to_interpret,sample_space)

    while start_looking(z,eta,main_result,binary_classifier)!=None:
        eta = start_looking(z,eta,main_result,binary_classifier)
        z = make_z(eta,0,eta,observations_n,observation_to_interpret,sample_space)
    return eta,z

def find_enemy(a0,a1,z,binary_classifier,observations_n,observation_to_interpret,sample_space):
    eta = a0
    while 1:
        if not 1 in [binary_classifier.predict([i])[0] for i in z]:
            z = make_z(a1,a0,a1,observations_n,observation_to_interpret,sample_space)
            a0=a1
            a1=a1+eta
        else:
            break
    return z[[binary_classifier.predict([i])[0] for i in z].index(1)]

def find_enemy_star(enemy,binary_classifier,main_result,observation_to_interpret):
    get_indices = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]

    enemy_prime = enemy

    while binary_classifier.predict([enemy_prime])!= main_result:
        enemy_star = enemy_prime
        to_minimize = []
        for j in range(len(enemy_prime)):
            if enemy_prime[j]!= observation_to_interpret[j]:
                to_minimize.append(np.absolute(enemy_prime[j]-observation_to_interpret[j]))
        to_change = get_indices(min(to_minimize),to_minimize)
        for i in to_change:
            enemy_prime[i] = observation_to_interpret[i]
    return enemy_star

def find_counterfactual(radius_eta,observations_n,binary_classifier,observation_to_interpret,sample_space):
    main_result = binary_classifier.predict([observation_to_interpret])
    # ### Steps 1-5 Algorithm 1
    eta,z = find_eta(radius_eta,observations_n,observation_to_interpret,sample_space,main_result,binary_classifier)
    # ### Steps 6-12 Algorithm 1
    enemy = find_enemy(eta,2*eta,z,binary_classifier,observations_n,observation_to_interpret,sample_space)
    # ### Steps 1-7 Algorithm 2
    enemy_star = find_enemy_star(enemy,binary_classifier,main_result,observation_to_interpret)
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