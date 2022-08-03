from gym_flp.envs.flp_env_2D_Umgebung_Mehrziel import qapEnv
from gym_flp.envs.Hilfsfunktionen import Flusskennzahlen
import tensorflow as tf
import tensorflow
#from tensorflow.keras.datasets import cifar10
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from collections import deque
import numpy as np
import time
import random
import os
from tqdm import tqdm


env = qapEnv(instance = 'Neos-n6', mode = 'rgb_array')
LOAD_MODEL = "models/TEST-Differenz-vorher-nachher-128F-64Mini-0.001lr-lin-20k__1220.00max__661.80avg__218.00min__1621650949.model"
model = load_model(LOAD_MODEL)
EPISODES = 1
end_costs=[]
average_end_costs=[]

Probeläufe = 100
MHCmin, MHCmax, Rücklaufmin, Rücklaufmax, Diagonalmin, Diagonalmax, Lärmmin, Lärmmax, Lärmpunktzahlmin, Lärmpunktzahlmax = env.Probelauf(Probeläufe)
Gesamtreward_Liste=[]

for episode in range(1, EPISODES + 1):
    #print('')
    done=False
    episode_reward=0
    s0 = env.reset()
    #print(s0)
    print(env.internal_state)
    current_state = s0
    print('Anfangs-Lärmpunktzahl: ' + str(env.initial_Lärmpunktzahl))
    print('Anfangs-MHC: ' + str(env.initial_MHC))
    print('Anfangs-Rückläufe: ' + str(env.initial_Rückläufe))
    #print(env.initial_Gesamtdiagonalabweichung)
    print('')
    while not done:
        env.render()
        #Predictions = model.predict(np.array(current_state).reshape(-1, *current_state.shape)/255)[0]
        Predictions_state = np.expand_dims(current_state, axis=0)/255
        Predictions = model.predict(Predictions_state)
        best_predict = np.argmax(Predictions)
        action=best_predict
        #action = np.random.randint(0, len(env.actions))
        #print(action)
        new_state, reward, done, a = env.step(action)
        
        episode_reward += reward
        print('Lärmpunktzahl: ' + str(env.Lärmpunktzahl))
        print('MHC: ' + str(env.last_MHC))
        print('Rückläufe: ' + str(env.last_Rückläufe))
        #print('Diagonalabweichung: ' + str(env.last_Gesamtdiagonalabweichung))
        #print('Lärm: ' + str(env.last_Lärmdurchschnitt))
        #print('MHC Reward: ' + str(env.MHCreward))
        #print('Rücklaufreward: ' + str(env.Rücklaufreward))
        #print('Diagonalabweichung Reward: ' + str(env.Gesamtdiagonalabweichungreward))
        #print('Lärm Reward: ' + str(env.Lärmreward))
        #print('Lärm Reward Intervalle: ' + str(env.Lärmrewardintervalle))
        #print('Reward für Aktion: ' + str(reward))
        #print('Episoden Reward: ' + str(episode_reward))

        print('')
        current_state = new_state
    Gesamtreward_Liste.append(episode_reward)  
    if episode % 10 ==0:
        average_costs = sum(Gesamtreward_Liste[-10:])/len(Gesamtreward_Liste[-10:])
        average_end_costs.append(average_costs)

