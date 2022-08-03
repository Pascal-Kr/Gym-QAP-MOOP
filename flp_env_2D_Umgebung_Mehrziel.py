import numpy as np
import gym
import gym_flp
from gym import spaces
from numpy.random import default_rng
import pickle
import os
import math
import matplotlib.pyplot as plt
from PIL import Image
from gym_flp import rewards
from IPython.display import display, clear_output
import anytree
from anytree import Node, RenderTree, PreOrderIter, LevelOrderIter, LevelOrderGroupIter
from gym_flp.envs.Hilfsfunktionen import Flusskennzahlen
    

class qapEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array', 'human']}  

    def __init__(self, mode=None, instance=None):
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        self.DistanceMatrices, self.FlowMatrices = pickle.load(open(os.path.join(__location__,'discrete', 'qap_matrices.pkl'), 'rb'))
        self.transport_intensity = None
        self.instance = instance
        self.mode = mode
        
        
        while not (self.instance in self.DistanceMatrices.keys() or self.instance in self.FlowMatrices.keys() or self.instance in ['Neos-n6', 'Neos-n7', 'Brewery']):
            print('Available Problem Sets:', self.DistanceMatrices.keys())
            self.instance = input('Pick a problem:').strip()
     
        self.D = self.DistanceMatrices[self.instance]
        self.F = self.FlowMatrices[self.instance]
        self.n = len(self.D[0])
        self.x = math.ceil((math.sqrt(self.n)))
        
        
        Distanzen = gym_flp.envs.Hilfsfunktionen.Flusskennzahlen()
        Fluss = gym_flp.envs.Hilfsfunktionen.Flusskennzahlen()
        self.F = Fluss.Flussmatrix()
        self.Lärmdummy = gym_flp.envs.Hilfsfunktionen.Flusskennzahlen()
        self.Lärm1 = self.Lärmdummy.Lärmmatrix()
        self.Fabriklänge = 60  #24   #48
        self.Fabrikbreite = 60  #24   #48
        self.Messpunkte = 15
        Maschinenmittelpunkte = gym_flp.envs.Hilfsfunktionen.Flusskennzahlen()
        self.Mittelpunktex, self.Mittelpunktey = Maschinenmittelpunkte.Maschinenmittelpunkte(self.Fabriklänge, self.Fabrikbreite, self.x)
        self.D = Distanzen.Distanzmatrixneu(self.Mittelpunktex, self.Mittelpunktey)
        
        # Determine problem size relevant for much stuff in here:
        self.n = len(self.D[0])
        self.x = math.ceil((math.sqrt(self.n)))
        self.y = math.ceil((math.sqrt(self.n)))
        self.size = int(self.x*self.y)
        self.observation_space_values=(self.x,self.y,3)
        self.max_steps = self.n - 1

        self.action_space = spaces.Discrete(int((self.n**2-self.n)*0.5)+1)
                
        # If you are using images as input, the input values must be in [0, 255] as the observation is normalized (dividing by 255 to have values in [0, 1]) when using CNN policies.       
        if self.mode == "rgb_array":
            self.observation_space = spaces.Box(low = 0, high = 255, shape=(1, self.n, 3), dtype = np.uint8) # Image representation
        elif self.mode == 'human':
            self.observation_space = spaces.Box(low=1, high = self.n, shape=(self.n,), dtype=np.float32)
        
        self.states = {}    # Create an empty dictonary where states and their respective reward will be stored for future reference
        self.actions = self.pairwiseExchange(self.n)
        
        # Initialize Environment with empty state and action
        self.action = None
        self.state = None
        self.internal_state = None
        
        #Initialize moving target to incredibly high value. To be updated if reward obtained is smaller. 
        
        self.movingTargetRewardMHC = np.inf 
        self.movingTargetRewardRückläufe = np.inf 
        self.movingTargetRewardDiagonalabweichung = np.inf 
        self.movingTargetRewardLärm = np.inf
        self.Actual_MHCmin = np.inf
        self.Actual_Rücklaufmin = np.inf
        self.Actual_Diagonalmin = np.inf
        self.Actual_Lärmmin = np.inf
 
    
        self.MHC = gym_flp.envs.Hilfsfunktionen.Flusskennzahlen()    # Create an instance of class MHC in module mhc.py from package rewards
        self.Rückläufe = gym_flp.envs.Hilfsfunktionen.Flusskennzahlen()
        self.Gesamtdiagonalabweichung = gym_flp.envs.Hilfsfunktionen.Flusskennzahlen()
        self.Reward = gym_flp.envs.Hilfsfunktionen.Flusskennzahlen() 
    
    def reset(self):
        self.step_counter = 0  #Zählt die Anzahl an durchgeführten Aktionen
        self.state_1D = default_rng().choice(range(1,self.n+1), size=self.n, replace=False) 
        
        self.internal_state = self.state_1D.copy()
        self.fromState = self.internal_state.copy()
        newState = self.fromState.copy()
        MHC, self.TM = self.MHC.computeMHC(self.D, self.F, newState)
        Rückläufe, self.Flussmatrixneu = self.Rückläufe.computeRückläufe(self.F, newState)
        Gesamtdiagonalabweichung, self.Flussmatrixneu = self.Gesamtdiagonalabweichung.computeDiagonalabweichung(self.F, newState)
        self.Lärmpositionen = self.Lärmdummy.computeLärm(self.Lärm1, newState)
        LärmMesspunkte, Lärmdurchschnitt = self.Lärmdummy.Lärmberechnung(self.Fabriklänge, self.Fabrikbreite, self.Mittelpunktex, self.Mittelpunktey, self.Lärmpositionen, self.n, self.Messpunkte)

        self.Unter55, self.Unter60, self.Unter65, self.Unter70, self.Unter75, self.Unter80, self.Unter85, self.Über85 = self.Lärmdummy.Lärmintervalle(LärmMesspunkte)
        self.Lärmpunktzahl = self.Lärmdummy.LärmIntervallpunkte(self.Unter55, self.Unter60, self.Unter65, self.Unter70, self.Unter75, self.Unter80, self.Unter85, self.Über85)
        self.Lärmbereichswerte, self.Lärmmin, self.Lärmmax = self.Lärmdummy.Lärmbereiche(LärmMesspunkte, self.n, self.Messpunkte)
        self.Lärmbereichswerte2, self.Lärmmin2, self.Lärmmax2 = self.Lärmdummy.Lärmbereiche2(LärmMesspunkte, self.n)
        self.Unter55, self.Unter60, self.Unter65, self.Unter70, self.Unter75, self.Unter80, self.Unter85, self.Über85  = self.Lärmdummy.Lärmintervalle(self.Lärmbereichswerte2)
        self.Lärmpunktzahl = self.Lärmdummy.LärmIntervallpunkte(self.Unter55, self.Unter60, self.Unter65, self.Unter70, self.Unter75, self.Unter80, self.Unter85, self.Über85)
        self.xyz = newState  #Permutationsmatrix anzeigen (optional)
        
        state_2D = np.array(self.get_image())
        
        self.initial_MHC = MHC
        self.last_MHC = self.initial_MHC
        self.transformedMHC = ((MHC-self.MHCmin)/(self.MHCmax-self.MHCmin))
        self.last_transformedMHC = self.transformedMHC 
        
        self.initial_Rückläufe = Rückläufe
        self.last_Rückläufe = self.initial_Rückläufe
        self.transformedRückläufe = ((Rückläufe-self.Rücklaufmin)/(self.Rücklaufmax-self.Rücklaufmin))
        self.last_transformedRückläufe = self.transformedRückläufe
        
        
        self.initial_Gesamtdiagonalabweichung = Gesamtdiagonalabweichung
        self.last_Gesamtdiagonalabweichung = self.initial_Gesamtdiagonalabweichung
        self.transformedDiagonalabweichung = ((Gesamtdiagonalabweichung-self.Diagonalmin)/(self.Diagonalmax-self.Diagonalmin))
        self.last_transformedDiagonalabweichung = self.transformedDiagonalabweichung
        
        self.initial_LärmMesspunkte = LärmMesspunkte
        self.last_LärmMesspunkte = self.initial_LärmMesspunkte
        
        self.initial_Lärmdurchschnitt = Lärmdurchschnitt
        self.last_Lärmdurchschnitt = self.initial_Lärmdurchschnitt
        self.transformedLärmdurchschnitt = ((Lärmdurchschnitt-self.Lärmutopia)/(self.Lärmnadir-self.Lärmutopia))
        self.last_transformedLärmdurchschnitt = self.transformedLärmdurchschnitt  
        
        self.initial_Lärmpunktzahl = self.Lärmpunktzahl
        self.last_Lärmpunktzahl = self.initial_Lärmpunktzahl
        self.transformedLärmpunktzahl = ((self.Lärmpunktzahl-self.Lärmpunktzahlmin)/(self.Lärmpunktzahlmax-self.Lärmpunktzahlmin))
        self.last_transformedLärmpunktzahl = self.transformedLärmpunktzahl  
        return state_2D
    
    def step(self, action):
        # Create new State based on action 
        self.step_counter += 1 
        
        self.fromState = self.internal_state.copy()
        
        swap  = self.actions[action]
        self.fromState[swap[0]-1], self.fromState[swap[1]-1] = self.fromState[swap[1]-1], self.fromState[swap[0]-1]
        
        newState = self.fromState.copy()
        MHC, self.TM = self.MHC.computeMHC(self.D, self.F, newState)
        Rückläufe, self.Flussmatrixneu = self.Rückläufe.computeRückläufe(self.F, newState)
        Gesamtdiagonalabweichung, self.Flussmatrixneu = self.Gesamtdiagonalabweichung.computeDiagonalabweichung(self.F, newState)
        self.Lärmpositionen = self.Lärmdummy.computeLärm(self.Lärm1, newState)
        LärmMesspunkte, Lärmdurchschnitt = self.Lärmdummy.Lärmberechnung(self.Fabriklänge, self.Fabrikbreite, self.Mittelpunktex, self.Mittelpunktey, self.Lärmpositionen, self.n, self.Messpunkte)        
        self.Unter55, self.Unter60, self.Unter65, self.Unter70, self.Unter75, self.Unter80, self.Unter85, self.Über85  = self.Lärmdummy.Lärmintervalle(LärmMesspunkte)
        self.Lärmpunktzahl = self.Lärmdummy.LärmIntervallpunkte(self.Unter55, self.Unter60, self.Unter65, self.Unter70, self.Unter75, self.Unter80, self.Unter85, self.Über85)
        self.Lärmbereichswerte, self.Lärmmin, self.Lärmmax = self.Lärmdummy.Lärmbereiche(LärmMesspunkte, self.n, self.Messpunkte)
        self.Lärmbereichswerte2, self.Lärmmin2, self.Lärmmax2 = self.Lärmdummy.Lärmbereiche2(LärmMesspunkte, self.n)
        self.Unter55, self.Unter60, self.Unter65, self.Unter70, self.Unter75, self.Unter80, self.Unter85, self.Über85  = self.Lärmdummy.Lärmintervalle(self.Lärmbereichswerte2)
        self.Lärmpunktzahl = self.Lärmdummy.LärmIntervallpunkte(self.Unter55, self.Unter60, self.Unter65, self.Unter70, self.Unter75, self.Unter80, self.Unter85, self.Über85)
        
                
        self.last_MHC = MHC
        self.transformedMHC = ((MHC-self.MHCmin)/(self.MHCmax-self.MHCmin))

        self.MHCreward= (self.last_transformedMHC - self.transformedMHC)*100
        self.last_transformedMHC = self.transformedMHC    
        

        
        self.last_Rückläufe = Rückläufe
        self.transformedRückläufe = ((Rückläufe-self.Rücklaufmin)/(self.Rücklaufmax-self.Rücklaufmin))
        
        self.Rücklaufreward= (self.last_transformedRückläufe - self.transformedRückläufe)*100
        self.last_transformedRückläufe = self.transformedRückläufe

        
        self.last_Gesamtdiagonalabweichung = Gesamtdiagonalabweichung
        self.transformedDiagonalabweichung = ((Gesamtdiagonalabweichung-self.Diagonalmin)/(self.Diagonalmax-self.Diagonalmin))
    
        self.Gesamtdiagonalabweichungreward= (self.last_transformedDiagonalabweichung - self.transformedDiagonalabweichung)*100
        self.last_transformedDiagonalabweichung = self.transformedDiagonalabweichung
        
            
        self.last_Lärmdurchschnitt = Lärmdurchschnitt        
        self.transformedLärmdurchschnitt = ((Lärmdurchschnitt-self.Lärmmin)/(self.Lärmmax-self.Lärmmin))

        self.Lärmreward= (self.last_transformedLärmdurchschnitt - self.transformedLärmdurchschnitt)*100
        self.last_transformedLärmdurchschnitt = self.transformedLärmdurchschnitt  
                  
        self.last_Lärmpunktzahl = self.Lärmpunktzahl        
        self.transformedLärmpunktzahl = ((self.Lärmpunktzahl-self.Lärmpunktzahlmin)/(self.Lärmpunktzahlmax-self.Lärmpunktzahlmin))        
        self.Lärmrewardintervalle= (self.last_transformedLärmpunktzahl - self.transformedLärmpunktzahl)*100
        self.last_transformedLärmpunktzahl = self.transformedLärmpunktzahl
        
        
        reward = self.Reward.computeGesamtreward(self.MHCreward, self.Rücklaufreward, self.Lärmrewardintervalle)
        
        
        self.Actual_Minimum = self.movingTargetRewardMHC
            
        newState = np.array(self.get_image())
        self.state = newState.copy()
            
        self.internal_state = self.fromState.copy()
        
        if self.step_counter==self.max_steps:
            done = True
        else:
            done = False
        
        return newState, reward, done, {}
        #return newState, reward, done
    
    def render(self, mode=None):
        if self.mode == 'rgb_array':
            #img = Image.fromarray(self.state, 'RGB')     
            img = self.get_image()

        
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    
    def close(self):
        pass
        
    def pairwiseExchange(self, x):
        actions = [(i,j) for i in range(1,x) for j in range(i+1,x+1) if not i==j]
        actions.append((1,1))
        return actions      
    
        # FOR CNN #
    def get_image(self):
        rgb = np.zeros((self.x,self.y,3), dtype=np.uint8)
            
        sources = np.sum(self.TM, axis = 1)
        sinks = np.sum(self.TM, axis = 0)
            
        R = np.array((self.fromState-np.min(self.fromState))/(np.max(self.fromState)-np.min(self.fromState))*255).astype(int)
        G = np.array((sources-np.min(sources))/(np.max(sources)-np.min(sources))*255).astype(int)
        B = np.array((sinks-np.min(sinks))/(np.max(sinks)-np.min(sinks))*255).astype(int)
                        
        k=0
        a=0
        Zeilen_ZAEHLER =0
        for s in range(len(self.fromState)):
            rgb[k][a] = [R[s], G[s], B[s]]
            a+=1
            if a>(self.x-1):
                Zeilen_ZAEHLER+=1
                k= Zeilen_ZAEHLER
                a=0
        
        newState = np.array(rgb)
        self.state = newState.copy()
        img = Image.fromarray(self.state, 'RGB')                     

        return img
    
    
    def Probelauf(self, Durchgänge):
        self.MHCmin = np.inf
        self.MHCmax = 0
        self.Rücklaufmin = np.inf
        self.Rücklaufmax = 0
        self.Diagonalmin = np.inf
        self.Diagonalmax = 0
        self.Lärmutopia = np.inf
        self.Lärmnadir = 0
        self.Lärmpunktzahlmin = np.inf
        self.Lärmpunktzahlmax = 0
        for i in range(Durchgänge):
            self.state_1D = default_rng().choice(range(1,self.n+1), size=self.n, replace=False)
            Probestate = self.state_1D
            
            MHC, self.TM = self.MHC.computeMHC(self.D, self.F, Probestate)
            if MHC <= self.MHCmin:
                self.MHCmin = MHC
            if MHC >= self.MHCmax:
                self.MHCmax = MHC
            
            Rückläufe, self.Flussmatrixneu = self.Rückläufe.computeRückläufe(self.F, Probestate)
            if Rückläufe <= self.Rücklaufmin:
                self.Rücklaufmin = Rückläufe
            if Rückläufe >= self.Rücklaufmax:
                self.Rücklaufmax = Rückläufe
            
            Gesamtdiagonalabweichung, self.Flussmatrixneu = self.Gesamtdiagonalabweichung.computeDiagonalabweichung(self.F, Probestate)
            if Gesamtdiagonalabweichung <= self.Diagonalmin:
                self.Diagonalmin = Gesamtdiagonalabweichung
            if Gesamtdiagonalabweichung >= self.Diagonalmax:
                self.Diagonalmax = Gesamtdiagonalabweichung
            
            self.Lärmpositionen = self.Lärmdummy.computeLärm(self.Lärm1, Probestate)
            LärmMesspunkte, Lärmdurchschnitt = self.Lärmdummy.Lärmberechnung(self.Fabriklänge, self.Fabrikbreite, self.Mittelpunktex, self.Mittelpunktey, self.Lärmpositionen, self.n, self.Messpunkte)
            self.Lärmbereichswerte, self.Lärmmin, self.Lärmmax = self.Lärmdummy.Lärmbereiche(LärmMesspunkte, self.n, self.Messpunkte)
            self.Lärmbereichswerte2, self.Lärmmin2, self.Lärmmax2 = self.Lärmdummy.Lärmbereiche2(LärmMesspunkte, self.n)
            if Lärmdurchschnitt <= self.Lärmutopia:
                self.Lärmutopia = Lärmdurchschnitt
            if Lärmdurchschnitt >= self.Lärmnadir:
                self.Lärmnadir = Lärmdurchschnitt
            
            self.Unter55, self.Unter60, self.Unter65, self.Unter70, self.Unter75, self.Unter80, self.Unter85, self.Über85  = self.Lärmdummy.Lärmintervalle(LärmMesspunkte)
            self.Unter55, self.Unter60, self.Unter65, self.Unter70, self.Unter75, self.Unter80, self.Unter85, self.Über85  = self.Lärmdummy.Lärmintervalle(self.Lärmbereichswerte2)
            Lärmpunktzahl = self.Lärmdummy.LärmIntervallpunkte(self.Unter55, self.Unter60, self.Unter65, self.Unter70, self.Unter75, self.Unter80, self.Unter85, self.Über85)    
            if Lärmpunktzahl <= self.Lärmpunktzahlmin:
                self.Lärmpunktzahlmin = Lärmpunktzahl
            if Lärmpunktzahl >= self.Lärmpunktzahlmax:
                self.Lärmpunktzahlmax = Lärmpunktzahl
                
        return self.MHCmin, self.MHCmax, self.Rücklaufmin, self.Rücklaufmax, self.Diagonalmin, self.Diagonalmax, self.Lärmutopia, self.Lärmnadir, self.Lärmpunktzahlmin, self.Lärmpunktzahlmax