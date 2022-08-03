import numpy as np
import math
import statistics

class Flusskennzahlen():

    def __init__(self, shape=None, dtype=np.float32):
        self.shape = shape
        
    def computeMHC(self, D, F, s):
        # Compute reward for taking action on old state:  
        # Calculate permutation matrix for new state
        P = self.permutationMatrix(s)     
        Distanz = np.dot(D,P)
        Fluss=np.dot(F,P.T)
        transport_intensity = np.dot(Distanz, Fluss)
        MHC = np.trace(transport_intensity)
                
        return MHC, transport_intensity
    
    def computeRückläufe(self, F, s):
        # Compute reward for taking action on old state:  
        P = self.permutationMatrix(s)
        Fluss=np.dot(P,F)
        Fluss= Fluss.T    
        Positionsfluss = np.zeros((len(Fluss[0]),len(Fluss[0])))
        Rückläufe = 0          #Zählt zeilenweise die Rückläufe bzw. Werte unter der Hauptdiagonalen
        q = 0                  #Dummy-Variable für Das Zählen der Zeilen
        for i in range(len(Fluss[0])):
            Maschine = s[i]
            for j in range(len(Fluss[0])):
                Positionsfluss[i][j] = Fluss[Maschine-1][j]
        
        for x in range(len(Fluss[0])):  
            for y in range(q):      #Zählt nur bis zum Diagonalelement der jeweiligen Zeile hoch
                Rückläufe+= Positionsfluss[x,y]  #Zählt die Rückläufe über alle Zeilen hoch
            q=q+1                   #Sprung in nächste Zeile
                
        return Rückläufe, Fluss
    
    def computeDiagonalabweichung(self, F, s):
        # Compute reward for taking action on old state:  
        P = self.permutationMatrix(s)
        Fluss=np.dot(F,P.T)    #Fluss aufgrund aktueller Maschinen-Anordnung
        Gesamtzähler=0
        Aktuelles_Diagonalelement=0
        for x in range(len(Fluss[0])):
            Diagonalabweichung=0
            for y in range(len(Fluss[0])):
                if Fluss[x,y] != 0 and y != Aktuelles_Diagonalelement:
                    AktuellerSpeicherwert= abs(y - Aktuelles_Diagonalelement) 
                    if AktuellerSpeicherwert>Diagonalabweichung:
                        Diagonalabweichung=AktuellerSpeicherwert
            Aktuelles_Diagonalelement+=1
            Gesamtzähler += Diagonalabweichung
        return Gesamtzähler, Fluss




    
    def permutationMatrix(self, a):
        P = np.zeros((len(a), len(a)))
        for idx,val in enumerate(a):
            P[idx][val-1]=1
        return P
    

    
    def computeLärm(self, Lärmmatrix, s):
        P = self.permutationMatrix(s)
        Lärmposition = np.zeros(len(Lärmmatrix))
        for i in range(len(Lärmmatrix)):
            #Lärm der Maschine x angeben wenn an Pos y steht
            for j in range(len(Lärmmatrix)):
                if P[i,j]==1:
                    Lärmposition[i]= Lärmmatrix[j]
        
                
        return Lärmposition
    
    
    

    def computeGesamtreward(self, Wert1, Wert2, Wert3):  #Gewichtungen beliebig anpassbar
        Gewichtung1 = 1  #MHC 0.5
        Gewichtung2 = 0   #Rückläufe 0.25
        Gewichtung3 = 0   #Lärm
        #Gewichtung4= 0   #Diagonalabweichung
        Reward = Gewichtung1 * Wert1 + Gewichtung2 * Wert2 + Gewichtung3 * Wert3
        return Reward




    def Flussmatrix(self):
        F = np.zeros((9, 9))

        F[0][0]=0
        F[0][1]=10
        F[0][2]=12
        F[0][3]=15
        F[0][4]=17
        F[0][5]=11
        F[0][6]=20
        F[0][7]=22
        F[0][8]=19

        F[1][0]=1
        F[1][1]=0
        F[1][2]=13
        F[1][3]=18
        F[1][4]=7
        F[1][5]=2
        F[1][6]=1       
        F[1][7]=1
        F[1][8]=104
        
        
        F[2][0]=2
        F[2][1]=3
        F[2][2]=0
        F[2][3]=100
        F[2][4]=109
        F[2][5]=17
        F[2][6]=100
        F[2][7]=1
        F[2][8]=31

        
        F[3][0]=5
        F[3][1]=1
        F[3][2]=11
        F[3][3]=0
        F[3][4]=0
        F[3][5]=78
        F[3][6]=247
        F[3][7]=178
        F[3][8]=1

        
        F[4][0]=2
        F[4][1]=17
        F[4][2]=12
        F[4][3]=9
        F[4][4]=0
        F[4][5]=1
        F[4][6]=10
        F[4][7]=1
        F[4][8]=79
        
        F[5][0]=9
        F[5][1]=14
        F[5][2]=8
        F[5][3]=21
        F[5][4]=30
        F[5][5]=0
        F[5][6]=0
        F[5][7]=1
        F[5][8]=0
        
        F[6][0]=11
        F[6][1]=19
        F[6][2]=25
        F[6][3]=31
        F[6][4]=7
        F[6][5]=2
        F[6][6]=0
        F[6][7]=0
        F[6][8]=0

        F[7][0]=5
        F[7][1]=4
        F[7][2]=12
        F[7][3]=19
        F[7][4]=23
        F[7][5]=31
        F[7][6]=40
        F[7][7]=0
        F[7][8]=12
        
        F[8][0]=8
        F[8][1]=11
        F[8][2]=25
        F[8][3]=29
        F[8][4]=9
        F[8][5]=7
        F[8][6]=2
        F[8][7]=5
        F[8][8]=0        

        
        return F
    
    
    
    
    def Lärmmatrix(self):
        L = np.zeros(9)

        L[0]=80  #Drehen    #80
        L[1]=75  #Bohren    #75
        L[2]=70  #Fräsen    #85
        L[3]=95 #Sägen     #105   #95  #60
        L[4]=70  #Lackieren   #70
        L[5]=55  #Prüfen      #55
        L[6]=63  #Warenausgang  #63
        L[7]=72
        L[8]=63  #Warenausgang  #63
        
        return L
    
    def Maschinenmittelpunkte(self,L,W, Maschinenanzahl):
       Maschinenmittelpunktex=[]
       Maschinenmittelpunktey=[]
       x_Schrittweite = L / (Maschinenanzahl*2)
       y_Schrittweite = W / (Maschinenanzahl*2)
       for M in range(Maschinenanzahl):
           for N in range(Maschinenanzahl):             #anstatt 4 evtl. range(1,self.x+1) aus Umgebung nehmen
               Maschinenmittelpunktex.append(x_Schrittweite + 2*M*x_Schrittweite)      
               Maschinenmittelpunktey.append(y_Schrittweite + 2*N*y_Schrittweite)



       return Maschinenmittelpunktex, Maschinenmittelpunktey
   
    def Lärmberechnung(self,L,W, MPx, MPy, Lärm, Maschinenanzahl, Messpunkte):
        #Messpunkte  #pro Zeile/Spalte
        x_Schrittweite=L/Messpunkte   
        y_Schrittweite=L/Messpunkte
        Lärm_insgesamt=[]
        for k in range(Messpunkte+1):   #x-Richtung MP
            for l in range(Messpunkte+1): #y-RIchtung MP
                i=0
                MP=[x_Schrittweite*k,y_Schrittweite*l]  #Messpunkte betrachten
                #print(MP)
                temp=0
                r = np.zeros(Maschinenanzahl)
                Lärm_MP = np.zeros(Maschinenanzahl)
                for i in range(Maschinenanzahl): 
                    r[i] = math.sqrt(((MPx[i]-MP[0])**2+(MPy[i]-MP[1])**2))
                    if r[i]==0:
                        r[i]=1
                    Lärm_MP[i] = Lärm[i] - 20 * math.log10(r[i])
                    
                    temp += 10**(0.1*Lärm_MP[i])
                    
                Lärm_neu = 10 * math.log10(temp)
                Lärm_insgesamt.append(Lärm_neu)
        Lärm_Durchschnitt = statistics.mean(Lärm_insgesamt)


        return Lärm_insgesamt, Lärm_Durchschnitt
        
    
    
    def Lärmbereiche(self, Lärminsgesamt, Maschinenanzahl, Messpunkte):
        Zähldummy = 0
        Lärmbereichswerte = []
        Sprung_Bereich = 0
        for k in range(1, Maschinenanzahl+1):   #9 Bereiche zu untersuchen
            Sprung_Bereich += 5  #Verschiebung in x-Richtung zum nächsten Bereich  #Sprungbereich = math.ceil((math.sqrt(Maschinenanzahl)))
            Lärmwerte=[]
            for m in range(math.ceil((math.sqrt(Maschinenanzahl)))+3):     #6 "Zeilen" pro Bereich
                for l in range(Zähldummy,Zähldummy+6):                   #6 "Spalten" pro Bereich
                    #print(l)
                    AktuellerLärmwert = Lärminsgesamt[l]
                    Lärmwerte.append(AktuellerLärmwert)
                Zähldummy+=Messpunkte
            Lärmwert = statistics.mean(Lärmwerte)
            Lärmbereichswerte.append(Lärmwert)
            Zähldummy = Sprung_Bereich
            if k % 3 == 0:      #Nach oberen Bereich Dummy = 6
                Sprung_Bereich += 65   #Messpunkte+1
                Zähldummy = Sprung_Bereich
        Lärm_Min = min(Lärmbereichswerte)
        Lärm_Max = max(Lärmbereichswerte)
        return Lärmbereichswerte, Lärm_Min, Lärm_Max
    


    
    def Distanzmatrixneu(self, MPx, MPy):
        D = np.zeros((9, 9))

        for i in range(9):
            for j in range(9):
                D[i][j] = math.sqrt((MPx[j]-MPx[i])**2+(MPy[j]-MPy[i])**2)
  
        return D
    
    
    def Lärmintervalle(self,LärmMesspunkte):
        Unter55=0
        Unter60=0
        Unter65=0
        Unter70=0
        Unter75=0
        Unter80=0
        Unter85=0
        Über85=0
        for z in range(len(LärmMesspunkte)):
            if LärmMesspunkte[z]<55:
                Unter55+=1
            if LärmMesspunkte[z]>=55 and LärmMesspunkte[z]<60:
                Unter60+=1
            if LärmMesspunkte[z]>=60 and LärmMesspunkte[z]<65:
                Unter65+=1
            if LärmMesspunkte[z]>=65 and LärmMesspunkte[z]<70:
                Unter70+=1
            if LärmMesspunkte[z]>=70 and LärmMesspunkte[z]<75:
                Unter75+=1
            if LärmMesspunkte[z]>=75 and LärmMesspunkte[z]<80:
                Unter80+=1
            if LärmMesspunkte[z]>=80 and LärmMesspunkte[z]<85:
                Unter85+=1
            if LärmMesspunkte[z]>=85:
                Über85+=1
        return Unter55, Unter60, Unter65, Unter70, Unter75, Unter80, Unter85, Über85
    
    def LärmIntervallpunkte(self, Unter55, Unter60, Unter65, Unter70, Unter75, Unter80, Unter85, Über85):
        a=1
        b=2
        c=3
        d=4
        e=5
        f=6
        g=9
        h=12
        Punktzahl= a * Unter55 + b * Unter60 + c * Unter65 + d * Unter70 + e * Unter75 + f * Unter80 + g * Unter85 + h * Über85
        return Punktzahl
    
    
    def Lärmbereiche2(self, Lärminsgesamt, Maschinenanzahl):
        Messpunkte= 16
        Zähldummy = 0
        Lärmbereichswerte = []
        Sprung_Bereich = 0
        for k in range(1, 240):   #9 Bereiche zu untersuchen
            Sprung_Bereich += 1  #Verschiebung in x-Richtung zum nächsten Bereich  #Sprungbereich = math.ceil((math.sqrt(Maschinenanzahl)))
            Lärmwerte=[]
            for m in range(2):     #6 "Zeilen" pro Bereich
                for l in range(Zähldummy,Zähldummy+2):                   #6 "Spalten" pro Bereich
                    #print(l)
                    AktuellerLärmwert = Lärminsgesamt[l]
                    Lärmwerte.append(AktuellerLärmwert)
                Zähldummy+=Messpunkte
            Lärmwert = statistics.mean(Lärmwerte)
            Lärmbereichswerte.append(Lärmwert)
            Zähldummy = Sprung_Bereich
        Lärm_Min = min(Lärmbereichswerte)
        Lärm_Max = max(Lärmbereichswerte)
        return Lärmbereichswerte, Lärm_Min, Lärm_Max
    
    
    
 