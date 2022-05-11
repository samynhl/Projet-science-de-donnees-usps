# -*- coding: utf-8 -*-

"""
Package: iads
File: Classifiers.py
Année: LU3IN026 - semestre 2 - 2021-2022, Sorbonne Université
"""

# Classfieurs implémentés en LU3IN026
# Version de départ : Février 2022

# Import de packages externes
import numpy as np
import pandas as pd
import copy

# ---------------------------
class Classifier:
    """ Classe (abstraite) pour représenter un classifieur
        Attention: cette classe est ne doit pas être instanciée.
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        raise NotImplementedError("Please Implement this method")
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        raise NotImplementedError("Please Implement this method")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")

    def accuracy(self, desc_set, label_set):
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        correct_pred = 0
        for i in range(len(desc_set)):
            if self.predict(desc_set[i])==label_set[i]:
                correct_pred+=1
        return correct_pred/len(desc_set)

# ---------------------------

class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """

    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        self.input_dimenstion = input_dimension
        self.k = k
        
    def distance(self, x1, x2):
        return np.dot(x1-x2, x1-x2)

    def score(self,x):
        """ rend la proportion de +1 parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        """
        score = 0
        # tableau des distances
        dist = np.asarray([self.distance(x,y) for y in self.desc_set])
        for i in np.argsort(dist)[:self.k]:
            score += 1 if self.label_set[i] == +1 else 0
        return 2 * (score/self.k -.5)
    
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
            x: une description : un ndarray
        """
        return -1 if self.score(x) < 0.5 else 1

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """  
        self.desc_set = desc_set
        self.label_set = label_set

class ClassifierKNN_MC(Classifier):
    """
    Classifieur KNN multi-classe
    """

    def __init__(self, input_dimension, k, nb_class):
        """
        :param input_dimension (int) : dimension d'entrée des exemples
        :param k (int) : nombre de voisins à considérer
        :param nb_class (int): nombre de classes
        Hypothèse : input_dimension > 0
        """
        self.k = k
        self.nb_class = nb_class
        self.data_set = None
        self.label_set = None

    def train(self, data_set, label_set):
        self.data_set = data_set
        self.label_set = label_set

    def score(self, x):
        dist = np.linalg.norm(self.data_set-x, axis=1)
        argsort = np.argsort(dist)
        classes = self.label_set[argsort[:self.k]]
        uniques, counts = np.unique(classes, return_counts=True)
        return uniques[np.argmax(counts)]/self.nb_class

    def predict(self, x):
        return self.score(x)*self.nb_class

# ---------------------------
class ClassifierLineaireRandom(Classifier):
    """ Classe pour représenter un classifieur linéaire aléatoire
        Cette classe hérite de la classe Classifier
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        v = np.random.uniform(-1,1,self.input_dimension)
        self.w = np.random.uniform(-1,1,self.input_dimension) / np.linalg.norm(v)
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """       
        #print("Pas d'apprentissage pour ce classifieur")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(self.w,x)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return -1 if self.score(x)<=0 else +1

# ---------------------------
class ClassifierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension, learning_rate, init=0):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate : epsilon
                - init est le mode d'initialisation de w: 
                    - si 0 (par défaut): initialisation à 0 de w,
                    - si 1 : initialisation par tirage aléatoire de valeurs petites
        """
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
        if (init == 0):
            self.w = np.zeros(input_dimension)
        elif (init == 1):
            self.w = 0.001 * ((2 * np.random.uniform(0, 1, input_dimension) - 1))
        else:
            return -1
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """        
        index_list =[i for i in range(len(desc_set))]
        np.random.shuffle(index_list)
        for i in (index_list):
            Xi, Yi = desc_set[i], label_set[i]
            y_hat = np.dot(self.w, Xi)
            if (y_hat*Yi<=0):
                # self.w += self.learning_rate*np.dot(Xi,Yi)
                np.add(self.w, self.learning_rate*Yi*Xi, out=self.w, casting="unsafe")
     
    def train(self, desc_set, label_set, niter_max=100, seuil=0.01):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - niter_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.01) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """ 
        self.desc_set, self.label_set = desc_set, label_set
        normes_diff_list = []
        for i in range(niter_max):
            w0 = self.w.copy()
            self.train_step(desc_set, label_set)
            normes_diff_list.append(np.linalg.norm(w0-self.w))
            if normes_diff_list[-1]<seuil:
                break
        return normes_diff_list
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.vdot(self.w, x)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return -1 if self.score(x)<=0 else +1
# ---------------------------

# CLasse (abstraite) pour représenter des noyaux
class Kernel():
    """ Classe pour représenter des fonctions noyau
    """
    def __init__(self, dim_in, dim_out):
        """ Constructeur de Kernel
            Argument:
                - dim_in : dimension de l'espace de départ (entrée du noyau)
                - dim_out: dimension de l'espace de d'arrivée (sortie du noyau)
        """
        self.input_dim = dim_in
        self.output_dim = dim_out
        
    def get_input_dim(self):
        """ rend la dimension de l'espace de départ
        """
        return self.input_dim

    def get_output_dim(self):
        """ rend la dimension de l'espace d'arrivée
        """
        return self.output_dim
    
    def transform(self, V):
        """ ndarray -> ndarray
            fonction pour transformer V dans le nouvel espace de représentation
        """        
        raise NotImplementedError("Please Implement this method")

class KernelBias(Kernel):
    """
    Classe pour un noyau simple 2D -> 3D
    Rajoute une colonne à 1
    """
    def transform(self, V):
        V_proj = np.append(V,np.ones((len(V),1)),axis=1)
        return V_proj

class KernelPoly(Kernel):
    def transform(self,V):
        """ ndarray de dim 2 -> ndarray de dim 6            
            ...
        """
        V_proj = np.hstack((np.ones((len(V),1)), V))
        t1 = V[:,0].reshape(len(V), 1)
        V_proj = np.hstack((V_proj, t1*t1))
        t2 = V[:,1].reshape(len(V), 1)
        V_proj = np.hstack((V_proj, t2*t2))
        V_proj = np.hstack((V_proj, t1*t2))
        return V_proj

class ClassifierPerceptronKernel(Classifier):
    """ Perceptron de Rosenblatt kernelisé
    """
    def __init__(self, input_dimension, learning_rate, noyau, init=0):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (espace originel)
                - learning_rate : epsilon
                - noyau : Kernel à utiliser
                - init est le mode d'initialisation de w: 
                    - si 0 (par défaut): initialisation à 0 de w,
                    - si 1 : initialisation par tirage aléatoire de valeurs petites
        """
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
        self.noyau = noyau
        if (init == 0):
            self.w = np.zeros(input_dimension)
        elif (init == 1):
            v = np.random.uniform(0, 1, input_dimension)
            v = (2*v - 1)*0.001
            self.w = v.copy()
        
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments: (dans l'espace originel)
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """        
        index_list =[i for i in range(len(desc_set))]
        np.random.shuffle(index_list)
        # Transformation des données en dimension 6
        desc_set_transformed = self.noyau.transform(desc_set)
        for i in (index_list):
            Xi, Yi = desc_set_transformed[i,:], label_set[i]
            y_hat = np.dot(Xi, self.w)
            if (y_hat*Yi<=0):    # Il y a erreur, donc correction
                self.w += self.learning_rate*np.dot(Xi,Yi)
     
    def train(self, desc_set, label_set, niter_max=100, seuil=0.01):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments: (dans l'espace originel)
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - niter_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.01) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """
        normes_diff_list, accuracy_list = [], []
        for i in range(niter_max):
            w0 = self.w.copy()
            self.train_step(desc_set, label_set)
            accuracy_list.append(self.accuracy(desc_set, label_set))
            normes_diff_list.append(np.linalg.norm(w0-self.w))
            if normes_diff_list[-1]<seuil:
                break
        return normes_diff_list, accuracy_list
    
    def score(self,x):
        """ rend le score de prédiction sur x 
            x: une description (dans l'espace originel)
        """
        a = len(x)
        x_hat = x.reshape(1,a)
        x_hat = self.noyau.transform(x_hat)
        return np.vdot(self.w, x_hat)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description (dans l'espace originel)
        """
        return -1 if self.score(x)<=0 else +1

# ---------------------------

class ClassifierPerceptronBiais(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension, learning_rate, init=0):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate : epsilon
                - init est le mode d'initialisation de w: 
                    - si 0 (par défaut): initialisation à 0 de w,
                    - si 1 : initialisation par tirage aléatoire de valeurs petites
        """
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
        self.allw = []
        if (init == 0): self.w = np.zeros(input_dimension)
        elif (init == 1): self.w = 0.001 * (2 * np.random.uniform(0, 1, input_dimension) - 1)
        self.allw.append(self.w.copy())
        
    def get_allw(self):
        return self.allw
    
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """        
        index_list =[i for i in range(len(desc_set))]
        np.random.shuffle(index_list)
        for i in (index_list):
            Xi, Yi = desc_set[i,:], label_set[i]
            y_hat = np.dot(self.w, Xi)
            if (y_hat*Yi<1):    # Il y a erreur, donc correction
                self.w += self.learning_rate*np.dot(Xi,Yi)
                self.allw.append(self.w.copy())

    def train(self, desc_set, label_set, niter_max=100, seuil=0.01):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - niter_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.01) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """ 
        self.desc_set, self.label_set = desc_set, label_set
        normes_diff_list = []
        for i in range(niter_max):
            w0 = self.w.copy()
            self.train_step(desc_set, label_set)
            normes_diff_list.append(np.linalg.norm(w0-self.w))
            if normes_diff_list[-1]<seuil:
                break
        return normes_diff_list
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.vdot(self.w, x)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return -1 if self.score(x)<=0 else +1

# Perceptron Multi classe
class PerceptronMultiOOA(Classifier):
    """ Perceptron multiclass OneVsAll
    """
    def __init__(self, input_dimension, learning_rate,nbC=10 , init=0):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate : epsilon
                - nbC : nombre de classes du dataset
                - init est le mode d'initialisation de w: 
                    - si 0 (par défaut): initialisation à 0 de w,
                    - si 1 : initialisation par tirage aléatoire de valeurs petites
        """
        # Liste de 10 classifieurs perceptrons: 1 pour chaque classe
        self.classifieurs = [ClassifierPerceptron(input_dimension, learning_rate) for i in range(10)]

    def train(self, desc_set, label_set):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - niter_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.01) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """        
        for i in range(len(self.classifieurs)):
            # print('Iteration {}'.format(i))
            ytmp = np.where(label_set==i, 1, -1)
            self.classifieurs[i].train(desc_set,ytmp)
    
    def score(self,x):
        """ rend le score de prédiction de chaque classifieur sur x (valeur réelle)
            x: une description
        """
        a = []
        for i in range(len(self.classifieurs)):
            a.append(self.classifieurs[i].score(x))
        return a
    
    def predict(self, x):
        """ rend la prediction (classe) sur x de 0:9
            x: une description
        """
        return np.argmax(self.score(x))
    
    def accuracy(self, desc_set, label_set):
        yhat = np.array([self.predict(x) for x in desc_set])
        return np.where(label_set == yhat, 1., 0.).mean()


# -----------------------------------------------------------------
class ClassifierADALINE(Classifier):
    """ Perceptron de ADALINE
    """
    def __init__(self, input_dimension, learning_rate, history=False, niter_max=1000):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
                - learning_rate : epsilon
                - history : stockage des poids w en cours d'apprentissage
                - niter_max : borne sur les iterations
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
        self.niter_max = niter_max
        self.history = history
        self.allw = []
        # Initialisation de w aléatoire
        self.w = 0.001 * (2 * np.random.uniform(0, 1, input_dimension) - 1)
        self.allw.append(self.w)
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        seuil=0.01
        for k in range(self.niter_max):
            index_list =[i for i in range(len(desc_set))]
            np.random.shuffle(index_list)
            for i in (index_list):
                Xi, Yi = desc_set[i,:], label_set[i]
                grad = np.dot(Xi.T,np.dot(Xi,self.w)-Yi)
                w0 = self.w.copy()
                self.w -= self.learning_rate*grad
                self.allw.append(self.w.copy())
            difference = np.linalg.norm(w0-self.w)
            if (difference<seuil):
                break
    
    def get_allw(self):
        return self.allw if self.history else []
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.vdot(self.w, x)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return -1 if self.score(x)<=0 else +1

class ClassifierADALINE2(Classifier):
    """ Perceptron de ADALINE2
    """
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
                - learning_rate : epsilon
                - history : stockage des poids w en cours d'apprentissage
                - niter_max : borne sur les iterations
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        # code de la classe ADALINE Analytique
        alpha = 1e-5
        a = desc_set.T@desc_set + alpha*np.eye(desc_set.shape[1])
        b = desc_set.T@label_set
        self.w = np.linalg.solve(a,b)
    
    def get_allw(self):
        return self.allw if self.history else []
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.vdot(self.w, x)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return -1 if self.score(x)<=0 else 1