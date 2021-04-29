import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import sklearn.svm
import random
import warnings
import csv
from math import floor
from sklearn.metrics import classification_report, balanced_accuracy_score,confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
tic=time.time()

warnings.filterwarnings("ignore")


class ELM (BaseEstimator, ClassifierMixin):

    """
    3 step model ELM
    """

    def __init__(self,hid_num,a=1):
        """
        Args:
        hid_num (int): number of hidden neurons
        a (int) : const value of sigmoid funcion
        """
        self.hid_num = hid_num
        self.a = a

    def _sigmoid(self, x):
        """
        sigmoid function
        Args:
        x (float): input
        Returns:
        float: output of sigmoid
        """
        sigmoid_range = 34.538776394910684
        x = np.clip(x, -sigmoid_range, sigmoid_range)
        return 1 / (1 + np.exp(-self.a * x))

    def _add_bias(self, X):
        """add bias to list
        Args:
        x_vs [[float]] Array: vec to add bias
        Returns:
        [float]: added vec
        Examples:
        >>> e = ELM(10, 3)
        >>> e._add_bias(np.array([[1,2,3], [1,2,3]]))
        array([[1., 2., 3., 1.],
               [1., 2., 3., 1.]])
        """

        return np.c_[X, np.ones(X.shape[0])]

    def _ltov(self, n, label):
        """
        trasform label scalar to vector
        Args:
        n (int) : number of class, number of out layer neuron
        label (int) : label
        Exmples:
        >>> e = ELM(10, 3)
        >>> e._ltov(3, 1)
        [1, -1, -1]
        >>> e._ltov(3, 2)
        [-1, 1, -1]
        >>> e._ltov(3, 3)
        [-1, -1, 1]
        """
        return [-1 if i != label else 1 for i in range(1, n + 1)]

    def fit(self, X, y):
        """
        learning
        Args:
        X [[float]] array : feature vectors of learnig data
        y [[float]] array : labels of leanig data
        """
        # number of class, number of output neuron
        self.out_num = max(y)

        if self.out_num != 1:
            y = np.array([self._ltov(self.out_num, _y) for _y in y])

        # add bias to feature vectors
        X = self._add_bias(X)

        # generate weights between input layer and hidden layer
        np.random.seed()
        self.W = np.random.uniform(-1., 1.,
                                   (self.hid_num, X.shape[1]))

        # find inverse weight matrix
        _H = np.linalg.pinv(self._sigmoid(np.dot(self.W, X.T)))

        self.beta = np.dot(_H.T, y)

        return self

    def predict(self, X):
        """
        predict classify result
        Args:
        X [[float]] array: feature vectors of learnig data
        Returns:
        [int]: labels of classification result
        """
        _H = self._sigmoid(np.dot(self.W, self._add_bias(X).T))
        y = np.dot(_H.T, self.beta)

        if self.out_num == 1:
            return np.sign(y)
        else:
            return np.argmax(y, 1) + np.ones(y.shape[0])



# Required functions for GA
def reduce_features(solution, features):
    selected_elements_indices = np.where(solution == 1)[0]
    reduced_features = features[:, selected_elements_indices]
    return reduced_features


def classification_accuracy(labels, predictions):
    correct = np.where(labels == predictions)[0]
    accuracy = correct.shape[0]/labels.shape[0]
    return accuracy

def metrics(labels,predictions,classes):
    print("Classification Report:")
    print(classification_report(labels, predictions, target_names = classes))
    matrix = confusion_matrix(labels, predictions)
    print("Confusion matrix:")
    print(matrix)
    print("Classwise Accuracy :{}".format(matrix.diagonal()/matrix.sum(axis = 1)))
    print("Balanced Accuracy Score: ",balanced_accuracy_score(labels,predictions))
    

def cal_pop_fitness(pop, train_datas, train_labels, validation_datas, validation_labels, test_datas, test_labels,classifier):
    accuracies1 = np.zeros(pop.shape[0])
    accuracies2 = np.zeros(pop.shape[0])
    idx = 0

    for curr_solution in pop:
        reduced_train_features = reduce_features(curr_solution, train_datas)
        reduced_test_features = reduce_features(curr_solution, test_datas)
        reduced_validation_features = reduce_features(curr_solution, validation_datas)
        X=reduced_train_features
        y=train_labels

        if classifier == 'MLP':
            ## MLP CLASSIFIER ##
            MLP_classifier = MLPClassifier(activation = 'tanh',solver = 'lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
            MLP_classifier.fit(X, y)
            predictions1 = MLP_classifier.predict(reduced_validation_features)
            predictions2 = MLP_classifier.predict(reduced_test_features)

        elif classifier == 'ELM':
            ## ELM CLASSIFIER ##
            ELM_classifier = ELM(hid_num = 500)
            ELM_classifier.fit(X,y)
            predictions1 = ELM_classifier.predict(reduced_validation_features)
            predictions2 = ELM_classifier.predict(reduced_test_features)

        else:
            ## SVM CLASSIFIER ##
            SVM_classifier = sklearn.svm.SVC(kernel='rbf',gamma='scale',C=5000)
            SVM_classifier.fit(X, y)
            predictions1 = SVM_classifier.predict(reduced_validation_features)
            predictions2 = SVM_classifier.predict(reduced_test_features)

        accuracies1[idx] = classification_accuracy(validation_labels, predictions1)
        accuracies2[idx] = classification_accuracy(test_labels, predictions2)      
        
        idx = idx + 1
    return accuracies1, accuracies2 , predictions1, predictions2

def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = np.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents


def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually, it is at the center.
    crossover_point = np.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
        
        # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]
        
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring


def mutation(offspring_crossover, num_mutations=2):
    mutation_idx = np.random.randint(low=0, high=offspring_crossover.shape[1], size=num_mutations)
    # Mutation changes a single gene in each offspring randomly.
    for idx in range(offspring_crossover.shape[0]):
        # The random value to be added to the gene.
        offspring_crossover[idx, mutation_idx] = 1 - offspring_crossover[idx, mutation_idx]
    return offspring_crossover



#Implementation of Genetic algorithm
f = 'F:/ND sirs project/ALL DATASETS/HErlev/HErlev_ResNet18_b_original_.csv'
dataframe = pd.read_csv(f)
df= pd.read_csv('F:/ND sirs project/ALL DATASETS/HErlev/HErlev_Class.csv')
f= open(f,'r')
reader = csv.reader(f)
labels = next(reader)

f = 'F:/ND sirs project/ALL DATASETS/HErlev/HErlev_GoogLeNet_b_original_.csv'
dataframe2 = pd.read_csv(f)
f= open(f,'r')
reader = csv.reader(f)
labels2 = next(reader)

dataframe[labels2]= dataframe2
labels += labels2

feature_df = dataframe[labels]
class_df=df['Class']

data_inputs = np.asarray(feature_df)
data_outputs = np.asarray(class_df)

##NO OF CLASSES
unique=np.unique(data_outputs)
num_classes=unique.shape[0]
classes=[]
for i in range(num_classes):
    classes.append('Class'+str(i+1))


# The main program
num_samples = data_inputs.shape[0]
num_feature_elements = data_inputs.shape[1]
print("Number of features: ",num_feature_elements)


########################### For KFold Cross Validation ##############################


from numpy import array
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
Fold = 5

kfold = KFold(Fold, True, 1)
   


# The main program


##################################################### CHOOSE PARAMETERS #########################################################################
sol_per_pop = 100 # Population size
num_parents_mating = (int)(sol_per_pop*0.5) # Number of parents inside the mating pool.
num_mutations = 6#(int)(sol_per_pop*num_feature_elements*6/100) # Number of elements to mutate.
num_generations = 20 #Number of generations in each fold
classifier = 'MLP'
#################################################################################################################################################


print("Population size: {}".format(sol_per_pop))
print("Number of parents inside mating pool: {}".format(num_parents_mating))
print("Number of elements to mutate: {}".format(num_mutations))

val=0
test=0
acc=0
f=0
for train_index, test_index in kfold.split(data_inputs):
    train_data, train_label = np.asarray(data_inputs[train_index]), np.asarray(data_outputs[train_index])
    test_data, test_label = np.asarray(data_inputs[test_index]), np.asarray(data_outputs[test_index])

    validation_data, test_data, validation_label, test_label = train_test_split(test_data, test_label, test_size=0.5, random_state=4)

    # Defining the population shape.
    pop_shape = (sol_per_pop, num_feature_elements)

    # Creating the initial population.
    new_population = np.random.randint(low=0, high=2, size=pop_shape)
    new_population_test = np.random.randint(low=0, high=2, size=pop_shape)

    best_outputs_test = []
    best_outputs = []

    for generation in range(num_generations):
        print("\nFold: {}, Generation : {}".format(f+1,generation+1))
        # Measuring the fitness of each chromosome in the population.
        fitness, test_accuracy, prediction_validation,  prediction_test = cal_pop_fitness(new_population, train_data, train_label, validation_data, validation_label,
                                                                                          test_data, test_label, classifier)

        #Uncomment the following block of code if you want the metrics for each generation
        
        print(classification_report(validation_label, prediction_validation, target_names=classes))
        print(classification_report(test_label, prediction_test, target_names=classes))
        matrix_test = confusion_matrix(test_label, prediction_test)
        matrix_validation = confusion_matrix(validation_label, prediction_validation)
        print("Confusion matrix for test set:")
        print(matrix_test)
        print("Confusion Matrix for validation set: ")
        print(matrix_validation)
        print("Classwise Accuracy for test:{}".format(matrix_test.diagonal()/matrix_test.sum(axis = 1)))
        print("Classwise Accuracy for validation:{}".format(matrix_validation.diagonal()/matrix_validation.sum(axis = 1)))
        
        
        balanced = balanced_accuracy_score(test_label, prediction_test)
        print("Balanced Accuracy Score for test images: {}".format(balanced))
        balanced_val = balanced_accuracy_score(validation_label, prediction_validation)
        print("Balanced Accuracy Score for test images: {}".format(balanced_val))
        
        best_outputs.append(np.max(fitness))
        best_outputs_test.append(np.max(test_accuracy))
        
        print("Best validation result : ", best_outputs[-1])
        print("Best test result : ", best_outputs_test[-1])

        #Save the best combination for printing later
        if best_outputs[-1]>=0.99*val and best_outputs_test[-1]>=0.99*test and balanced>=0.99*acc:
            best_match_idx = np.where(fitness == np.max(fitness))[0]
            best_match_idx = best_match_idx[0]
            
            best_solution = new_population[best_match_idx, :]
            best_solution_indices = np.where(best_solution == 1)[0]
            indi=best_solution_indices.tolist()
            best_sol=[]
            for i in range(num_feature_elements):
                if i in indi:
                    best_sol.append(1)
                else:
                    best_sol.append(0)
            best_sol=np.array(best_sol)
            best_solution_num_elements = best_solution_indices.shape[0]
            best_solution_fitness = fitness[best_match_idx]
            predict_test=prediction_test
            predict_validation=prediction_validation

            t_label=test_label
            val_label=validation_label

            val=best_outputs[-1]
            test=best_outputs_test[-1]
            acc=balanced
            gen=generation+1
            fold=f+1

        # Selecting the best parents in the population for mating.
        parents = select_mating_pool(new_population, fitness, num_parents_mating)

        # Generating next generation using crossover.
        offspring_crossover = crossover(parents, offspring_size=(pop_shape[0]-parents.shape[0], num_feature_elements))

        # Adding some variations to the offspring using mutation.
        offspring_mutation = mutation(offspring_crossover, num_mutations=num_mutations)

        # Creating the new population based on the parents and offspring.
        new_population[0:parents.shape[0], :] = parents
        new_population[parents.shape[0]:, :] = offspring_mutation

    f+=1
        
# Getting the best solution after iterating finishing all generations.
print("\n\n\nBest scores obtained at Fold {}, Generation: {}".format(fold,gen))
print("Best match Index : ", best_match_idx)
print("Best Solution : ", best_sol)
print("Selected indices : ", best_solution_indices)
print("Number of selected elements : ", best_solution_num_elements)
print("Best solution fitness : ", best_solution_fitness)


print("\n\nMetrics for Validation Set:")
metrics(val_label,predict_validation,classes)
##print("Classification Report:")
##print(classification_report(val_label, predict_validation, classes))
##matrix_validation = confusion_matrix(val_label, predict_validation)
##print("Confusion matrix:")
##print(matrix_validation)
##print("Classwise Accuracy :{}".format(matrix_validation.diagonal()/matrix_validation.sum(axis = 1)))
##print("Balanced Accuracy Score: ",balanced_accuracy_score(val_label,predict_validation))

print("\n\nMetrics for Test Set:")
metrics(t_label,predict_test,classes)
##print("Classification Report:")
##print(classification_report(t_label, predict_test, classes))
##matrix_test = confusion_matrix(t_label, predict_test)
##print("Confusion matrix:")
##print(matrix_test)
##print("Balanced Accuracy Score: ",acc)

print("GoogleNet")
toc=time.time()
print("\nComputation time is {} minutes".format((toc-tic)/60))
