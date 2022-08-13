#!/usr/bin/python3
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
df = pd.read_excel('Hazara1.xlsx')#, sheet_name='Sheet1' by default sheet name is 1, if you other number then mention it using the argument sheet_name


column_names = list(df.columns.values) # Extracting Column names

 


df.dropna(how="all", inplace=True)


from sklearn.preprocessing import LabelEncoder
import numpy as np
import math




names_population = list(df.Population.unique()) # Put here the name of last column, Population in this case
dict_of_populations = { i : names_population[i] for i in range(0, len(names_population) ) }


X = df[column_names[:-1]].values # Number of columns (not including classification/population column)
y = df[column_names[-1]].values
enc = LabelEncoder()
label_encoder = enc.fit(y)
y = label_encoder.transform(y) + 1
label_dict = dict_of_populations # Classification/Population

length_of_cols= len(column_names) - 1 # all columns except last column to be used in program


from matplotlib import pyplot as plt
np.set_printoptions(precision=15) # Number of array lists that can be printed.For printing whole array np.set_printoptions(threshold='nan')
mean_vectors = []

for cl in range(1,len(names_population)+1): # Where range len(names_population	) shows the number of populations
    mean_vectors.append(np.mean(X[y==cl],axis=0))
    print(cl, mean_vectors[cl-1])




S_W = np.zeros((length_of_cols,length_of_cols)) # According to the number of features i.e number of columns before populations.
for cl,mv in zip(range(1,length_of_cols), mean_vectors): # same as above
    class_sc_mat = np.zeros((length_of_cols,length_of_cols)) # same as above
    for row in X[y == cl]:
        row, mv = row.reshape(length_of_cols,1), mv.reshape(length_of_cols,1) # same as above
        class_sc_mat += (row-mv).dot((row-mv).T)
    S_W += class_sc_mat

print('within-class Scatter Matrix:\n', S_W)



overall_mean = np.mean(X, axis=0)
S_B = np.zeros((length_of_cols,length_of_cols)) # same as above
for i,mean_vec in enumerate(mean_vectors): 
    n = X[y==i+1,:].shape[0]
    mean_vec = mean_vec.reshape(length_of_cols,1) # same as above
    overall_mean = overall_mean.reshape(length_of_cols,1) # same as above
    S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
 


eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
for i in range(len(eig_vals)):
    eigvec_sc = eig_vecs[:,i].reshape(length_of_cols,1)  # same as above
    print('\nEigenvector {}: \n{}'.format(i+1, eigvec_sc.real))
    print('Eigenvalue {:}: {:.2e}'.format(i+1, eig_vals[i].real))
 


for i in range(len(eig_vals)):
    eigv = eig_vecs[:,i].reshape(length_of_cols,1) # same as above
    np.testing.assert_array_almost_equal(np.linalg.inv(S_W).dot(S_B).dot(eigv),
    eig_vals[i] * eigv, decimal=6, err_msg='', verbose=True)

print("Ok")



eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True) 
for i in eig_pairs:
    print(i[0])

 

eigv_sum = sum(eig_vals)
for i,j in enumerate(eig_pairs):
    print('eigenvalue {0:}: {1:.2%}'.format(i+1, (j[0]/eigv_sum).real))



W = np.hstack((eig_pairs[0][1].reshape(length_of_cols,1), eig_pairs[1][1].reshape(length_of_cols,1))) # same as above
print('Matrix W:\n', W.real)


X_lda = X.dot(W)
from matplotlib import pyplot as plt

def plot_step_lda():
    ax = plt.subplot(111)
    for label,marker,color in zip(
        range(0,5),('o','o','o','o','o','o'),('blue','green', 'red' ,'magenta','yellow','brown')): # where range is acc. to # of population
        plt.scatter(x=X_lda[:,0].real[y == label],
        y=X_lda[:,1].real[y == label],
        marker = marker,color = color,alpha = 0.5,label=label_dict[label],s=8)
    plt.xlabel('LD1')
    plt.ylabel('LD2')
    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title('Linear Discriminant Analysis')
    plt.tick_params(axis="both", which="both", bottom="off", top="off", 
    labelbottom="on", left="off", right="off", labelleft="on")
    ax.spines["top"].set_visible(False)  
    ax.spines["bottom"].set_visible(False)  
    ax.spines["left"].set_visible(False)  
    ax.spines["right"].set_visible(False)  
    plt.grid()
    plt.tight_layout
    plt.savefig('hazara_path6.eps', format='eps', dpi=1000) # saves the figure in eps format
    plt.show()
 
plot_step_lda()


