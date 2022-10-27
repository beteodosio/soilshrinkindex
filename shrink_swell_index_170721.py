import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop

from matplotlib import pyplot
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd 

from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras import regularizers
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np
from sklearn import metrics

save_path = "."
#%%

# Import data from the CSV file "dataset" ###

read_data = pd.read_csv('geodataset.csv', skiprows=range(0,1), delimiter=',', index_col=False,
                   names=['Iss','LL','PL','PI','LS'])

dataset = pd.DataFrame(read_data, columns=['Iss','LL','PL','PI','LS'])

y = dataset.iloc[:,0]
X = dataset.iloc[:,1:5]


#Normalisation
# LL_min = np.min(dataset.iloc[:,1])
# LL_max =np.max(dataset.iloc[:,1])

# PL_min = np.min(dataset.iloc[:,2])
# PL_max = np.max(dataset.iloc[:,2])

# PI_min = np.min(dataset.iloc[:,3])
# PI_max = np.max(dataset.iloc[:,3])

# LS_min = np.min(dataset.iloc[:,4])
# LS_max = np.max(dataset.iloc[:,4])

# X_LL =  (dataset.iloc[:,1] - LL_min)/(LL_max - LL_min)
# X_PL =  (dataset.iloc[:,2] - PL_min)/(PL_max - PL_min)
# X_PI =  (dataset.iloc[:,3] - PI_min)/(PI_max - PI_min)
# X_LS =  (dataset.iloc[:,4] - LS_min)/(LS_max - LS_min)

# X = pd.concat([X_LL, X_PL, X_PI, X_LS], axis=1)

# fit scaler on training data
norm = StandardScaler().fit(X)
# transform training data
X = norm.transform(X)


# split into input (X) and output (y) variables
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)




#%%
# Create model
model = Sequential()
model.add(Dense(64, kernel_initializer='he_uniform', activation="relu", input_dim=4,kernel_regularizer=regularizers.l2(2)))
model.add(Dense(64, kernel_initializer='he_uniform', activation="relu"))
model.add(Dense(64, kernel_initializer='he_uniform', activation="relu"))
model.add(Dense(64, kernel_initializer='he_uniform', activation="relu"))
model.add(Dense(64, kernel_initializer='he_uniform', activation="relu"))
model.add(Dense(64, kernel_initializer='he_uniform', activation="relu"))
model.add(Dense(64, kernel_initializer='he_uniform', activation="relu"))
model.add(Dense(64, kernel_initializer='he_uniform', activation="relu"))
model.add(Dense(64, kernel_initializer='he_uniform', activation="relu"))
model.add(Dense(64, kernel_initializer='he_uniform', activation="relu"))
model.add(Dense(64, kernel_initializer='he_uniform', activation="relu"))
model.add(Dense(64, kernel_initializer='he_uniform', activation="relu"))
model.add(Dense(64, kernel_initializer='he_uniform', activation="relu"))
# Since the regression is performed, a Dense layer containing a single neuron with a linear activation function.
# Typically ReLu-based activation are used but since it is performed regression, it is needed a linear activation.
model.add(Dense(1, activation="linear"))

#%%
# Compile model: The model is initialized with the Adam optimizer and then it is compiled.
model.compile(loss='mean_squared_error', metrics=['mae'], optimizer=Adam(lr=1e-5, decay=1e-8))

# Patient early stopping
#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)

# Fit the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2500, batch_size=1, verbose=2)
#, callbacks=[es]
# Calculate predictions
PredTestSet = model.predict(X_train)

PredValSet = model.predict(X_test)

# Save predictions
numpy.savetxt("trainresults.csv", PredTestSet, delimiter=",")
numpy.savetxt("valresults.csv", PredValSet, delimiter=",")

# Predict
pred = model.predict(X_test)

# Measure RMSE error.  RMSE is common for regression.
score = np.sqrt(metrics.mean_squared_error(pred,y_test))
print(f"Before save score (RMSE): {score}")

#%%
# model.save(os.path.join(save_path,"FINAL_Iss_050721_2.h5"))
# model.save_weights("FINAL_Iss_050721_2_weights.h5")


#%%
# Plot training history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
#%%
# Plot actual vs prediction for training set
TestResults = numpy.genfromtxt("trainresults.csv", delimiter=",")
plt.plot(y_train,TestResults,'ro')
plt.title('Training Set')
plt.xlabel('Actual')
plt.ylabel('Predicted')

# Compute R-Square value for training set
TestR2Value = r2_score(y_train,TestResults)
print("Training Set R-Square=", TestR2Value)
#%%

# Plot actual vs prediction for validation set
ValResults = numpy.genfromtxt("valresults.csv", delimiter=",")
plt.plot(y_test,ValResults,'ro')
plt.title('Validation Set')
plt.xlabel('Actual')
plt.ylabel('Predicted')

# Compute R-Square value for validation set
ValR2Value = r2_score(y_test,ValResults)
print("Validation Set R-Square=",ValR2Value)
#%%
#PREDICTING A DATASET
from tensorflow.keras.models import load_model
model_weights = model.load_weights("FINAL_Iss_170721_2_weights.h5")

#%%
weight = model.get_weights()
# np.savetxt('weight.csv' , weight , fmt='%s', delimiter=',')

#%%
from tensorflow.keras.models import load_model
model_pred = load_model(os.path.join(save_path,"FINAL_Iss_170721_2.h5"))
pred = model_pred.predict(X_test)
# Measure RMSE error.  RMSE is common for regression.
score = np.sqrt(metrics.mean_squared_error(pred,y_test))
print(f"After load score (RMSE): {score}")

#%%
### SENSITIVITY ANALYSIS ###
from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
import numpy as np
#%%
#Defining model inputs
problem = {
    'num_vars': 4,
    'names': ['LL','PL','PI','LS'],
    'bounds': [[15, 130],
               [5, 60],
               [0, 100],
               [0, 40]]
}
#%%
#Generate samples
param_values = saltelli.sample(problem, 131072)
#%%
#Run model
read_data_pred = param_values

read_data = pd.read_csv('geodataset.csv', skiprows=range(0,1), delimiter=',', index_col=False,
                   names=['Iss','LL','PL','PI','LS'])

dataset_pred = pd.DataFrame(read_data, columns=['Iss','LL','PL','PI','LS'])

# fit scaler on training data
X_model = dataset_pred.iloc[:,1:5]
norm = StandardScaler().fit(X_model)
# transform training data
X_pred = norm.transform(param_values)

# we call the predict method
predictions = model_pred.predict(X_pred)
predictions = predictions.ravel()
# print the predictions
print(predictions)
# Y = np.zeros([param_values.shape[0]])
# Y = Ishigami.evaluate(param_values)
#%%
#Perform analysis
Si = sobol.analyze(problem, predictions)
# Si.plot()


# print(Si['S1'])
# print(Si['ST'])

# total_Si, first_Si, second_Si = Si.to_df()

Si.plot()

#%%
print(Si['S1'])
#Here, we see that PL and LS exhibit first-order sensitivities but x3 appears to have no first-order effects.
#%%
print(Si['ST'])
#If the total-order indices are substantially larger than the first-order indices, then there is likely higher-order interactions occurring.
#%%
print("LL-PL:", Si['S2'][0,1])
print("LL-PI:", Si['S2'][0,2])
print("LL-LS:", Si['S2'][0,3])
print("PL-PI:", Si['S2'][1,2])
print("PL-LS:", Si['S2'][1,3])
print("PI-LS:", Si['S2'][2,3])
#We can see there are strong interactions between PI-LS. Some computing error will appear in the sensitivity indices. 
#For example, we observe a negative value for the x2-x3 index. Typically, these computing errors shrink as the number of samples increases.
#%%
total_Si, first_Si, second_Si = Si.to_df()
#%%
# using Sobol, Plots
s = SobolSeq(2)
p = hcat([next!(s) for i = 1:131072]...)'
scatterplot(p[:,1], p[:,2])