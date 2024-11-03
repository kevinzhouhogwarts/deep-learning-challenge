# Deep Learning Challenge

## Overview
The purpose of the analysis is to create and attempt to optimize neural network model to predict the success of a funded project based on several project variables. 

Files:
The first draft of the neural network model was created in AlphabetSoupCharity_Initial and its output was saved to AlphabetSoupCharity.h5.
After optimization in AlphabetSoupCharity_Optimization.ipynb, an optimized model was saved to AlphabetSoupCharity_Optimization.h5.

## Method and Results

### Target variable:
IS_SUCCESSFUL variable from the dataset was selected as the dependent variable. It seems to represent an internal determiantion of the success/failure of a funding target.

### Feature variables:
* APPLICATION_TYPE: Seems to be an internal classifcation
* AFFILIATION: Related to the type/scope of the organization or project
* CLASSIFICATION: Seems to be an internal classification
* USE_CASE: Target use of the funds
* ORGANIZATION: Type of organization overseeing the project
* STATUS: Whether the project is ongoing or not
* INCOME_AMT: A bracket classification of organization income
* SPECIAL_CONSIDERATIONS: The detail of considerations is not revealed 
* ASK_AMT: Amount of project funded request

### Irrelevant variables:
The name of the organization overseeing the project and the organization's EIN number were removed from the dataset.

### Preliminary Model
For the initial draft, a model with three layers of 16, 4, and 1 layer(s) was selected, each using the ReLU activation function. 

### Optimization
During optimization, the following was attempted:
* Increasing the number of neurons per layer
* Changing the activation function in each layer
* Streamline columns from the initial dataset

Also, to save time, an EarlyStopping callback was implemented to stop the trial if the accuracy did not change after 4 epochs.

### Neurons per Layer
Doubling the number of neurons in the input layer and the hidden layer, from 16 and 4 to 32 and 8, did not produce a significant difference in accuracy. Even decreasing the number of neurons to 4 and 2 did not significantly decrease the accuracy. This suggests that for this dataset, the number of neurons does not significantly impact the model's performance.

Models attempted (first and second layers):
nn.add(tf.keras.layers.Dense(units=32, activation="relu", input_dim=42))
nn.add(tf.keras.layers.Dense(units=8, activation="relu"))

nn.add(tf.keras.layers.Dense(units=8, activation="relu", input_dim=42))
nn.add(tf.keras.layers.Dense(units=2, activation="relu"))

### Activation Function
Because the output is binary, the activation function of the output layer was changed from ReLU to Sigmoid. This also did not have a significant impact on the accuracy.

Output layer:
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

Then, replacing the first and second hidden layers' ReLU function with LeakyReLU and ELU was attempted. Neither produced a significant difference.

I would have liked to check the activation outputs of each layer in order to check for dead neurons, but was unable to successfully do that. However, since LeakyReLU and ELU did not make a difference, it's possible dead neurons was not an issue.

### Streamline Columns Using PCA
Next, PCA was applied to the scaled/quantized dataset in order to identify several nonlinearly-related components to represent the original data. After viewing the elbow curve of cumulative explained variance versus the number of components, a principal component number of 34 was selected. At the same time, the input dimensions of the model had to be changed to match.

![Alt text]("Resources/pca_elbow_curve.png")

However, after refitting the model using the PCA training subset of 34 components, the accuracy was not significantly impacted.

At this point, a diverse array of options had been attempted, with only a small decrease in loss and no increase in accuracy, so I used hyperparameter optimization to see if automatic tuning could produce any effect.

### Hyperparameter Tuning
Using the PCA columns, the model for hyperparameter optimization was configured to vary the number of neurons per layer and the activation functions of the two hidden layers between ReLU and sigmoid. However, no significant increase in accuracy nor decrease in loss was observed. 

## Conclusions
The target accuracy of 75% was not achieved even after hyperparameter optimization. However, the neural network model did perform at a decent level. Therefore, as a next step I would use an ensemble approach with several models that are common for binary classifcation problems, including random forest and Support Vector Machine.
