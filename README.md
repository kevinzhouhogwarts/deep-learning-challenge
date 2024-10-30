# Deep Learning Challenge

## Introduction
The first draft of the neural network model was created in AlphabetSoupCharity_Initial and its output was saved to AlphabetSoupCharity.h5.
After optimization in AlphabetSoupCharity_Optimization.ipynb, an optimized model was saved to AlphabetSoupCharity_Optimization.h5.


## Method
During optimization, the following was attempted:
* Increasing the number of neurons per layer
* Changing the activation function in each layer
* Dropping columns from the initial dataset

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

### Drop Columns Using PCA
Next, PCA was applied to the scaled/quantized dataset in order to identify several nonlinearly-related components to represent the original data. After viewing the elbow curve of cumulative explained variance versus the number of components, a principal component number of 34 was selected. At the same time, the input dimensions of the model had to be changed to match.

![image](https://github.com/user-attachments/assets/62739b8f-cce1-4f51-a803-5b192a2ac3b5)

However, after refitting the model using the PCA training subset of 34 components, the accuracy was not significantly impacted.

At this point, a diverse array of options have been attempted. Perhaps hyperparameter optimization may be attempted.

## Conclusions
