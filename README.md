# deep-learning-challenge

##
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
First, because the output is binary, the activation function of the output layer was changed from ReLU to Sigmoid. This also did not have a significant impact on the accuracy.

Output layer:
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

Then, replacing the input layer's ReLU function with LeakyReLU and ELU was attempted.
