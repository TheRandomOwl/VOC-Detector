from voc import *
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

# Generate sample data

multimodal_true = run('multimodal_true',flip = True)
multimodal_true.zero()
multimodal_false = run('multimodal_false',flip = True)
multimodal_false.zero()

num_samples = 1000
num_features = 421
A = [s.y for s in multimodal_false.signals]
B = [s.y for s in multimodal_true.signals]
X = A + B  # Replace with your actual data
y = [0 for i in range(len(A))] + [1 for i in range(len(B))]  # Replace with your actual labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a multi-layer perceptron (MLP) classifier
model = MLPClassifier(hidden_layer_sizes=(100,421,100), activation='relu', random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model on the test set
accuracy = model.score(X_test, y_test)
accuracy_false = model.score(A, [0 for i in range(len(A))])
accuracy_true = model.score(B, [1 for i in range(len(B))])
print("Accuracy:", accuracy)
print('Accuracy on false:', accuracy_false)
print('Accuracy on true:', accuracy_true)

#model.predict([s.y for s in voc.r.signals])

with open('multi_nnet.p','wb') as f:
	pickle.dump(model,f)