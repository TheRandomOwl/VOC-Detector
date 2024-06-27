import voc
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Generate sample data
num_samples = 1000
num_features = 1004
A = [s.y for s in voc.proplarge.signals]
B = [s.y for s in voc.phenlarge.signals]
X = A + B  # Replace with your actual data
y = [0 for i in range(len(A))] + [1 for i in range(len(B))]  # Replace with your actual labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a multi-layer perceptron (MLP) classifier
model = MLPClassifier(hidden_layer_sizes=(16, 8), activation='relu', random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model on the test set
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

#model.predict([s.y for s in voc.r.signals])