
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


features, labels = make_moons(n_samples=500, noise=0.2, random_state=42)

train_features, test_features, train_labels, test_labels = train_test_split(
    features,
    labels,
    test_size=0.2,
    random_state=42,
)

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)


ann = MLPClassifier(
    hidden_layer_sizes=(10,),
    max_iter=2000,
    random_state=42,
)


ann.fit(train_features, train_labels)


predicted_labels = ann.predict(test_features)


accuracy = accuracy_score(test_labels, predicted_labels)
print("Accuracy:", round(accuracy * 100, 2), "%")
print("\nConfusion matrix:")
print(confusion_matrix(test_labels, predicted_labels))
print("\nClassification report:")
print(classification_report(test_labels, predicted_labels))
