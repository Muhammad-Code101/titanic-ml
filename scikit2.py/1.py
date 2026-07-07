import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("titanic.csv")
df["Age"] = df["Age"].fillna(df["Age"].median())
df = df.drop("Cabin", axis=1)
df.dropna()

df["Sex"] = df["Sex"].map({"male":0, "female":1})

X = df[["Age", "Sex", "Pclass", "Fare"]]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
model.fit(X_train, y_train)

prediction = model.predict(X_test)
accuracy = accuracy_score(y_test, prediction)

print(f"Neural Network Accuracy: {accuracy * 100:.1f}%")
print(f"Confusion matrix:\n {confusion_matrix(y_test, prediction)}")