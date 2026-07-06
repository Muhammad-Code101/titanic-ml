from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

df = pd.read_csv("titanic.csv")

df["Age"] = df["Age"].fillna(df["Age"].median())
df = df.drop("Cabin", axis=1)
df = df.dropna()

df["Sex"] = df["Sex"].map({"male":0, "female":1})

X = df[["Age", "Sex", "Pclass", "Fare"]]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print(f"Accuracy: {accuracy * 100:.1f}%")

#

cm = confusion_matrix(y_test, prediction)
print("Confusion matrix:")
print(cm)

#

model2 = RandomForestClassifier(n_estimators=100, random_state=42)
model2.fit(X_train, y_train)
prediction2 = model2.predict(X_test)
accuracy2 = accuracy_score(y_test, prediction2)
print(f"Random Forest Classifier: {accuracy2 * 100:.1f}%")

#

my_data = pd.DataFrame([[20, 1, 1, 100]], columns=["Age", "Sex", "Pclass", "Fare"])
prediction3 = model2.predict(my_data)
probability = model2.predict_proba(my_data)
print(f"Survived: {'yes' if prediction[0] == 1 else 'No'}")
print(f"Survival probability: {probability[0][1] * 100:.1f}%")