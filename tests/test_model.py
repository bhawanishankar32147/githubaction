import joblib
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def test_accuracy():
    iris = load_iris()
    _, X_test, _, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42)
    # Load the saved model
    clf = joblib.load("app/iris.json")
    acc = accuracy_score(y_test, clf.predict(X_test))
    assert acc > 0.85, f"Accuracy too low: {acc}"
