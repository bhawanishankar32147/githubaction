from iris_model.train import train_model

def test_accuracy_threshold():
    accuracy = train_model()
    assert accuracy >= 0.7, "Model accuracy is too low!"
