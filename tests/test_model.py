from app.model import train_and_save_model

def test_accuracy():
    acc = train_and_save_model()
    assert acc > 0.85, f"Accuracy too low: {acc}"
