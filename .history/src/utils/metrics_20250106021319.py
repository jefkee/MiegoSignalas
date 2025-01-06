from sklearn.metrics import accuracy_score

def evaluate_model(y_true, y_pred):
    """Calculate model performance metrics"""
    return {
        'accuracy': accuracy_score(y_true, y_pred)
    } 