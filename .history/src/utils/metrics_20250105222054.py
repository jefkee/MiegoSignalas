from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate_model(y_true, y_pred):
    """Calculate model performance metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    return metrics 