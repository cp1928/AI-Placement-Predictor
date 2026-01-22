from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on test data.

    Parameters:
    model : Trained model
    X_test : Test features
    y_test : Test labels

    Returns:
    accuracy : float
    report : str
    confusion_mat : array
    """

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)

    return accuracy, report, confusion_mat
