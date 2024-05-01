"""
  Title: Neural Network Model

  Description:
  The selected model for this assignment is MLPClassifier, which is a neural network provided by the sklearn library.
  This model will undergo training using the MNIST dataset to accurately classify numbers based on the input data.
"""

# Use scikit learn
from sklearn.neural_network import MLPClassifier
# Performance metrics in model evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
# Used to preform gridsearch
from sklearn.model_selection import GridSearchCV
# Used to map results
import pandas as pd


def create_mlp_classifier(hidden_layer_sizes=(16,), activation='relu', solver='adam', max_iter=200, random_state=1):
    """
    Creates and returns an MLPClassifier model configured with the specified parameters.

    hidden_layer_sizes: Tuple representing the number and size of the hidden layers
    activation: Type of activation function ('relu', 'logistic', 'tanh')
    solver: Type of solver to use for weight optimization ('adam', 'sgd', 'lbfgs')
    max_iter: Maximum number of iterations during training
    random_state: Seed for the random number generator for reproducibility
    """

    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state
    )

    # Return Created Model
    return model


def train_model(model, train_data_X, train_data_Y):
    """
    Trains a given MLPClassifier model using the provided training data.

    Args:
        model (MLPClassifier): The MLPClassifier instance to be trained.
        train_data_X (array-like, sparse matrix): Feature dataset used for training the model.
        train_data_Y (array-like): Labels or target values corresponding to the training data.

    Returns:
        MLPClassifier: The trained model.
    """
    # Fit the model to the training data
    print("\n" + "-" * 20 + "Training Model" + "-" * 20)
    model.fit(train_data_X, train_data_Y)
    print("Training Concluded")
    # Return Model
    return model


def predict_values(model, data_X):
    """
    Predicts output values using a trained model on the provided validation dataset.

    Args:
        model (MLPClassifier): The trained model.
        validation_data_X (array-like, sparse matrix): Feature dataset used for making predictions.

    Returns:
        array: Predicted values for the dataset.
    """
    # Use the model to predict the outputs for the validation dataset
    print("\n" + "-" * 20 + "Predicting Labels" + "-" * 20)
    predictions = model.predict(data_X)
    print("Prediction Concluded")
    return predictions


def model_accuracy(true_Y, predictions):
    """
    Evaluates the models accuracy

    Args:
        train_validation_Y (array-like): Labels or target values corresponding to the training data.
        predictions (array-like): Predicted values for the dataset

    Returns:
        Float: The accuracy score of the model
    """
    return accuracy_score(true_Y, predictions)


def calculate_precision(true_Y, predictions, average='macro'):
    """
    Calculates the precision for the given true and predicted labels.

    Args:
        true_Y (array-like): True labels of the data.
        predictions (array-like): Predicted labels by the model.
        average (str, optional): This parameter is required for multiclass/multilabel targets.
            'binary': Only report results for the class specified by pos_label.
            'micro': Calculate metrics globally by counting the total true positives, false negatives, and false positives.
            'macro': Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
            'weighted': Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).

    Returns:
        float: The precision of the model on the provided data.
    """
    return precision_score(true_Y, predictions, average=average)


def calculate_recall(true_Y, predictions, average='macro'):
    """
    Calculates the recall for the given true and predicted labels.

    Args:
        true_Y (array-like): True labels of the data.
        predictions (array-like): Predicted labels by the model.
        average (str): Specifies the averaging method when data is multi-class.
                       'macro' (default), 'micro', 'weighted', or None.

    Returns:
        float: The recall of the model on the provided data.
    """
    return recall_score(true_Y, predictions, average=average)


def calculate_f1_score(true_Y, predictions, average='macro'):
    """
    Calculates the F1 score for the given true and predicted labels.

    Args:
        true_Y (array-like): True labels of the data.
        predictions (array-like): Predicted labels by the model.
        average (str): Specifies the averaging method when data is multi-class.
                       'macro' (default), 'micro', 'weighted', or None.

    Returns:
        float: The F1 score of the model on the provided data.
    """
    return f1_score(true_Y, predictions, average=average)


def generate_classification_report(true_Y, predictions, target_names=None):
    """
    Generates a detailed classification report for the given true and predicted labels.

    Args:
        true_Y (array-like): True labels of the data.
        predictions (array-like): Predicted labels by the model.
        target_names (list of str, optional): List of strings representing the names of the classes. 
            Each element must correspond to the label at the corresponding index.

    Returns:
        str: Text summary of the precision, recall, F1 score for each class.
    """
    return classification_report(true_Y, predictions, target_names=target_names)


def find_optimal_model(model, train_data_X, train_data_Y, param_grid, cv=5, scoring='accuracy'):
    """
    Performs grid search to find the optimal model configuration based on the specified parameter grid.

    Args:
        model (estimator): A model instance from scikit-learn.
        train_data_X (array-like): Training features.
        train_data_Y (array-like): Training target variable.
        param_grid (dict): Dictionary with parameters names (`str`) as keys and lists of parameter settings to try as values.
        cv (int, optional): Number of cross-validation folds. Defaults to 5.
        scoring (str, optional): A single string to evaluate the predictions on the test set. For example, 'accuracy'. Defaults to 'accuracy'.

    Returns:
        sklearn.base.BaseEstimator: The best estimator found by the grid search.
    """
    # Create the GridSearchCV object
    grid_search = GridSearchCV(
        estimator=model, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=7, verbose=2)

    # Fit GridSearchCV
    grid_search.fit(train_data_X, train_data_Y)

    # Best estimator found by grid search
    best_model = grid_search.best_estimator_

    # Converts the cv_results_ attribute, which is a dictionary containing detailed results of the grid search, into a pandas DataFrame
    results_df = pd.DataFrame(grid_search.cv_results_)

    """creates a list of column names to extract from the results_df DataFrame.
    It includes the mean_test_score column (which contains the average score across all cross-validation folds for each parameter combination)
    and dynamically adds columns for each parameter in the grid (e.g., param_hidden_layer_sizes, param_max_iter)
    """
    focus_cols = ['mean_test_score'] + \
        ['param_' + key for key in param_grid.keys()]

    # Filters results_df to include only the columns specified in focus_cols
    results_df = results_df[focus_cols]

    # Create Pivot Table for Hiddel layers and the iterations evaluated on the mean accuracy
    if len(param_grid) > 1:
        results_matrix = results_df.pivot_table(
            index='param_hidden_layer_sizes', columns='param_max_iter', values='mean_test_score')
    else:
        param = 'param_' + list(param_grid.keys())[0]
        results_matrix = results_df.set_index(param).rename(
            columns={'mean_test_score': 'Accuracy'})

    return best_model, results_matrix
