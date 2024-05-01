"""
  Student: Dewald Oosthuizen
  Number: 38336529
  Objective: Design, Implement, and Evaluate a Neural Network Classifier for Accurately Recognizing Handwritten Digits from the MNIST Dataset
"""


# Used to retrive CSV file data
import Read_CSV
# Used to Handel CSV file data
import pandas as pd
# Visualize data
import Data_Visualization as dv
# Neural Network
import Neural_Network_Model as NNM
# Preporcessing
import Processing as pre


def main():
    # Retrieve Training Data
    training_CSV_data = Read_CSV.Retrive_data()

    # First Three Rows of data
    dv.visualize_first_three_rows(training_CSV_data)
    # Labels of the Data
    dv.visualize_labels(training_CSV_data)
    # Data Shape
    dv.visualize_data_shape(training_CSV_data, "Data")
    # Labels Shape
    dv.visualize_data_shape(training_CSV_data['label'], "Label")
    # Show A Image from the data with default 0
    dv.visualize_number(training_CSV_data, 10)

    """Preprocess data for use in Neural Network"""
    # Get all the Files
    test_CSV_data = Read_CSV.Retrive_data()  # Get test dataset
    # Get data and labels of CSV
    train_data_X, train_data_Y = pre.split_data_and_label(training_CSV_data)
    test_data_X, test_data_Y = pre.split_data_and_label(test_CSV_data)
    # Scale CSV data to be between 0 and 1 (More Numerically Stabel)
    train_data_X = pre.normalize_pixel_values(train_data_X)
    test_data_X = pre.normalize_pixel_values(test_data_X)
    # Create Validation set from train data
    train_data_X, train_validation_X, train_data_Y, train_validation_Y = pre.split_train_validation(
        train_data_X, train_data_Y, 0.2, 42)

    """---------------------------------------------------Create the model and Train it on data using default values-------------------------------------------------------------------------------------"""
    print("\n" + "-" * 20 + "Base Model" + "-" * 20)
    print("The forthcoming model will be constructed using default variables and thereafter assessed against optimised models.")
    # Create Neural Network model using MLPClassifier
    Model = NNM.create_mlp_classifier()
    # Print Paramaters
    dv.model_parameters(Model)
    # Training model
    Trained_Model = NNM.train_model(Model, train_data_X, train_data_Y)
    # Predict Values and Validate
    prediction = NNM.predict_values(Trained_Model, train_validation_X)
    # Calculate all performance metrics
    metrics = {
        'accuracy': NNM.accuracy_score(train_validation_Y, prediction) * 100,
        'precision': NNM.calculate_precision(train_validation_Y, prediction, average='macro') * 100,
        'recall': NNM.calculate_recall(train_validation_Y, prediction, average='macro') * 100,
        'f1_score': NNM.calculate_f1_score(train_validation_Y, prediction, average='macro') * 100,
    }
    # Display all performance metrics
    dv.display_Validation_model_performance(
        "Validating Base Model Data Prediction", metrics)
    # Display Confution matrix of perdicted data
    confusion_matrix = dv.create_confusion_matrix(
        train_validation_Y, prediction, display_matrix=False)
    # Display Confution matrix in terminal
    dv.display_confusion_matrix(confusion_matrix)

    """-------------------------------------------------------------------Using Base model to predict using test data---------------------------------------------------------------------------------"""
    print("\n" + "-" * 20 +
          "Prediction utilising test data in the base model" + "-" * 20)
    print("In the upcoming phase, the verified base model will be used to predict labels using the test data.")
    # Predict Values using test image data
    prediction = NNM.predict_values(Trained_Model, test_data_X)
    # Calculate all performance metrics
    metrics_Base = {
        'accuracy': NNM.accuracy_score(test_data_Y, prediction) * 100,
        'precision': NNM.calculate_precision(test_data_Y, prediction, average='macro') * 100,
        'recall': NNM.calculate_recall(test_data_Y, prediction, average='macro') * 100,
        'f1_score': NNM.calculate_f1_score(test_data_Y, prediction, average='macro') * 100,
        'report': NNM.generate_classification_report(test_data_Y, prediction, target_names=[str(i) for i in range(10)])
    }
    # Display all performance metrics
    dv.display_model_performance(
        "Base Model Test Data Prediction", metrics_Base)
    # Display Confution matrix of perdicted data
    confusion_matrix = dv.create_confusion_matrix(
        test_data_Y, prediction, display_matrix=False)
    # Display Confution matrix in terminal
    dv.display_confusion_matrix(confusion_matrix)

    """--------------------------------------------------------The following model will be made using optimized parameters-------------------------------------------------------------------------------"""
    print("\n" + "-" * 20 + "Optimized Model Using Training Dataset" + "-" * 20)
    print(f"Models will be created and tested to determin the best model parameters using the training dataset. After the best model will predict using the test data and evaluated\n")
    # Create base model
    Model = NNM.create_mlp_classifier()
    # Create List of parameters
    param_grid = {
        'hidden_layer_sizes': [(28, 28, 28, 28), (128,), (200, 200), (300,), (300, 200, 128)],
        'max_iter': [200, 300],
    }

    # Test Each Combination and return best model
    optimal_model, results_matrix = NNM.find_optimal_model(
        Model, train_data_X, train_data_Y, param_grid, 4, "accuracy")

    # Analyse the best model
    # Print Paramaters
    dv.model_parameters(optimal_model)
    # Plot Accuracy of model
    dv.plot_grid_search_results(results_matrix, title='MLP Hyperparameter Tuning',
                                xlabel='Max Iterations', ylabel='Hidden Layer Sizes')
    """--------------------------------------------------------Using Optimal Model on Test Dataset-------------------------------------------------------------------------------"""
    print("\n" + "-" * 20 + "Optimized Model Using Test Dataset" + "-" * 20)
    # Predict Values using test image data
    prediction = NNM.predict_values(optimal_model, test_data_X)
    # Calculate all performance metrics
    metrics_Optimal = {
        'accuracy': NNM.accuracy_score(test_data_Y, prediction) * 100,
        'precision': NNM.calculate_precision(test_data_Y, prediction, average='macro') * 100,
        'recall': NNM.calculate_recall(test_data_Y, prediction, average='macro') * 100,
        'f1_score': NNM.calculate_f1_score(test_data_Y, prediction, average='macro') * 100,
        'report': NNM.generate_classification_report(test_data_Y, prediction, target_names=[str(i) for i in range(10)])
    }
    # Display all performance metrics
    dv.display_model_performance(
        "Optimal Model Test Data Prediction", metrics_Optimal)
    # Display Confution matrix of perdicted data
    confusion_matrix = dv.create_confusion_matrix(
        test_data_Y, prediction, display_matrix=False)
    # Display Confution matrix in terminal
    dv.display_confusion_matrix(confusion_matrix)

    """--------------------------------------------------------Compare Optimal With Base-------------------------------------------------------------------------------"""
    # Names of metrics to compare
    metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
    # Compare Values using a histogram
    dv.compare_model_performance_change(
        metrics_Base, metrics_Optimal, metric_names)


if __name__ == "__main__":
    main()
