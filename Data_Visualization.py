"""
  Title: Data Visualization for the chosen CSV file/files

  Description:
  This File containes all the functions for basic visualizaiton of the chosen dataset
"""

# For Mathimatical Operations
import numpy as np
# Handel CSV file
import pandas as pd
# Data Visualization
import matplotlib.pyplot as plt
# Used to Handel CSV file data
import pandas as pd
# Used to draw attractive and informative statistical graphics
import seaborn as sns
# Performance metrics in model evaluation
from sklearn.metrics import confusion_matrix


def visualize_data_shape(data, name):
    """Prints the shape of the CSV file data."""
    print(f"\n" + "-" * 20 + "CSV File " + f"{name}" + " Shape" + "-" * 20)
    print(data.shape)


def visualize_first_three_rows(data):
    """Prints the first three rows of the CSV file data."""
    print("\n" + "-" * 20 + "CSV File Data" + "-" * 20)
    print(data.head(3))


def visualize_labels(data):
    """Prints the labels from the CSV file data."""
    print("\n" + "-" * 20 + "CSV File Labels" + "-" * 20)
    data_labels = data['label']
    print(data_labels.head(3))


def visualize_number(data, index=0):
    """Display or plot a number in the data"""
    # Retrieve List of Labels
    data_labels = data['label']

    print("\n" + "-" * 20 +
          f"Plot of {index} and Lable {data_labels[index]}" + "-" * 20)
    # Plot opens in separate window not cocle
    print("Ready for View in matplot Window")

    # Create Plot size
    plt.figure(figsize=(7, 7))  # 7 x 7
    # Drop labels for ploting
    pixel_data = data.drop('label', axis=1)
    # Convert Number data into 2D grid data
    grid_data = pixel_data.iloc[index].values.reshape(28, 28)
    # Show Image From Data at Index
    plt.imshow(grid_data, interpolation="none", cmap="gray")
    # Change Plot Title
    plt.title(
        f"Image at Index {index} with Label {data_labels[index]}", fontsize=20, color="red")
    # Show Image
    plt.show()


def model_parameters(model):
    """Prints Parameters of the model"""
    print("\n" + "-" * 20 + "Neural Network Model Parameters" + "-" * 20)
    print("Hidden Layers:", model.hidden_layer_sizes)
    print("Activation Function:", model.activation)
    print("Optimizer (Solver):", model.solver)
    print("Max Iterations:", model.max_iter)


def create_confusion_matrix(true_labels, predicted_labels, display_matrix=False):
    """
    Generates a confusion matrix to evaluate the accuracy of a classification.

    Args:
        true_labels (array-like): True labels of the data.
        predicted_labels (array-like): Predicted labels by the model.
        display_matrix (bool): If True, display the matrix as a heatmap.

    Returns:
        array: The confusion matrix as a 2D array.
    """
    # Calculate the confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Optionally display the confusion matrix
    if display_matrix:
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()

    return cm


def display_confusion_matrix(confusion_matrix):
    """Displays a confusion matrix in the terminal as a neatly formatted table."""
    # Determine the size of the matrix
    size = len(confusion_matrix)

    # Print the column headers
    print("\n" + "-" * 20 + "Confusion Matrix" + "-" * 20)
    print(" " * 7, end="")
    for i in range(size):
        print(f"{i:<6}", end="")
    print()  # New line after headers

    # Print the matrix rows with row headers
    for index, row in enumerate(confusion_matrix):
        print(f"Label {index} ", end="")
        for value in row:
            print(f"{value:<6}", end="")
        print()  # New line after each row


def display_Validation_model_performance(model_name, metrics):
    """
    Displays the performance metrics of the Validated model (Does not need detailed report).

    Args:
        model_name (str): Name of the model evaluated (Optimal, base, validation base model, etc.)
        metrics (dict): A dictionary containing performance metrics of the model.
                        Expected keys: 'accuracy', 'precision', 'recall', 'f1_score'.
    """
    print("\n" + "-" * 20 + f"{model_name}: Performance Summary" + "-" * 20)
    print(f"Accuracy: {metrics['accuracy']:.4f}%")
    print(f"Precision: {metrics['precision']:.4f}%")
    print(f"Recall: {metrics['recall']:.4f}%")
    print(f"F1 Score: {metrics['f1_score']:.4f}%")


def display_model_performance(model_name, metrics):
    """
    Displays the performance metrics of the model.

    Args:
        model_name (str): Name of the model evaluated (Optimal, base, validation base model, etc.)
        metrics (dict): A dictionary containing performance metrics of the model.
                        Expected keys: 'accuracy', 'precision', 'recall', 'f1_score', 'report'.
    """
    print("\n" + "-" * 20 + f"{model_name}: Performance Summary" + "-" * 20)
    print(f"Accuracy: {metrics['accuracy']:.4f}%")
    print(f"Precision: {metrics['precision']:.4f}%")
    print(f"Recall: {metrics['recall']:.4f}%")
    print(f"F1 Score: {metrics['f1_score']:.4f}%\n")

    print("Detailed Classification Report:")
    print(metrics['report'])


def compare_model_performance_change(metrics_base, metrics_optimal, metric_names):
    """
    Compares performance metrics of two models by showing the changes between them using a bar chart.

    Args:
        metrics_base (dict): Metrics dictionary for the base model.
        metrics_optimal (dict): Metrics dictionary for the optimized model.
        metric_names (list of str): List of metric names to compare (e.g., 'accuracy', 'precision', 'recall', 'f1_score').

    Displays:
        Bar chart depicting the difference in specified metrics between the two models.
    """
    n_metrics = len(metric_names)
    index = np.arange(n_metrics)
    bar_width = 0.5

    # Calculate differences
    differences = [metrics_optimal[metric] - metrics_base[metric]
                   for metric in metric_names]

    plt.figure(figsize=(10, 5))
    rects = plt.bar(index, differences, bar_width, color='green',
                    label='Difference (Optimal - Base)')

    plt.xlabel('Metrics')
    plt.ylabel('Difference in Scores')
    plt.title('Change in Model Metrics (Optimal vs Base)')
    plt.xticks(index, metric_names)
    plt.legend()

    # Adding value labels on top of each bar
    for rect in rects:
        height = rect.get_height()
        plt.annotate(f'{height:.4f}',
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def plot_grid_search_results(matrix, title='Grid Search Results', xlabel='Max Iterations', ylabel='Hidden Layer Sizes'):
    """
    Plots a heatmap of the grid search results.

    Args:
        matrix (DataFrame): The results matrix from GridSearchCV containing mean test scores.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(matrix, annot=True, cmap='viridis', fmt=".3f", linewidths=.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
