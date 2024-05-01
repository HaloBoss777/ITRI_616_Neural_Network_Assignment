"""
  Title: Processing of a CSV file

  Description:
  This file contains all the necessary functions for pre and post-processing of a dataset.
"""
# Used to Handel CSV file data
import pandas as pd
# Used to Split the data into test and validation
from sklearn.model_selection import train_test_split
# Used to Scale data
from sklearn.preprocessing import StandardScaler


def split_data_and_label(CSV_data):
    """Splits CSV data into Image data and the label"""
    if 'label' in CSV_data.columns:
        X = CSV_data.drop('label', axis=1)  # All data except the label column
        y = CSV_data['label']  # Just the label column
    else:
        raise ValueError("Label column not found in the dataset")
    return X, y


def normalize_pixel_values(data):
    """
    Normalizes pixel values from the range of 0 to 255 to a new range of 0 to 1.
    This enhances the numerical stability of the data and is especially useful for image data
    """
    normalized_data = data / 255.0  # Make Sure every value is a float
    return normalized_data


def split_train_validation(train_data_X, train_data_Y, test_size=0.2, random_state=42):
    """
    Splits the training data into train and validation sets.

    train_data_X: Features(Image Data) of the training data.
    train_data_Y: Labels of the training data.
    test_size: Proportion of the dataset to include in the validation split (20% default).
    random_state: Controls the shuffling applied to the data before applying the split (42 to enable reproducibility).
    return: train_X, val_X, train_Y, val_Y
    """
    train_X, val_X, train_Y, val_Y = train_test_split(
        train_data_X, train_data_Y, test_size=test_size, random_state=random_state
    )
    return train_X, val_X, train_Y, val_Y
