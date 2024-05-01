"""
  Title: Load CSV file

  Description:
  This file includes all the essential functions to load a selected CSV file for use in the assignment.
"""

# Using Pandas for CSV reads
import pandas as pd
# File File on operating system
import tkinter as tk
from tkinter import filedialog

# Inisilize Parent window for dialogbox
# Create a tkinter root window
root = tk.Tk()
# Hide the root window
root.withdraw()


def Retrive_data():
    """Retrive Data in chosen CSV File"""
    # Ask user for File
    csv_file_path = filedialog.askopenfilename()
    # Retrive Chosen CSV file
    CSV_data = pd.read_csv(csv_file_path)
    return CSV_data
