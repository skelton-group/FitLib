# FitLib/Utilities.py


# ----------------
# Module Docstring
# ----------------

""" Miscellaneous utilities. """


# -------
# Imports
# -------

import csv

import numpy as np


# -------------
# I/O Functions
# -------------

def ReadCSV(file_path, num_header_rows = 0):
    """ Read a comma-separated values (CSV) file, skipping the first num_header_rows rows. """

    data_cols = None
    
    with open(file_path, 'r') as input_reader:
        input_reader_csv = csv.reader(input_reader)
        
        # Skip header rows if required.
        
        for _ in range(0, num_header_rows):
            next(input_reader_csv)
        
        for row in input_reader_csv:
            items = [float(item) for item in row]
            
            if data_cols is None:
                data_cols = [
                    [item] for item in items
                    ]
            else:
                assert len(items) == len(data_cols)
                
                for i, col in enumerate(data_cols):
                    col.append(items[i])
    
    assert data_cols is not None
    
    return [
        np.array(col, dtype = np.float64)
            for col in data_cols
        ]

def ReadText(file_path):
    """
    Read a plain-text data file.
    
    Notes:
        '#' characters are treated as comments and text following them is stripped before parsing.
        Blank and comment lines are skipped.
    """
    
    data_cols = None
    
    with open(file_path, 'r') as input_reader:
        for line in input_reader:
            line = line.strip()
            
            # Strip comments prefixed by the # character.
            
            if '#' in line:
                line = line[:line.find('#')]
            
            if line != "":
                items = [
                    float(item) for item in line.split()
                    ]
                
                if data_cols is None:
                    data_cols = [
                        [item] for item in items
                        ]
                else:
                    assert len(items) == len(data_cols)
                    
                    for i, col in enumerate(data_cols):
                        col.append(items[i])
    
    assert data_cols is not None
    
    return [
        np.array(col, dtype = np.float64)
            for col in data_cols
        ]
