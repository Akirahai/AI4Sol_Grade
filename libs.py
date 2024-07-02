import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import seaborn as sns

from datasets import load_dataset, Dataset
from datasets import DatasetDict
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding


import evaluate
import argparse
import os
from tqdm import tqdm
from tabulate import tabulate

import datetime

accuracy = evaluate.load("accuracy")


id2label = {0: '3', 1: '4', 2: '5', 3: '6'}
label2id = {'3': 0, '4': 1, '5': 2, '6': 3}


import numpy as np


# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     predictions = np.argmax(predictions, axis=1)
#     return accuracy.compute(predictions=predictions, references=labels) 

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    # Debugging: Print the shapes of predictions and labels
    print(f"Predictions shape: {predictions.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Ensure predictions is a numpy array
    predictions = np.array(predictions)

    # Check for inhomogeneous shapes and handle accordingly
    try:
        # Assuming your predictions are logits or probabilities and you want the class with the highest score
        if predictions.ndim > 2:
            # If predictions have more than 2 dimensions, flatten them appropriately
            predictions = predictions.reshape(predictions.shape[0], -1)
        predictions = np.argmax(predictions, axis=1)
    except ValueError as e:
        print(f"ValueError: {e}")
        print("Predictions array has inconsistent shapes. Debugging...")
        print(predictions)
        raise e

    return accuracy.compute(predictions=predictions, references=labels)
