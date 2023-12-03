import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_curve
np.set_printoptions(threshold=10000, suppress = True)
import matplotlib.pyplot as plt
import os
import cv2
import pickle