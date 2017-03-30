# Javier Vazquez
# Todo: Add your names
# Adult Classifier
# Date: March 29 ,2017
# Description: Predict whether income exceeds $50K/yr based on census data.
# Also known as "Census Income" dataset.
# Source: http://archive.ics.uci.edu/ml/datasets/Adult


import os
import linearClassifierClass
import pandas as pd
import numpy as np


def main():
    lc = linearClassifierClass.linearClassifier()


if __name__ == '__main__':
    print(os.path.basename(__file__))
    main()