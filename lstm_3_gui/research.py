from jesse.strategies import Strategy, cached
import jesse.indicators as ta
from jesse import utils
import pandas as pd
from jesse import research
import joblib
import numpy as np
import jesse.helpers as jh
# from jesse.research import backtest 
import json
from indicators import feature_indicators

# import torch

# from .lstm_model import BiLSTMClassifierWithAttention




def load_indicator_names(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

indicator_names = load_indicator_names('/Users/duncanmaclennan/Documents/Jesse/trade/my-bot/models/SOL_feature_names_0.json')

print(indicator_names)