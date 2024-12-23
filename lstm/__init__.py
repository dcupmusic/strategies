from jesse.strategies import Strategy, cached
import jesse.indicators as ta
from jesse import utils
import pandas as pd
from jesse import research
import joblib
import numpy as np
import jesse.helpers as jh
from jesse.research import backtest
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from functions import process_data, tt_split, save_model
import random
import warnings
import gc 

warnings.filterwarnings('ignore', category=RuntimeWarning)


exchange = 'Bybit USDT Perpetual'
symbol = 'SOL-USDT'
train_start_date = '2023-01-01'
train_end_date = '2024-03-01'
back_start_date = train_end_date
back_end_date = '2024-03-25'
timeframe = '1h'
sol_candles = research.get_candles(exchange, symbol, timeframe, train_start_date, train_end_date)
backtest_candles = research.get_candles(exchange, symbol, '1h', train_start_date, back_end_date)



class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        weights = torch.tanh(self.linear(lstm_output))
        weights = F.softmax(weights, dim=1)
        
        context = weights * lstm_output
        context = torch.sum(context, dim=1)
        return context, weights

class BiLSTMClassifierWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_rate):
        super(BiLSTMClassifierWithAttention, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1)        
        self.bn_conv1 = nn.BatchNorm1d(hidden_dim)        
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)        
        self.attention = Attention(hidden_dim * 2)        
        self.dropout = nn.Dropout(dropout_rate)        
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)        
        self.bn_fc1 = nn.BatchNorm1d(hidden_dim)        
        self.fc2 = nn.Linear(hidden_dim, output_dim)        
        self.dropout_fc = nn.Dropout(dropout_rate / 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)        
        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = F.relu(x)        
        x = x.permute(0, 2, 1)        
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)        
        out, _ = self.lstm(x, (h0, c0))        
        context, _ = self.attention(out)        
        out = self.fc1(context)
        out = self.bn_fc1(out)
        out = F.relu(out)
        out = self.dropout_fc(out)
        out = self.fc2(out)
        
        return out





# # # # # # # # # # #
# Feature Engineering
# # # # # # # # # # #
    
feature_indicators = {
    'ACOSC': ta.acosc(sol_candles, sequential=True).osc,
    'AD': ta.ad(sol_candles, sequential=True),
    'ADX': ta.adx(sol_candles, period=14, sequential=True),
    'ALIGATOR': ta.alligator(sol_candles, sequential=True).jaw,
    'ALMA': ta.alma(sol_candles, period=9, sigma=6, sequential=True),
    'AO': ta.ao(sol_candles, sequential=True).osc,
    'APO': ta.apo(sol_candles, fast_period=12, slow_period=26, matype=0, sequential=True),
    'AROONOSC': ta.aroonosc(sol_candles, sequential=True),
    'ATR': ta.atr(sol_candles, 14, sequential=True),
    'BOLLINGERBANDS': ta.bollinger_bands(sol_candles, 20, 2, sequential=True).lowerband,
    'BBWIDTH': ta.bollinger_bands_width(sol_candles, 20, 2, sequential=True),
    'BOP': ta.bop(sol_candles, sequential=True),
    'COPPOCK': ta.cc(sol_candles, sequential=True),
    'CCI': ta.cci(sol_candles, 20, sequential=True),
    'CFO': ta.cfo(sol_candles, 10, sequential=True),
    'CENTREOFGRAVITY': ta.cg(sol_candles, period=10, sequential=True),
    'CKSP': ta.cksp(sol_candles, sequential=True).long,
    'CHANDELIEREXIT': ta.chande(sol_candles,sequential=True),
    'CHOP': ta.chop(sol_candles, sequential=True),
    'CMO': ta.cmo(sol_candles, 14, sequential=True),
    'CORREL': ta.correl(sol_candles, 10, sequential=True),
    'CVI': ta.cvi(sol_candles, 10, sequential=True),
    'CWMA': ta.cwma(sol_candles, 10, sequential=True),
    'DECYCLER': ta.decycler(sol_candles, sequential=True),
    'DEMA': ta.dema(sol_candles, 10, sequential=True),
    'DEVSTOP': ta.devstop(sol_candles, sequential=True),
    'DIPLUS': ta.di(sol_candles, 14, sequential=True).plus,
    'DIMINUS': ta.di(sol_candles, 14, sequential=True).minus,
    'DONCHAIN': ta.donchian(sol_candles, sequential=True).lowerband,
    'DPO': ta.dpo(sol_candles, 20, sequential=True),
    'DTI': ta.dti(sol_candles, sequential=True),
    'EMA': ta.ema(sol_candles, period=10, sequential=True),
    'EMD': ta.emd(sol_candles, 10, sequential=True).lowerband,
    'EMV': ta.emv(sol_candles, sequential=True),
    'EPMA': ta.epma(sol_candles, 10, 20, sequential=True),
    'FISHER': ta.fisher(sol_candles, 10, sequential=True).fisher,
    'FWMA': ta.fwma(sol_candles, 10, sequential=True),        
    'GATOROSC': ta.gatorosc(sol_candles, sequential=True).lower,
    'GAUSSIAN': ta.gauss(sol_candles, 10, sequential=True),
    'HMA': ta.hma(sol_candles, sequential=True),
    'HT_DCPERIOD': ta.ht_dcperiod(sol_candles, sequential=True),
    'HT_DCPHASE': ta.ht_dcphase(sol_candles, sequential=True),
    'HT_PHASOR': ta.ht_phasor(sol_candles, sequential=True).inphase,
    'HT_SINE': ta.ht_sine(sol_candles, sequential=True).sine,
    'HT_TRENDLINE': ta.ht_trendline(sol_candles, sequential=True),
    'HT_TRENDMODE': ta.ht_trendmode(sol_candles, sequential=True),
    'IFT_RSI': ta.ift_rsi(sol_candles, sequential=True),
    'ITREND': ta.itrend(sol_candles, sequential=True).signal,
    'JMA': ta.jma(sol_candles, sequential=True),
    'JSA': ta.jsa(sol_candles, sequential=True),
    'KAMA': ta.kama(sol_candles, sequential=True),
    'KAUFMANSTOP': ta.kaufmanstop(sol_candles, sequential=True),
    'KDJ': ta.kdj(sol_candles, sequential=True).k,
    'KELTNERCHANNEL': ta.keltner(sol_candles, sequential=True).lowerband,
    'KST': ta.kst(sol_candles, sequential=True).signal,
    'KURTOSIS': ta.kurtosis(sol_candles, sequential=True),
    'KVO': ta.kvo(sol_candles, sequential=True),
    'LINEARREG': ta.linearreg(sol_candles, sequential=True),
    'LINEARREG_ANGLE': ta.linearreg_angle(sol_candles, sequential=True),
    'LINEARREG_INTERCEPT': ta.linearreg_intercept(sol_candles, sequential=True),
    'LINEARREG_SLOPE': ta.linearreg_slope(sol_candles, sequential=True),
    'LRSI': ta.lrsi(sol_candles, sequential=True),
    'MAAQ': ta.maaq(sol_candles, sequential=True),
    'MACD': ta.macd(sol_candles, 12, 26, 9, sequential=True).macd,
    'MAMA': ta.mama(sol_candles, 0.5, 0.05, sequential=True).mama,
    'MARKETFI': ta.marketfi(sol_candles, sequential=True),
    'MAASU': ta.mass(sol_candles, sequential=True),
    'MCGINLEYDYNAMIC': ta.mcginley_dynamic(sol_candles, sequential=True),
    'MEAN_AD': ta.mean_ad(sol_candles, sequential=True),
    'MFI': ta.mfi(sol_candles, sequential=True),
    'MIDPOINT': ta.midpoint(sol_candles, sequential=True),
    'MIDPRICE': ta.midprice(sol_candles, sequential=True),
    'MINMAX': ta.minmax(sol_candles, sequential=True).last_min,
    'MOM': ta.mom(sol_candles, sequential=True),        
    'MWDX': ta.mwdx(sol_candles, sequential=True),
    'MSW': ta.msw(sol_candles, sequential=True).sine,
    'NATR': ta.natr(sol_candles, sequential=True),
    'NVI': ta.nvi(sol_candles, sequential=True),
    'OBV': ta.obv(sol_candles, sequential=True),
    'PFE': ta.pfe(sol_candles, sequential=True),
    'PMA': ta.pma(sol_candles, sequential=True).predict,
    'PPO': ta.ppo(sol_candles, sequential=True),
    'QSTICK': ta.qstick(sol_candles, sequential=True),
    'REFLEX': ta.reflex(sol_candles, sequential=True),
    'ROC': ta.roc(sol_candles, sequential=True),
    'ROCP': ta.rocp(sol_candles, sequential=True),
    'ROCR': ta.rocr(sol_candles, sequential=True),
    'RSI': ta.rsi(sol_candles, sequential=True),
    'RVI': ta.rvi(sol_candles, sequential=True),        
    'SAR': ta.sar(sol_candles, sequential=True),
    'SKEW': ta.skew(sol_candles, sequential=True),
    'SMMA': ta.smma(sol_candles, sequential=True),
    'SRSI': ta.srsi(sol_candles, sequential=True).k,
    'STDDEV': ta.stddev(sol_candles, sequential=True),
    'STOCH': ta.stoch(sol_candles, sequential=True).k,
    'STOCHF': ta.stochf(sol_candles, sequential=True).k,
    'SUPERTREND': ta.supertrend(sol_candles, sequential=True).trend,
    'TEMA': ta.tema(sol_candles, sequential=True),
    'TSF': ta.tsf(sol_candles, sequential=True),
    'UI': ta.ui(sol_candles, sequential=True),
    'ULTOSC': ta.ultosc(sol_candles, sequential=True),
    'VAR': ta.var(sol_candles, sequential=True),
    'VI': ta.vi(sol_candles, sequential=True).plus,
    'VIDYA': ta.vidya(sol_candles, sequential=True),
    'VPCI': ta.vpci(sol_candles, sequential=True).vpci,
    'VWAP': ta.vwap(sol_candles, sequential=True),
    'VWMA': ta.vwma(sol_candles, sequential=True),
    'WAD': ta.wad(sol_candles, sequential=True),
    'WILLR': ta.willr(sol_candles, sequential=True),
    'WMA': ta.wma(sol_candles, sequential=True),
    'WT': ta.wt(sol_candles, sequential=True).wt1,
    'ZLEMA': ta.zlema(sol_candles, sequential=True),
    'ZSCORE': ta.zscore(sol_candles, sequential=True),
    
}

# # # # # # # # # # #


feature_names = list(feature_indicators.keys())


def objective(trial):
    

    
    
    coin = 'SOL'
    num_epochs = 100
    timestep = 20
    window_size = 30
    short_weight = 1 # trial.suggest_float('short_weight', 0, 2)
    none_weight = 1 # trial.suggest_int('none_weight', 0, 1)
    long_weight = 1 # trial.suggest_float('long_weight', 0, 2)
    hidden_dim = 32
    num_layers = 2
    dropout_rate = 0.1
    

     
    # num_features = 5
    # indicator_offset = trial.number +60
    # included_features_flags = [trial.suggest_categorical(f'include_feature_{feature_names[i]}', [True, False]) for i in range(indicator_offset, num_features + indicator_offset)]
    # included_features = [feature_names[i] for i, flag in enumerate(included_features_flags, start=indicator_offset) if flag]
    # included_indices = [i for i, flag in enumerate(included_features_flags, start=indicator_offset) if flag]

    def select_feature_indices(trial, num_features, num_to_select):
        selected_indices = []
        for i in range(num_to_select):
            while True:
                idx = trial.suggest_int('feature_idx_{}'.format(len(selected_indices)), 0, num_features - 1)
                if idx not in selected_indices:
                    selected_indices.append(idx)
                    break
        return selected_indices

    num_features = len(feature_names)
    num_to_select = 4
    included_indices = select_feature_indices(trial, num_features, num_to_select)
    included_features = [feature_names[i] for i in included_indices]
    trial.set_user_attr('selected_indices', included_indices)

    print(f"\033[93mTrial {trial.number} has begun....\033[0m")
    print("Included indices trial: ", trial.number, included_indices)
    print("Included features trial: ", trial.number, included_features)
    
    if len(included_features) < 1:
        print(f"No features were selected for trial {trial.number}. Skipping...")
        return 0
    
    df = pd.DataFrame(backtest_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])   
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')   
    
    indicators_df_list = [df]

    for feature in included_features:
        indicator_df = pd.DataFrame(feature_indicators[feature], columns=[feature])
        indicators_df_list.append(indicator_df)

    df_final = pd.concat(indicators_df_list, axis=1)

    df_final = df_final.drop(['open', 'high', 'low', 'volume'], axis=1)
    # df_final.to_csv('df_pro.csv', index=False)
    df_pro = process_data(df_final, window_size, coin)
    df_pro = df_pro.dropna()
    
    X = df_pro[['close'] + included_features]
    y = df_pro['signal']
    
    x_train, x_test, y_train, y_test, class_weights_tensor = tt_split(X, y, short_weight, none_weight, long_weight, timestep, coin)

    
    input_dim = x_train.shape[2]
    output_dim = 3


    model = BiLSTMClassifierWithAttention(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=output_dim, dropout_rate=dropout_rate)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=50, gamma=0.9)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
        
    model.train()
    
    for t in range(1, num_epochs+1):

        y_train_pred = model(x_train)
        loss = loss_fn(y_train_pred, y_train.long())  
        
        if t % num_epochs == 0:
            print("Epoch ", t, "Loss: ", loss.item())
            save_model(model, t, 'SOL')

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        scheduler.step()
    
        
    

    class lstm(Strategy):
        def __init__(self):
            super().__init__()
            self.model = BiLSTMClassifierWithAttention(input_dim, hidden_dim, num_layers, 3, dropout_rate)
            model_filename = f'models/{coin}_trained_model_lstm_{num_epochs}.pth'
            if model_filename:
                self.model.load_state_dict(torch.load(model_filename, map_location=torch.device('cpu')))
                self.model.eval()
            else:
                raise ValueError("Model filename not provided")
            scaler_params_filename = f"models/{coin}_scaler_params.joblib"
            self.scaler_params = joblib.load(scaler_params_filename)
            
        # @cached
        def generate_features(self):
            if len(self.candles) < timestep:
                return None
            recent_candles = self.candles[-timestep:]
            # # # # # # # # # # #
            # Feature Engineering
            # # # # # # # # # # # 
            feature_methods = {
                'ACOSC': lambda idx: ta.acosc(self.candles, sequential=True).osc[-timestep + idx],
                'AD': lambda idx: ta.ad(self.candles, sequential=True)[-timestep + idx],
                'ADX': lambda idx: ta.adx(self.candles, period=14, sequential=True)[-timestep + idx],
                'ALIGATOR': lambda idx: ta.alligator(self.candles, sequential=True).jaw[-timestep + idx],
                'ALMA': lambda idx: ta.alma(self.candles, period=9, sigma=6, sequential=True)[-timestep + idx],
                'AO': lambda idx: ta.ao(self.candles, sequential=True).osc[-timestep + idx],
                'APO': lambda idx: ta.apo(self.candles, fast_period=12, slow_period=26, matype=0, sequential=True)[-timestep + idx],
                'AROONOSC': lambda idx: ta.aroonosc(self.candles, sequential=True)[-timestep + idx],
                'ATR': lambda idx: ta.atr(self.candles, 14, sequential=True)[-timestep + idx],
                'BOLLINGERBANDS': lambda idx: ta.bollinger_bands(self.candles, 20, 2, sequential=True).lowerband[-timestep + idx],
                'BBWIDTH': lambda idx: ta.bollinger_bands_width(self.candles, 20, 2, sequential=True)[-timestep + idx],
                'BOP': lambda idx: ta.bop(self.candles, sequential=True)[-timestep + idx],
                'COPPOCK': lambda idx: ta.cc(self.candles, sequential=True)[-timestep + idx],
                'CCI': lambda idx: ta.cci(self.candles, 20, sequential=True)[-timestep + idx],
                'CFO': lambda idx: ta.cfo(self.candles, 10, sequential=True)[-timestep + idx],
                'CENTREOFGRAVITY': lambda idx: ta.cg(self.candles, period=10, sequential=True)[-timestep + idx],
                'CKSP': lambda idx: ta.cksp(self.candles, sequential=True).long[-timestep + idx],
                'CHANDELIEREXIT': lambda idx: ta.chande(self.candles,sequential=True)[-timestep + idx],
                'CHOP': lambda idx: ta.chop(self.candles, sequential=True)[-timestep + idx],
                'CMO': lambda idx: ta.cmo(self.candles, 14, sequential=True)[-timestep + idx],
                'CORREL': lambda idx: ta.correl(self.candles, 10, sequential=True)[-timestep + idx],
                'CVI': lambda idx: ta.cvi(self.candles, 10, sequential=True)[-timestep + idx],
                'CWMA': lambda idx: ta.cwma(self.candles, 10, sequential=True)[-timestep + idx],
                'DECYCLER': lambda idx: ta.decycler(self.candles, sequential=True)[-timestep + idx],
                'DEMA': lambda idx: ta.dema(self.candles, 10, sequential=True)[-timestep + idx],
                'DEVSTOP': lambda idx: ta.devstop(self.candles, sequential=True)[-timestep + idx],
                'DIPLUS': lambda idx: ta.di(self.candles, 14, sequential=True).plus[-timestep + idx],
                'DIMINUS': lambda idx: ta.di(self.candles, 14, sequential=True).minus[-timestep + idx],
                'DONCHAIN': lambda idx: ta.donchian(self.candles, sequential=True).lowerband[-timestep + idx],
                'DPO': lambda idx: ta.dpo(self.candles, 20, sequential=True)[-timestep + idx],
                'DTI': lambda idx: ta.dti(self.candles, sequential=True)[-timestep + idx],                
                'EMA': lambda idx: ta.ema(self.candles, period=10, sequential=True)[-timestep + idx],
                'EMD': lambda idx: ta.emd(self.candles, 10, sequential=True).lowerband[-timestep + idx],
                'EMV': lambda idx: ta.emv(self.candles, sequential=True)[-timestep + idx],
                'EPMA': lambda idx: ta.epma(self.candles, 10, 20, sequential=True)[-timestep + idx],
                'FISHER': lambda idx: ta.fisher(self.candles, 10, sequential=True).fisher[-timestep + idx],
                'FWMA': lambda idx: ta.fwma(self.candles, 10, sequential=True)[-timestep + idx],                
                'GATOROSC': lambda idx: ta.gatorosc(self.candles, sequential=True).lower[-timestep + idx],
                'GAUSSIAN': lambda idx: ta.gauss(self.candles, 10, sequential=True)[-timestep + idx],
                'HMA': lambda idx: ta.hma(self.candles, sequential=True)[-timestep + idx],
                'HT_DCPERIOD': lambda idx: ta.ht_dcperiod(self.candles, sequential=True)[-timestep + idx],
                'HT_DCPHASE': lambda idx: ta.ht_dcphase(self.candles, sequential=True)[-timestep + idx],
                'HT_PHASOR': lambda idx: ta.ht_phasor(self.candles, sequential=True).inphase[-timestep + idx],
                'HT_SINE': lambda idx: ta.ht_sine(self.candles, sequential=True).sine[-timestep + idx],
                'HT_TRENDLINE': lambda idx: ta.ht_trendline(self.candles, sequential=True)[-timestep + idx],
                'HT_TRENDMODE': lambda idx: ta.ht_trendmode(self.candles, sequential=True)[-timestep + idx],
                'IFT_RSI': lambda idx: ta.ift_rsi(self.candles, sequential=True)[-timestep + idx],
                'ITREND': lambda idx: ta.itrend(self.candles, sequential=True).signal[-timestep + idx],
                'JMA': lambda idx: ta.jma(self.candles, sequential=True)[-timestep + idx],
                'JSA': lambda idx: ta.jsa(self.candles, sequential=True)[-timestep + idx],                
                'KAMA': lambda idx: ta.kama(self.candles, sequential=True)[-timestep + idx],
                'KAUFMANSTOP': lambda idx: ta.kaufmanstop(self.candles, sequential=True)[-timestep + idx],
                'KDJ': lambda idx: ta.kdj(self.candles, sequential=True).k[-timestep + idx],
                'KELTNERCHANNEL': lambda idx: ta.keltner(self.candles, sequential=True).lowerband[-timestep + idx],
                'KST': lambda idx: ta.kst(self.candles, sequential=True).signal[-timestep + idx],
                'KURTOSIS': lambda idx: ta.kurtosis(self.candles, sequential=True)[-timestep + idx],
                'KVO': lambda idx: ta.kvo(self.candles, sequential=True)[-timestep + idx],
                'LINEARREG': lambda idx: ta.linearreg(self.candles, sequential=True)[-timestep + idx],
                'LINEARREG_ANGLE': lambda idx: ta.linearreg_angle(self.candles, sequential=True)[-timestep + idx],
                'LINEARREG_INTERCEPT': lambda idx: ta.linearreg_intercept(self.candles, sequential=True)[-timestep + idx],
                'LINEARREG_SLOPE': lambda idx: ta.linearreg_slope(self.candles, sequential=True)[-timestep + idx],
                'LRSI': lambda idx: ta.lrsi(self.candles, sequential=True)[-timestep + idx],
                'MAAQ': lambda idx: ta.maaq(self.candles, sequential=True)[-timestep + idx],
                'MACD': lambda idx: ta.macd(self.candles, 12, 26, 9, sequential=True).macd[-timestep + idx],
                'MAMA': lambda idx: ta.mama(self.candles, 0.5, 0.05, sequential=True).mama[-timestep + idx],
                'MARKETFI': lambda idx: ta.marketfi(self.candles, sequential=True)[-timestep + idx],
                'MAASU': lambda idx: ta.mass(self.candles, sequential=True)[-timestep + idx],
                'MCGINLEYDYNAMIC': lambda idx: ta.mcginley_dynamic(self.candles, sequential=True)[-timestep + idx],
                'MEAN_AD': lambda idx: ta.mean_ad(self.candles, sequential=True)[-timestep + idx],
                'MFI': lambda idx: ta.mfi(self.candles, sequential=True)[-timestep + idx],
                'MIDPOINT': lambda idx: ta.midpoint(self.candles, sequential=True)[-timestep + idx],
                'MIDPRICE': lambda idx: ta.midprice(self.candles, sequential=True)[-timestep + idx],
                'MINMAX': lambda idx: ta.minmax(self.candles, sequential=True).last_min[-timestep + idx],
                'MOM': lambda idx: ta.mom(self.candles, sequential=True)[-timestep + idx],                
                'MWDX': lambda idx: ta.mwdx(self.candles, sequential=True)[-timestep + idx],
                'MSW': lambda idx: ta.msw(self.candles, sequential=True).sine[-timestep + idx],
                'NATR': lambda idx: ta.natr(self.candles, sequential=True)[-timestep + idx],
                'NVI': lambda idx: ta.nvi(self.candles, sequential=True)[-timestep + idx],
                'OBV': lambda idx: ta.obv(self.candles, sequential=True)[-timestep + idx],
                'PFE': lambda idx: ta.pfe(self.candles, sequential=True)[-timestep + idx],
                'PMA': lambda idx: ta.pma(self.candles, sequential=True).predict[-timestep + idx],
                'PPO': lambda idx: ta.ppo(self.candles, sequential=True)[-timestep + idx],
                'QSTICK': lambda idx: ta.qstick(self.candles, sequential=True)[-timestep + idx],
                'REFLEX': lambda idx: ta.reflex(self.candles, sequential=True)[-timestep + idx],
                'ROC': lambda idx: ta.roc(self.candles, sequential=True)[-timestep + idx],
                'ROCP': lambda idx: ta.rocp(self.candles, sequential=True)[-timestep + idx],
                'ROCR': lambda idx: ta.rocr(self.candles, sequential=True)[-timestep + idx],
                'RSI': lambda idx: ta.rsi(self.candles, sequential=True)[-timestep + idx],
                'RVI': lambda idx: ta.rvi(self.candles, sequential=True)[-timestep + idx],                
                'SAR': lambda idx: ta.sar(self.candles, sequential=True)[-timestep + idx],
                'SKEW': lambda idx: ta.skew(self.candles, sequential=True)[-timestep + idx],
                'SMMA': lambda idx: ta.smma(self.candles, sequential=True)[-timestep + idx],
                'SRSI': lambda idx: ta.srsi(self.candles, sequential=True).k[-timestep + idx],
                'STDDEV': lambda idx: ta.stddev(self.candles, sequential=True)[-timestep + idx],
                'STOCH': lambda idx: ta.stoch(self.candles, sequential=True).k[-timestep + idx],
                'STOCHF': lambda idx: ta.stochf(self.candles, sequential=True).k[-timestep + idx],
                'SUPERTREND': lambda idx: ta.supertrend(self.candles, sequential=True).trend[-timestep + idx],
                'TEMA': lambda idx: ta.tema(self.candles, sequential=True)[-timestep + idx],
                'TSF': lambda idx: ta.tsf(self.candles, sequential=True)[-timestep + idx],
                'UI': lambda idx: ta.ui(self.candles, sequential=True)[-timestep + idx],
                'ULTOSC': lambda idx: ta.ultosc(self.candles, sequential=True)[-timestep + idx],
                'VAR': lambda idx: ta.var(self.candles, sequential=True)[-timestep + idx],
                'VI': lambda idx: ta.vi(self.candles, sequential=True).plus[-timestep + idx],
                'VIDYA': lambda idx: ta.vidya(self.candles, sequential=True)[-timestep + idx],
                'VPCI': lambda idx: ta.vpci(self.candles, sequential=True).vpci[-timestep + idx],
                'VWAP': lambda idx: ta.vwap(self.candles, sequential=True)[-timestep + idx],
                'VWMA': lambda idx: ta.vwma(self.candles, sequential=True)[-timestep + idx],
                'WAD': lambda idx: ta.wad(self.candles, sequential=True)[-timestep + idx],
                'WILLR': lambda idx: ta.willr(self.candles, sequential=True)[-timestep + idx],
                'WMA': lambda idx: ta.wma(self.candles, sequential=True)[-timestep + idx],
                'WT': lambda idx: ta.wt(self.candles, sequential=True).wt1[-timestep + idx],
                'ZLEMA': lambda idx: ta.zlema(self.candles, sequential=True)[-timestep + idx],
                'ZSCORE': lambda idx: ta.zscore(self.candles, sequential=True)[-timestep + idx],
            }
            # # # # # # # # # # #
             
            sequence = []
            for i, candle in enumerate(recent_candles):
                close_price = candle[2]
                features = [close_price]
                for feature_name, method in feature_methods.items():
                    if feature_name in included_features:
                        features.append(method(i))
                scaled_features = (np.array(features) - self.scaler_params['mean']) / self.scaler_params['scale']
                sequence.append(scaled_features)
            
            sequence_array = np.stack(sequence)
            feature_tensor = torch.tensor(sequence_array, dtype=torch.float).unsqueeze(0)
            return feature_tensor
        
        
        def should_long(self) -> bool:
            features = self.generate_features()
            with torch.no_grad():
                prediction = self.model(features)
                _, predicted_class = torch.max(prediction, 1)
            return predicted_class.item() == 2

        def should_short(self) -> bool:
            pass
        
        def should_cancel_entry(self) -> bool:
            return False
        
        def go_long(self):
            qty = utils.size_to_qty(self.balance, self.price, fee_rate=self.fee_rate)
            self.buy = qty, self.price

        def go_short(self):
            qty = utils.size_to_qty(self.balance, self.price, fee_rate=self.fee_rate)
            self.sell = qty, self.price

            
        def update_position(self):
            features = self.generate_features()
            with torch.no_grad():
                prediction = self.model(features)
                _, predicted_class = torch.max(prediction, 1)
            if self.is_long and predicted_class.item() == 0:
                self.liquidate()


    def run_backtest(model_filename):
        
        # os.environ['MODEL_FILENAME'] = model_filename
        
        exchange_name = 'Bybit USDT Perpetual'
        symbol = 'SOL-USDT'
        timeframe = '1h'
        config = {
            'starting_balance': 10_000,
            'fee': 0,
            'type': 'futures',
            'futures_leverage': 10,
            'futures_leverage_mode': 'cross',
            'exchange': exchange_name,
            'warm_up_candles': 300
        }
        routes = [
            {'exchange': exchange_name, 
            'strategy': lstm, 
            'symbol': symbol, 
            'timeframe': timeframe}
        ]
        extra_routes = []
        candles = {
            jh.key(exchange_name, symbol): {
                'exchange': exchange_name,
                'symbol': symbol,
                'candles': backtest_candles,
            },
        }
        result = backtest(
            config,
            routes,
            extra_routes,
            candles, 
            generate_charts=True
        )

        # del os.environ['MODEL_FILENAME']
        
        result['charts']
        result['logs']
        print("finishing balance: ", result['metrics'].get('finishing_balance', 10000))
        calmar_res = round(result['metrics'].get('calmar_ratio', 0), 2)
        if result['metrics']['total'] <= 2:
            calmar_res = 0
        del result
        return calmar_res

    calmar_ratio = run_backtest(f'models/SOL_trained_model_lstm_{num_epochs}.pth')
    del model, X, y, x_train, x_test, y_train, y_test, df_final, df_pro
    gc.collect()
    return calmar_ratio

if __name__ == "__main__":
    
    num_trials = 20
    study = optuna.create_study(direction='maximize')  
    study.optimize(objective, n_trials=num_trials, n_jobs=1)
    print('Best trial:', study.best_trial.params)

    from collections import defaultdict
    import plotly.graph_objects as go
    
    feature_scores = defaultdict(float)
    feature_counts = defaultdict(int)

    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            selected_indices = trial.user_attrs['selected_indices']
            selected_features = [feature_names[i] for i in selected_indices]
            
            for feature_name in selected_features:
                feature_scores[feature_name] += trial.value
                feature_counts[feature_name] += 1

    sorted_feature_names = sorted(feature_scores, key=feature_scores.get, reverse=True)
    scores = [feature_scores[name] for name in sorted_feature_names]

    fig = go.Figure(data=[
        go.Bar(y=sorted_feature_names, x=scores, orientation='h')
    ])

    fig.update_layout(
        template='plotly_dark',
        autosize=False,
        width=700,
        height=800,
        title="Feature Performance Scores",
        xaxis_title="Average Score",
        yaxis_title="Feature",
        yaxis={'autorange': 'reversed'}
    )

    fig.show()