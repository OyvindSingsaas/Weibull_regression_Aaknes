# Predict models

#%%
import tensorflow as tf
from tensorflow.keras import layers, models
from lifelines import WeibullAFTFitter
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from scipy.integrate import trapezoid
from calendar import monthrange
from datetime import datetime
from scipy.integrate import trapezoid


#%%
# ---- Import data, 2007 - April 2025:
df = pd.read_csv(data_path)
#%%

# ---- Some data preparations: 

df = df.rename(columns={'waiting_time': 'waiting time'})
unwanted_types = ['Spike', 'Noise', 'Rockfall_short', 'Rockfall_wide', 'Regional', 'Rockfall']
df = df[~df['Type'].isin(unwanted_types)]
df['Date'] = pd.to_datetime(df['Date'])

def get_season(month):
    if month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5]: return 'Spring'
    elif month in [6, 7, 8]: return 'Summer'
    else: return 'Autumn'

df['Season'] = df['Date'].dt.month.apply(get_season).astype(str)
df['Year'] = df['Date'].dt.year
df = df[df['Year'] != 2022]
df['Type'] = df['Type'].astype(str)
df = pd.get_dummies(df, columns=['Season', 'Type'], drop_first=True)
df['date_numeric'] = df['Date'].astype('int64') // 10**9
df = df.sort_values(by='Date')
df['Month'] = df['Date'].dt.month

df['Registered_Events_Last_1_Days'] = df['Date'].apply(lambda x: ((df['Date'] >= (x - pd.Timedelta(days=1))) & (df['Date'] < x)).sum())
df['Registered_Events_Last_3_Days'] = df['Date'].apply(lambda x: ((df['Date'] >= (x - pd.Timedelta(days=3))) & (df['Date'] < x)).sum())
df['Registered_Events_Last_7_Days'] = df['Date'].apply(lambda x: ((df['Date'] >= (x - pd.Timedelta(days=7))) & (df['Date'] < x)).sum())
df['Registered_Events_Last_14_Days'] = df['Date'].apply(lambda x: ((df['Date'] >= (x - pd.Timedelta(days=14))) & (df['Date'] < x)).sum())
df['Registered_Events_Last_5_Days'] = df['Date'].apply(lambda x: ((df['Date'] >= (x - pd.Timedelta(days=5))) & (df['Date'] < x)).sum())

upper_limit = df['waiting time'].quantile(0.99)
df = df[df['waiting time'] <= upper_limit]


# Choose columns
df = df[['Date', 'waiting time', 'temperature', 'rain', 'rain_last_3_days', 'rain_last_7_days', 'rain_last_14_days',
         'temp_avg_last_3_days','temp_avg_last_7_days', 'temp_avg_last_14_days', 'Season_Spring', 'Season_Summer', 'Season_Winter',
         'Month', 'Year','Registered_Events_Last_1_Days', 'Registered_Events_Last_3_Days', 'Registered_Events_Last_7_Days',
         'Registered_Events_Last_14_Days', 'date_numeric', 'snowmelt', 'WorkingGeophones', 'Registered_Events_Last_5_Days']]

df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

#%%

# --- NN model: 
class FeatureModel(tf.keras.Model):
    def __init__(self, input_dim):
        super(FeatureModel, self).__init__()
        self.feature_extractor = models.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(3, activation='linear', name="lag1")
        ])
        self.k = self.add_weight(name="k", shape=(), initializer=tf.keras.initializers.Zeros(), trainable=True)
        self.beta0 = self.add_weight(name="beta0", shape=(), initializer=tf.keras.initializers.Zeros(), trainable=True)
        self.beta = self.add_weight(name="beta", shape=(3,), initializer=tf.keras.initializers.Zeros(), trainable=True)

    def call(self, inputs):
        return self.feature_extractor(inputs)

def custom_loss(y_true, y_pred):
    alpha = tf.exp(model.beta0 + tf.linalg.matvec(y_pred, model.beta))
    k_pos = tf.nn.softplus(model.k)
    log_likelihood = (tf.math.log(k_pos) - tf.math.log(alpha) +
                      (k_pos - 1) * tf.math.log(y_true / alpha) -
                      tf.math.pow(y_true / alpha, k_pos))
    return -tf.reduce_mean(log_likelihood)

for month in [1,2,3, 4]:
    start_date = f"2025-{month:02d}-01"
    end_day = monthrange(2025, month)[1]
    end_date = f"2025-{month:02d}-{end_day}"
    print(f"\n### Month: {start_date} to {end_date} ###")

    train_df = df[df['Date'] < pd.Timestamp(start_date)]
    test_df = df[(df['Date'] >= pd.Timestamp(start_date)) & (df['Date'] <= pd.Timestamp(end_date))]

    if test_df.empty or train_df.empty:
        print(f"Skipping month {month}: Not enough data")
        continue


    feature_cols = ['waiting time', 'temperature', 'rain', 'rain_last_3_days', 'rain_last_7_days', 'rain_last_14_days',
                    'temp_avg_last_3_days', 'temp_avg_last_7_days', 'temp_avg_last_14_days',
                    'Registered_Events_Last_1_Days', 'Registered_Events_Last_3_Days', 'Registered_Events_Last_7_Days',
                    'Registered_Events_Last_14_Days', 'Season_Spring', 'Season_Summer', 'Season_Winter',
                    'date_numeric', 'snowmelt', 'WorkingGeophones']


    train_df = train_df[feature_cols]
    test_df = test_df[feature_cols]

    categorical_cols = ['Season_Spring', 'Season_Summer', 'Season_Winter']
    numerical_cols = [col for col in train_df.columns if col not in categorical_cols + ['waiting time']]

    X_train_num = train_df[numerical_cols]
    X_test_num = test_df[numerical_cols]
    y_train = train_df['waiting time'].values
    y_test = test_df['waiting time'].values

    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train_num)
    X_test_num_scaled = scaler.transform(X_test_num)

    X_train_cat = train_df[categorical_cols].values
    X_test_cat = test_df[categorical_cols].values

    X_train = np.hstack([X_train_num_scaled, X_train_cat])
    X_test = np.hstack([X_test_num_scaled, X_test_cat])

    model = FeatureModel(input_dim=X_train.shape[1])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=custom_loss)

    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=1000, batch_size=512, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=0)

    new_features_train = model.feature_extractor(X_train).numpy()
    new_features_test = model.feature_extractor(X_test).numpy()

    df_combined = pd.DataFrame(new_features_train, columns=[f'feat_{i}' for i in range(3)])
    df_combined['waiting time'] = y_train
    df_combined['event'] = 1

    aft_model = WeibullAFTFitter()
    aft_model.fit(df_combined, duration_col='waiting time', event_col='event')

    df_combined_test = pd.DataFrame(new_features_test, columns=[f'feat_{i}' for i in range(3)])
    df_combined_test['waiting time'] = y_test
    df_combined_test['event'] = 1

    print('AIC', aft_model.AIC_, 'Bic', aft_model.BIC_)

    # --- CRPS Calculation ---
    crps_values = []
    for i in range(len(X_test)):
        # Predict cumulative hazard and CDF
        cum_hazard_func = aft_model.predict_cumulative_hazard(df_combined_test.iloc[[i]])
        cdf_func = 1 - np.exp(-cum_hazard_func)

        # Extract the time grid and CDF values
        times = cum_hazard_func.index.values
        cdf_values = cdf_func.values.flatten()

        # True event time
        T_value = y_test[i]

        # Indicator function: 1 if t >= T_value else 0
        indicator = (times >= T_value).astype(float)

        # Compute squared differences
        squared_diffs = (cdf_values - indicator) ** 2

        # Approximate the integral (CRPS)
        crps = trapezoid(squared_diffs, times)

        crps_values.append(crps)

    # Calculate mean CRPS for this month
    mean_crps = np.mean(crps_values)
    print(f"Mean CRPS for {datetime(2024, month, 1).strftime('%B')} 22024016: {mean_crps:.4f}")

    # CDF Quantile Plot
    all_cdf_values = []
    for i in range(len(X_test)):
        cdf_data = aft_model.predict_cumulative_hazard(df_combined_test.iloc[[i]])
        T_value = df_combined_test.iloc[i]['waiting time']
        cdf_value = 1 - np.exp(-np.interp(T_value, cdf_data.index.to_numpy().ravel(), cdf_data.values.ravel()))
        all_cdf_values.append(cdf_value)

    all_cdf_values = np.array(all_cdf_values)
    quantile_bins = [0, 0.25, 0.5, 0.75, 1.0]
    quantile_labels = ['(0-0.25)', '(0.25-0.5)', '(0.5-0.75)', '(0.75-1)']
    cdf_quantiles = np.digitize(all_cdf_values, bins=quantile_bins, right=True)


    plt.figure(figsize=(6, 4))
    plt.hist(cdf_quantiles, bins=np.arange(1, 6) - 0.5, alpha=0.7, color='royalblue', density=True, rwidth=0.8)
    plt.xticks(ticks=range(1, 5), labels=quantile_labels, fontsize=12)
    plt.title(f"CDF-values ({datetime(2024, month, 1).strftime('%B')} 2024)", fontsize=20)
    plt.xlabel("CDF Quantile Bins", fontsize=18)
    plt.ylabel("Density", fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend([f"n = {len(y_test)}"], loc='upper right', fontsize=14)
    plt.tight_layout()
    plt.show()

 
    from scipy.stats import chisquare
    # Chi-Squared GOF Test
    num_bins = 4
    counts, _ = np.histogram(all_cdf_values, bins=np.linspace(0, 1, num_bins + 1))
    expected = np.full(num_bins, len(all_cdf_values) / num_bins)
    chi_stat, p_val = chisquare(counts, expected)
    print(f"Chi-Squared GOF Test: χ² = {chi_stat:.4f}, p = {p_val:.4f}")

        # --- Chi-Squared Test for Uniformity ---
    observed_counts, _ = np.histogram(all_cdf_values, bins=quantile_bins)
    expected_count = len(all_cdf_values) / 4  # 4 bins

    chi_squared_stat = np.sum((observed_counts - expected_count) ** 2 / expected_count)
    critical_value = 7.815  # chi-squared critical value with df=3 at alpha=0.05

    print(f"Chi-squared statistic: {chi_squared_stat:.3f}")
    if chi_squared_stat > critical_value:
        print("Chi-squared test result: Reject null hypothesis (non-uniform PIT)")
    else:
        print("Chi-squared test result: Fail to reject null hypothesis (uniform PIT)")



