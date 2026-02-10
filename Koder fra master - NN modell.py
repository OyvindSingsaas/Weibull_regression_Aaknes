# Kode for NN modell - with extra features 

import tensorflow as tf
from tensorflow.keras import layers, models
from lifelines import WeibullAFTFitter, ExponentialFitter, WeibullFitter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
from scipy.interpolate import interp1d
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
#from tensorflow.keras import layers, models, regularizers

# ---- Import data, 2007 - April 2025:
df = pd.read_csv('/data/Aknes_and_met_data.csv')

# ---- Some data preparations: 

# Rename
df = df.rename(columns={'waiting_time': 'waiting time'})

# Remove unwanted 'Type' values
unwanted_types = ['Spike', 'Noise', 'Rockfall_short', 'Rockfall_wide', 'Regional', 'Rockfall']
df = df[~df['Type'].isin(unwanted_types)]

# Convert 'Date' to datetime 
df['Date'] = pd.to_datetime(df['Date'])

# Define a function to determine season
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

# Apply function to create 'Season' column
df['Season'] = df['Date'].dt.month.apply(get_season)
df['Season'] = df['Season'].astype(str) 

# Remove year when eval up to 2021)
df['Year'] = df['Date'].dt.year
df = df[df['Year'] != 2022]
df = df[df['Year'] != 2023]
df = df[df['Year'] != 2024]
df = df[df['Year'] != 2025]


# Add month
df['Month'] = df['Date'].dt.month

df['Type'] = df['Type'].astype(str)
# One-hot encode 'Season' and 'Type'
df = pd.get_dummies(df, columns=['Season', 'Type'], drop_first=True)


# Convert to numeric (Unix timestamp in seconds)
df['date_numeric'] = df['Date'].astype('int64') // 10**9  # Convert nanoseconds to seconds

# Sort, (not necessary)
df = df.sort_values(by='Date')

# Create additional covariates
df['Registered_Events_Last_5_Days'] = df['Date'].apply(
    lambda x: ((df['Date'] >= (x - pd.Timedelta(days=5))) & (df['Date'] < x)).sum())
df['Registered_Events_Last_1_Days'] = df['Date'].apply(
    lambda x: ((df['Date'] >= (x - pd.Timedelta(days=1))) & (df['Date'] < x)).sum())
df['Registered_Events_Last_3_Days'] = df['Date'].apply(
    lambda x: ((df['Date'] >= (x - pd.Timedelta(days=3))) & (df['Date'] < x)).sum())
df['Registered_Events_Last_7_Days'] = df['Date'].apply(
    lambda x: ((df['Date'] >= (x - pd.Timedelta(days=7))) & (df['Date'] < x)).sum())
df['Registered_Events_Last_14_Days'] = df['Date'].apply(
    lambda x: ((df['Date'] >= (x - pd.Timedelta(days=14))) & (df['Date'] < x)).sum())

# Remove outliers
upper_limit = df['waiting time'].quantile(0.99)
filtered_df = df[df['waiting time'] <= upper_limit]

# Data set used in model, choose relevant columns
filtered_df = filtered_df[['waiting time', 'temperature', 'rain', 'rain_last_3_days', 'rain_last_7_days', 'rain_last_14_days', 'temp_avg_last_3_days','temp_avg_last_7_days', 'temp_avg_last_14_days', 'Season_Spring', 'Season_Summer', 'Season_Winter','Registered_Events_Last_1_Days', 'Registered_Events_Last_3_Days', 'Registered_Events_Last_7_Days', 'Registered_Events_Last_14_Days', 'date_numeric', 'snowmelt', 'WorkingGeophones']]


# ---- Scale covariates and split to train / test data: 

# Identify numerical and categorical features
categorical_cols = ['Season_Spring', 'Season_Summer', 'Season_Winter']
numerical_cols = [col for col in filtered_df.columns if col not in categorical_cols + ['waiting time']]

# Separate data
X = filtered_df.drop(columns=['waiting time'])
X_num = filtered_df[numerical_cols]
X_cat = filtered_df[categorical_cols]

# Scale numerical only
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)

# Combine back into one matrix
X_scaled = np.hstack([X_num_scaled, X_cat.values])  # NumPy arrays
y = filtered_df['waiting time'].values

# Split train / test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ---- Neural network model: 

# Define NN
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

# Initialialize model
model = FeatureModel(input_dim=X_train.shape[1])

# Define loss-function
def custom_loss(y_true, y_pred):
    alpha = tf.exp(model.beta0 + tf.linalg.matvec(y_pred, model.beta))  # α = exp(β₀ + Xβ)
    k_pos = tf.nn.softplus(model.k) 

    log_likelihood = (
        tf.math.log(k_pos) - tf.math.log(alpha) +
        (k_pos - 1) * tf.math.log(y_true / alpha) -
        tf.math.pow(y_true / alpha, k_pos)
    )

    return -tf.reduce_mean(log_likelihood) 

# Compile and train model
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=custom_loss)

# Add early stopping

early_stopping = EarlyStopping(
    monitor='val_loss',    # Watch validation loss
    patience=20,           # Wait this many epochs for improvement
    restore_best_weights=True  # Roll back to best weights
)

history = model.fit(
    X_train, y_train,
    epochs=1000,
    batch_size=128*4,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping])

# ---- Create new data set for AFT model and fit AFT model: 

# Extract new features from the trained model
new_features_train = model.feature_extractor(X_train)
new_features_test = model.feature_extractor(X_test)

# Convert to NumPy
new_features_train = new_features_train.numpy()
new_features_test = new_features_test.numpy()

# Add waiting time data and mark all as censored 
df_combined = pd.DataFrame(new_features_train, columns=[f'feat_{i}' for i in range(3)])
df_combined['waiting time'] = y_train
df_combined['event'] = 1  # all data censored

# Fit AFT model 
aft_model_nll_loss = WeibullAFTFitter()
aft_model_nll_loss.fit(df_combined, duration_col='waiting time', event_col='event') 

# print(aft_model_nll_loss.AIC_)
# print(aft_model_nll_loss.BIC_)

# Create test data set
df_combined_test = pd.DataFrame(new_features_test, columns=[f'feat_{i}' for i in range(3)])
df_combined_test['waiting time'] = y_test
df_combined_test['event'] = 1  # all data censored

# ---- Calculate CDF values to create PIT-histograms:

# Array for all CDF-values for all obs
all_cdf_values = []

# Calc one CDF-value for each obs in test-data
for i in range(len(X_test)):
    cdf_data = aft_model_nll_loss.predict_cumulative_hazard(df_combined_test.iloc[[i]])  # CDF data
    
    T_value = df_combined_test.iloc[i]['waiting time'] 
    cdf_value = 1 - np.exp(-np.interp(T_value, cdf_data.index.to_numpy().ravel(), cdf_data.values.ravel())) # CDF value for obs

    # Add value to list
    all_cdf_values.append(cdf_value)

all_cdf_values = np.array(all_cdf_values)

# Plot
plt.figure(figsize=(6, 4))
plt.hist(all_cdf_values, bins=50, alpha=0.7, color='blue', density=True)
#plt.title("CDF-values Weibull Model (w/ non-linear cov.)", fontsize = 20)
plt.title("CDF-values Weibull Model (w/ daily measures)", fontsize = 20)
plt.xlabel("CDF-values", fontsize = 18)
plt.ylabel("Density", fontsize = 18)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.show()

# ---- Calculatehe CRPS: 

from scipy.integrate import trapezoid

crps_values = []

for i in range(len(X_test)):
    # Predict cumulative hazard and CDF
    cum_hazard_func = aft_model_nll_loss.predict_cumulative_hazard(df_combined_test.iloc[[i]])
    cdf_func = 1 - np.exp(-cum_hazard_func)

    # Extract the time grid and CDF values
    times = cum_hazard_func.index.values
    cdf_values = cdf_func.values.flatten()

    # True event time
    T_value = y_test[[i]]

    # Indicator function: 1 if t >= T_value else 0
    indicator = (times >= T_value).astype(float)

    # Compute squared differences
    squared_diffs = (cdf_values - indicator) ** 2

    # Approximate the integral (CRPS)
    crps = trapezoid(squared_diffs, times)

    crps_values.append(crps)

# Calculate mean CRPS
mean_crps = np.mean(crps_values)

print(f"Mean CRPS over all points: {mean_crps:.4f}")