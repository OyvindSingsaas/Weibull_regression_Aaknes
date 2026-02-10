"""
quicktest.py

Minimal demonstration of Weibull regression and neural network Weibull model with PDPs,
using only the test_data folder. This script is self-contained and does not require private data.
"""

import pandas as pd
import numpy as np
from lifelines import WeibullAFTFitter
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt


# --- Load test data ---
data_path = "test_data/merged_event_and_met_data_23_adjusted_WT-test.csv"
df = pd.read_csv(data_path)

# --- Preprocessing (minimal, as in PDP.py) ---
df = df.rename(columns={'waiting_time': 'waiting time'})
df['Date'] = pd.to_datetime(df['Date'])
def get_season(month):
    if month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5]: return 'Spring'
    elif month in [6, 7, 8]: return 'Summer'
    else: return 'Autumn'
df['Season'] = df['Date'].dt.month.apply(get_season)
df = pd.get_dummies(df, columns=['Season'], drop_first=True)

# Select a subset of features for demonstration
features = ['temperature', 'rain', 'rain_last_3_days', 'Season_Spring', 'Season_Summer', 'Season_Winter']
available_features = [f for f in features if f in df.columns]
print(f"Using features: {available_features}")
df = df[['waiting time'] + available_features].dropna()

# --- Fit Weibull AFT model ---
aft = WeibullAFTFitter()
aft.fit(df.assign(event=1), duration_col='waiting time', event_col='event')
print("WeibullAFTFitter summary:")
print(aft.summary)

# --- Prepare data for NN model ---
X = df[available_features].values
y = df['waiting time'].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- Simple NN Weibull model ---
class SimpleWeibullNN(tf.keras.Model):
    def __init__(self, input_dim):
        super().__init__()
        self.dense = tf.keras.layers.Dense(8, activation='relu')
        self.k = self.add_weight(name="k", shape=(), initializer='zeros', trainable=True)
        self.beta0 = self.add_weight(name="beta0", shape=(), initializer='zeros', trainable=True)
        self.beta = self.add_weight(name="beta", shape=(8,), initializer='zeros', trainable=True)
    def call(self, x):
        return self.dense(x)
def nn_loss(y_true, y_pred):
    alpha = tf.exp(model.beta0 + tf.linalg.matvec(y_pred, model.beta))
    k_pos = tf.nn.softplus(model.k)
    log_likelihood = (
        tf.math.log(k_pos) - tf.math.log(alpha) +
        (k_pos - 1) * tf.math.log(y_true / alpha) -
        tf.math.pow(y_true / alpha, k_pos)
    )
    return -tf.reduce_mean(log_likelihood)
model = SimpleWeibullNN(X_scaled.shape[1])
model.compile(optimizer='adam', loss=nn_loss)
model.fit(X_scaled, y, epochs=50, batch_size=16, verbose=0)

feature_idx = available_features.index('temperature')
temp_range = np.linspace(X[:, feature_idx].min(), X[:, feature_idx].max(), 50)
"""
# --- Temperature effect in neural network Weibull model ---


pdp = []
for val in temp_range:
    X_temp = X_scaled.copy()
    X_temp[:, feature_idx] = val
    preds = model.dense(X_temp).numpy()
    alpha = np.exp(model.beta0.numpy() + preds @ model.beta.numpy())
    k_pos = tf.nn.softplus(model.k).numpy()
    expected = alpha * np.exp(np.euler_gamma / k_pos)  # mean of Weibull
    pdp.append(np.mean(expected))
plt.plot(temp_range, pdp)
plt.xlabel('Temperature')
plt.ylabel('Expected Waiting Time (NN Weibull)')
plt.title('Temperature Effect')
plt.show()
"""

# --- Compare temperature effect: AFT Weibull vs NN Weibull ---
# For AFT: Predict expectation for a range of temperature values, holding other features at their mean
mean_vals = df[available_features].mean().to_dict()
aft_expect = []
nn_expect = []
for val in temp_range:
    row = mean_vals.copy()
    row['temperature'] = val
    X_aft = pd.DataFrame([row])

    # AFT Weibull prediction
    aft_expect.append(aft.predict_expectation(X_aft)[0])

    # NN Weibull prediction on the same row
    X_nn = scaler.transform(X_aft[available_features])
    preds = model.dense(X_nn).numpy()
    alpha = np.exp(model.beta0.numpy() + preds @ model.beta.numpy())
    k_pos = tf.nn.softplus(model.k).numpy()
    expected = alpha * np.exp(np.euler_gamma / k_pos)
    nn_expect.append(np.mean(expected))

# Plot both curves
plt.figure(figsize=(8, 5))
plt.plot(temp_range, nn_expect, label='NN Weibull (rowwise)', color='tab:blue')
plt.plot(temp_range, aft_expect, label='AFT Weibull', color='tab:orange')
plt.xlabel('Temperature')
plt.ylabel('Expected Waiting Time')
plt.title('Temperature Effect: NN Weibull vs AFT Weibull (rowwise)')
plt.legend()
plt.tight_layout()
plt.show()
