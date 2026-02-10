#%%
# 
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
import os
from tensorflow.keras.callbacks import EarlyStopping


#%%

# Load the data
df = pd.read_csv('/Users/kjerstidengerud/Documents/Fysmat/Masteroppgave/Aaknes data/Egne datasett/merged_event_and_met_data_23_adjusted_WT.csv')

# Preprocessing:
df = df.rename(columns={'waiting_time': 'waiting time'})
unwanted_types = ['Spike', 'Noise', 'Rockfall_short', 'Rockfall_wide', 'Regional', 'Rockfall']
df = df[~df['Type'].isin(unwanted_types)]
df['Date'] = pd.to_datetime(df['Date'])
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'
df['Season'] = df['Date'].dt.month.apply(get_season)
df['Season'] = df['Season'].astype(str) 
df['Year'] = df['Date'].dt.year
df['Type'] = df['Type'].astype(str)
df = pd.get_dummies(df, columns=['Season', 'Type'], drop_first=True)
df['date_numeric'] = df['Date'].astype('int64') // 10**9  
df = df.sort_values(by='Date')
df['Registered_Events_Last_5_Days'] = df['Date'].apply(
    lambda x: ((df['Date'] >= (x - pd.Timedelta(days=5))) & (df['Date'] < x)).sum()
)

upper_limit = df['waiting time'].quantile(0.99)
filtered_df = df[df['waiting time'] <= upper_limit]

filtered_df = filtered_df[['waiting time', 'temperature', 'rain', 'rain_last_3_days', 'temp_avg_last_3_days', 'Season_Spring', 'Season_Summer', 'Season_Winter', 'Registered_Events_Last_5_Days', 'date_numeric', 'snowmelt', 'WorkingGeophones']]

categorical_cols = ['Season_Spring', 'Season_Summer', 'Season_Winter']
numerical_cols = [col for col in filtered_df.columns if col not in categorical_cols + ['waiting time']]

feature_names = numerical_cols + categorical_cols

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

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


#%% 

# Define the neural network 
class FeatureModel(tf.keras.Model):
    def __init__(self, input_dim):
        super(FeatureModel, self).__init__()
        self.feature_extractor = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(3, activation='linear', name="lag1")  # Compress features into 10 valuable latent features
        ])

        self.k = self.add_weight(name="k", shape=(), initializer=tf.keras.initializers.Zeros(), trainable=True)
        self.beta0 = self.add_weight(name="beta0", shape=(), initializer=tf.keras.initializers.Zeros(), trainable=True)
        self.beta = self.add_weight(name="beta", shape=(3,), initializer=tf.keras.initializers.Zeros(), trainable=True)


    def call(self, inputs):
        return self.feature_extractor(inputs)

# Initializing the model
model = FeatureModel(input_dim=X_train.shape[1])

# Define the custom loss function
def custom_loss(y_true, y_pred):
    alpha = tf.exp(model.beta0 + tf.linalg.matvec(y_pred, model.beta))  # α = exp(β₀ + Xβ)
    k_pos = tf.nn.softplus(model.k)  # Ensure k > 0

    log_likelihood = (
        tf.math.log(k_pos) - tf.math.log(alpha) +
        (k_pos - 1) * tf.math.log(y_true / alpha) -
        tf.math.pow(y_true / alpha, k_pos)
    )

    return -tf.reduce_mean(log_likelihood)  # Use mean instead of sum

# Compile and train the model
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=custom_loss)

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
    callbacks=[early_stopping]
)

#%%
# Create new data set

# Extract new features from the trained model
new_features_train = model.feature_extractor(X_train)
new_features_test = model.feature_extractor(X_test)

# Convert to NumPy for further analysis
new_features_train = new_features_train.numpy()
new_features_test = new_features_test.numpy()

# Fit the AFT model using lifelines
aft_model_nll_loss = WeibullAFTFitter()
df_combined = pd.DataFrame(new_features_train, columns=[f'feat_{i}' for i in range(3)])
df_combined['waiting time'] = y_train
df_combined['event'] = 1  # All data censored
aft_model_nll_loss.fit(df_combined, duration_col='waiting time', event_col='event')

#%%
# List of features to plot - original covariates
features_to_plot = [
    'rain',
    'temperature',
    'temp_avg_last_3_days',
    'rain_last_3_days',
    'WorkingGeophones',
    'snowmelt',
    'Registered_Events_Last_5_Days',
    'date_numeric',
    'Season_Spring', 'Season_Summer', 'Season_Winter'
]

features_to_scale = [
    'rain',
    'temperature',
    'temp_avg_last_3_days',
    'rain_last_3_days',
    'WorkingGeophones',
    'snowmelt',
    'Registered_Events_Last_5_Days',
    'date_numeric',
    #'Season_Spring', 'Season_Summer', 'Season_Winter'
]


# Custom labels for better plotting
custom_labels = {
    "rain": "Precipiation",
    "temperature": "Temperature",
    "temp_avg_last_3_days": "Avg. temp. last 3d",
    "rain_last_3_days": "Acc. prec 3d.",
    "WorkingGeophones": "Number of Working Geophones",
    "snowmelt": "Snowmelt",
    "Registered_Events_Last_5_Days": "Number of Events 5d",
    "date_numeric": "Date",
    "Season_Spring": "Spring",
    "Season_Summer": "Summer",
    "Season_Winter": "Winter"
}

# %%

#Store PDP data for all runs
pdp_results = {feature: [] for feature in features_to_plot}

# Loop 5 times
for run in range(10):
    print(f"Run {run + 1} / 5")

    # Re-split data with different seed each run
    #X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42 + run)

    # Initialize and train model
    model = FeatureModel(input_dim=X_train.shape[1])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=custom_loss)
    early_stopping = EarlyStopping(
    monitor='val_loss',    # Watch validation loss
    patience=20,           # Wait this many epochs for improvement
    restore_best_weights=True)  # Roll back to best weights
    

    #model.fit(X_train, y_train, epochs=300, batch_size=128*4, validation_data=(X_test, y_test))
    history = model.fit(
        X_train, y_train,
        epochs=1000,
        batch_size=128*4,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping])

    # Feature extraction
    new_features_train = model.feature_extractor(X_train).numpy()
    new_features_test = model.feature_extractor(X_test).numpy()

    # AFT model training
    df_combined = pd.DataFrame(new_features_train, columns=[f'feat_{i}' for i in range(3)])
    df_combined['waiting time'] = y_train
    df_combined['event'] = 1
    aft_model_nll_loss = WeibullAFTFitter()
    aft_model_nll_loss.fit(df_combined, duration_col='waiting time', event_col='event')

    # PDP calculations
    for i, feature_name in enumerate(features_to_plot):
        #feature_idx = list(X_scaled.columns).index(feature_name)
        feature_idx = feature_names.index(feature_name)
        feature_col = X_scaled[:, feature_idx]
        feature_col = pd.Series(feature_col)


        if feature_name in ['Season_Spring', 'Season_Summer', 'Season_Winter']:
            feature_values_original = [0, 1]
        
        else:
            feature_values_original = np.linspace(
                feature_col.quantile(0.05),
                feature_col.quantile(0.95),
                100
            )

        mean_expected_times = []

        for val in feature_values_original:
            X_temp = X_test.copy()
            if feature_name in ['Season_Spring', 'Season_Summer', 'Season_Winter']:
                #X_temp[:, feature_idx] = val
                season_indices = [feature_names.index(season) for season in ['Season_Spring', 'Season_Summer', 'Season_Winter']]
                feature_idx = feature_names.index(feature_name)

                if val == 1:
                    # Set all season columns to 0
                    for season_idx in season_indices:
                        X_temp[:, season_idx] = 0
                    # Set the current season to 1
                    X_temp[:, feature_idx] = 1
                else:
                    # Set the current season to 0
                    X_temp[:, feature_idx] = 0

            else:
                X_temp[:, feature_idx] = val #(val - feature_means[i]) / feature_stds[i]

            extracted_features = model.feature_extractor(X_temp).numpy()
            df_features = pd.DataFrame(extracted_features, columns=[f"feat_{j}" for j in range(3)])
            expected = aft_model_nll_loss.predict_expectation(df_features)
            mean_expected_times.append(np.mean(expected))

        # Store PDP line for this run
        pdp_results[feature_name].append((feature_values_original, mean_expected_times))


#%%
# Correct scale x-axis 

# Map feature name to original mean and std for inverse transformation
feature_means = dict(zip(numerical_cols, scaler.mean_))
feature_stds = dict(zip(numerical_cols, scaler.scale_))

for feature_name in features_to_plot:
    plt.figure(figsize=(8, 6))

    all_y_vals = []
    all_x_vals_unscaled = []

    for run_idx, (x_vals_scaled, y_vals) in enumerate(pdp_results[feature_name]):
        # Inverse transform x-axis if needed
        if feature_name in numerical_cols:
            mean = feature_means[feature_name]
            std = feature_stds[feature_name]
            x_vals_unscaled = x_vals_scaled * std + mean
        else:
            x_vals_unscaled = x_vals_scaled  # For categorical (0 or 1)

        plt.plot(x_vals_unscaled, y_vals, label=f"Run {run_idx + 1}", lw=2)
        all_y_vals.append(y_vals)
        all_x_vals_unscaled.append(x_vals_unscaled)

    # Compute and plot mean line
    all_y_vals = np.array(all_y_vals)
    mean_y_vals = np.mean(all_y_vals, axis=0)
    mean_x_vals = np.mean(all_x_vals_unscaled, axis=0)

    plt.plot(mean_x_vals, mean_y_vals, color='black', linestyle='--', lw=2, label='Mean')

    # Labels and styling
    plot_label = custom_labels.get(feature_name, feature_name)
    plt.xlabel(plot_label, fontsize=18)
    plt.ylabel("Expected Waiting Time [s]", fontsize=18)
    plt.title(f"PDP: {plot_label}", fontsize=20)
    plt.grid(True)
    plt.legend()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

    # Save the plot to the specified path
    plot_filename = f"PDP_{feature_name}_20runs.png"
    plot_filepath = os.path.join(save_dir, plot_filename)

# %%
# Map feature name to original mean and std for inverse transformation
feature_means = dict(zip(numerical_cols, scaler.mean_))
feature_stds = dict(zip(numerical_cols, scaler.scale_))

for feature_name in features_to_plot:
    plt.figure(figsize=(8, 6))

    all_y_vals = []
    all_x_vals_unscaled = []

    for x_vals_scaled, y_vals in pdp_results[feature_name]:
        # Inverse transform x-axis if needed
        if feature_name in numerical_cols:
            mean = feature_means[feature_name]
            std = feature_stds[feature_name]
            x_vals_unscaled = x_vals_scaled * std + mean
        else:
            x_vals_unscaled = x_vals_scaled  # For categorical

        all_y_vals.append(y_vals)
        all_x_vals_unscaled.append(x_vals_unscaled)

    # Compute and plot mean line
    all_y_vals = np.array(all_y_vals)
    mean_y_vals = np.mean(all_y_vals, axis=0)
    mean_x_vals = np.mean(all_x_vals_unscaled, axis=0)

    plt.plot(mean_x_vals, mean_y_vals, color='black', linestyle='--', lw=2, label='Mean')

    # Labels and styling
    plot_label = custom_labels.get(feature_name, feature_name)
    plt.xlabel(plot_label, fontsize=18)
    plt.ylabel("Expected Waiting Time [s]", fontsize=18)
    plt.title(f"PDP: {plot_label}", fontsize=20)
    plt.grid(True)
    plt.legend()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    #plt.show()

# %%
#Plotting just date
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

feature_name = "date_numeric"

plt.figure(figsize=(8, 6))

all_y_vals = []
all_x_vals_unscaled = []

for run_idx, (x_vals_scaled, y_vals) in enumerate(pdp_results[feature_name]):
    # Inverse transform and convert to datetime
    mean = feature_means[feature_name]
    std = feature_stds[feature_name]
    x_vals_unscaled = x_vals_scaled * std + mean
    x_vals_unscaled = pd.to_datetime(x_vals_unscaled, unit='s')  # Convert from Unix timestamp

    plt.plot(x_vals_unscaled, y_vals, label=f"Run {run_idx + 1}", lw=2)
    all_y_vals.append(y_vals)
    all_x_vals_unscaled.append(x_vals_unscaled)

# Compute and plot mean line
all_y_vals = np.array(all_y_vals)
mean_y_vals = np.mean(all_y_vals, axis=0)
mean_x_vals = pd.to_datetime(np.mean([x.view('int64') for x in all_x_vals_unscaled], axis=0))

plt.plot(mean_x_vals, mean_y_vals, color='black', linestyle='--', lw=2, label='Mean')

# Labels and styling
plot_label = custom_labels.get(feature_name, feature_name)
plt.xlabel(plot_label, fontsize=18)
plt.ylabel("Expected Waiting Time [s]", fontsize=18)
plt.title(f"PDP: {plot_label}", fontsize=20)
plt.grid(True)
plt.legend()
plt.xticks(fontsize=14, rotation=45)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()
