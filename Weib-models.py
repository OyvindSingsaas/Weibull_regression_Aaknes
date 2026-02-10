
#Code Weib models: exponential, Weibull and Weibull AFT

# Simulation study Åknes data

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import WeibullAFTFitter, ExponentialFitter, WeibullFitter
from sklearn.preprocessing import StandardScaler
from scipy.stats import weibull_min
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid



#%%
# ---- Import data:
dff = pd.read_csv(data_path)


# --- Some data preparations: 

# Remove unwanted 'Type' values
unwanted_types = ['Spike', 'Noise', 'Rockfall_short', 'Rockfall_wide', 'Regional', 'Rockfall']
dff = dff[~dff['Type'].isin(unwanted_types)]

# Rename
dff = dff.rename(columns={
    'waiting_time': 'T',
    'Season_Spring': 'Sesong_Vår',
    'rain': 'Nedbør',
    'temperature': 'Temperatur'})

# Remove outliers
upper_limit = dff['T'].quantile(0.99)
dff = dff[dff['T'] <= upper_limit]

dff['Date'] = pd.to_datetime(dff['Date'])

# Define a function to determine the season
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
dff['Season'] = dff['Date'].dt.month.apply(get_season)

dff['Season'] = dff['Season'].astype(str)  # Ensure it's a string
dff['Year'] = dff['Date'].dt.year
dff['Type'] = dff['Type'].astype(str)
# One-hot encode 'Season' and 'Type'
dff = pd.get_dummies(dff, columns=['Season', 'Type'], drop_first=True)

# Rename
dff = dff.rename(columns={
    'waiting_time': 'T',
    'Season_Spring': 'Sesong_Vår',
    'rain': 'Nedbør',
    'temperature': 'Temperatur'
})

# Change True/False til 1/0
dff['Sesong_Vår'] = dff['Sesong_Vår'].astype(int)
dff['Season_Summer'] = dff['Season_Summer'].astype(int)
dff['Season_Winter'] = dff['Season_Winter'].astype(int)

# Remove 2022
dff['Year'] = pd.to_datetime(dff['Date']).dt.year
dff = dff[dff['Year'] != 2022]
dff = dff.sort_values(by='Date')

# Number of events last five days
dff['Registered_Events_Last_5_Days'] = dff['Date'].apply(
    lambda x: ((dff['Date'] >= (x - pd.Timedelta(days=5))) & (dff['Date'] < x)).sum())

# Convert to numeric (Unix timestamp in seconds)
dff['date_numeric'] = dff['Date'].astype('int64') // 10**9  # Convert nanoseconds to seconds - delt på 1000 igjen
dff = dff[['T', 'Sesong_Vår', 'Season_Summer', 'Season_Winter', 'Nedbør', 'Temperatur', 'snowmelt', 'WorkingGeophones', 
           'rain_last_3_days', 'temp_avg_last_3_days', 'Registered_Events_Last_5_Days', 'date_numeric']]

# Standardize
scaler = StandardScaler()
cols_to_standardize = ['Nedbør', 'Temperatur', 'rain_last_3_days', 'temp_avg_last_3_days', 'Registered_Events_Last_5_Days', 'date_numeric', 'snowmelt', 'WorkingGeophones']
dff[cols_to_standardize] = scaler.fit_transform(dff[cols_to_standardize])

# Mark as censored
dff['event'] = 1  

# Split data
train_dff, test_dff = train_test_split(dff, test_size=0.2, random_state=42)

#%% 
# ---- Fit AFT model:

# Weibull AFT model: 
aft2 = WeibullAFTFitter()
aft2.fit(train_dff, duration_col="T", event_col='event')

print(f"Weibull AFT2 (with covariates) - AIC: {aft2.AIC_}, BIC: {aft2.BIC_}")

#%%
# ----- PIT histogram Weibull AFT model

# Array for å lagre CDF-verdier for alle observasjoner
all_cdf_values = []

# Compute a CDF-verdi for each row in df
for i in range(len(test_dff)):
    cdf_data = aft2.predict_cumulative_hazard(test_dff.iloc[[i]])  # Hent CDF data (DataFrame)
    T_value = test_dff.iloc[i]['T']  # Første kolonne (tid)
    cdf_value = 1 - np.exp(-np.interp(T_value, cdf_data.index.to_numpy().ravel(), cdf_data.values.ravel()))
    all_cdf_values.append(cdf_value)

# Convert to numpy
all_cdf_values = np.array(all_cdf_values)

# Plot histogram of CDF values
plt.figure(figsize=(6, 4))
plt.hist(all_cdf_values, bins=50, alpha=0.7, color='darkgreen', density=True)
plt.title("CDF-values Weibull Model (w/ linear cov.)", fontsize = 20)
plt.xlabel("CDF-values", fontsize = 18)
plt.ylabel("Density", fontsize = 18)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.show()

samples = test_dff['T']

#%%
# ---- CRPS Weibull AFT model: 

crps_values = []

for i in range(len(test_dff)):
    # Predict cumulative hazard and CDF
    cum_hazard_func = aft2.predict_cumulative_hazard(test_dff.iloc[[i]])
    cdf_func = 1 - np.exp(-cum_hazard_func)

    # Extract the time grid and CDF values
    times = cum_hazard_func.index.values
    cdf_values = cdf_func.values.flatten()

    # True event time
    T_value = test_dff.iloc[i]['T']

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

#%%

# ---- Fit Exponential model and Weib model: 

train_df = train_dff
test_df = test_dff

# Exp
exp = ExponentialFitter()
exp.fit(durations = train_df['T'], event_observed = train_df['event'])

# Weibull-model
weibull = WeibullFitter()
weibull.fit(train_df["T"], event_observed=train_df["event"])

# Evaluer AIC og BIC for modellene
print("\nAIC and BIC for each model:")
print(f"Weibull AFT (with covariates) - AIC: {aft2.AIC_}, BIC: {aft2.BIC_}")
print(f"Exponential model - AIC: {exp.AIC_}, BIC: {exp.BIC_}")
print(f"Weibull model (without covariates) - AIC: {weibull.AIC_}, BIC: {weibull.BIC_}")


# %%
# ---- PIT exp model: 

#cdf exp
lambda_exp = 1/exp.params_.values[0]
cdf_exp = 1 - np.exp(-lambda_exp * test_df["T"])
pdf_exp = exp.density_

# Plot histogram for CDF-values
plt.hist(cdf_exp, bins=50, alpha=0.7, color='orange', density=True)
plt.title("CDF-values Exponential Model", fontsize = 20)
plt.xlabel("CDF-values", fontsize = 18)
plt.ylabel("Density", fontsize = 18)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.tight_layout()
plt.show()



#%%
# ----  CRPS exp:

# Step 1: Calculate lambda for the exponential model
lambda_exp = 1 / exp.params_.values[0]

# Step 2: Create a time grid (from 0 to max T in test_df)
time_grid = np.linspace(0, test_df["T"].max(), 1000)  # 1000 points from 0 to max T

# Step 3: Calculate CDF on the time grid for the exponential model
cdf_exp_grid = 1 - np.exp(-lambda_exp * time_grid)

# Step 4: Initialize list to store CRPS values
crps_values = []

# Step 5: Calculate CRPS for each true event time in test_df["T"]
for true_T in test_df["T"]:
    # Step 6: Compute the indicator function: 1 if t >= T else 0
    indicator = (time_grid >= true_T).astype(float)

    # Step 7: Compute squared differences between CDF and indicator function
    squared_diffs = (cdf_exp_grid - indicator) ** 2

    # Step 8: Approximate the integral (CRPS) using the trapezoidal rule
    crps = trapezoid(squared_diffs, time_grid)
    
    # Append the CRPS for this T value
    crps_values.append(crps)

# Step 9: Calculate the mean CRPS over all T values
mean_crps = np.mean(crps_values)

print(f"Mean CRPS over all points, exp: {mean_crps:.4f}")


# %%

# ---- PIT Weibull wo cov:

pdf_weibull = weibull.density_
cdf_weibull = 1 - np.exp(- (test_df['T'] / weibull.params_[0])**weibull.params_[1])

# Plot histogram for CDF-verdiene
plt.hist(cdf_weibull, bins=50, alpha=0.7, color='red', density=True)
plt.title("CDF-values Weibull Model (w/o cov.)", fontsize = 20)
plt.xlabel("CDF-values", fontsize = 18)
plt.ylabel("Density", fontsize = 18)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.tight_layout()
plt.show()


#%%
#CRPS Weibull wo cov
# Step 1: Calculate lambda (scale) and k (shape) from the Weibull model
lambda_weibull = weibull.params_[0]  # Scale parameter
k_weibull = weibull.params_[1]       # Shape parameter

# Step 2: Create a time grid (from 0 to max T in test_df)
time_grid = np.linspace(0, test_df["T"].max(), 1000)  # 1000 points from 0 to max T

# Step 3: Calculate CDF on the time grid for the Weibull model
cdf_weibull_grid = 1 - np.exp(- (time_grid / lambda_weibull)**k_weibull)

# Step 4: Initialize list to store CRPS values
crps_values_weibull = []

# Step 5: Calculate CRPS for each true event time in test_df["T"]
for true_T in test_df["T"]:
    # Step 6: Compute the indicator function: 1 if t >= T else 0
    indicator = (time_grid >= true_T).astype(float)

    # Step 7: Compute squared differences between CDF and indicator function
    squared_diffs = (cdf_weibull_grid - indicator) ** 2

    # Step 8: Approximate the integral (CRPS) using the trapezoidal rule
    crps_weibull = trapezoid(squared_diffs, time_grid)
    
    # Append the CRPS for this T value
    crps_values_weibull.append(crps_weibull)

# Step 9: Calculate the mean CRPS over all T values
mean_crps_weibull = np.mean(crps_values_weibull)

print(f"Mean CRPS (Weibull) over all points: {mean_crps_weibull:.4f}")


# %%
