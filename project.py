# import libraries
import pandas as pd
from pyextremes import get_extremes, EVA
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("final final data.csv")

# Convert Date column to datetime and set it as index
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df.set_index('Date', inplace=True)

# Select the 'Fatalities' column
data = df["Fatalities"]

# Create a complete date range from the minimum to the maximum date and make it the index of the data, fill missing va;ues with 0
full_date_range = pd.date_range(start=data.index.min(), end=data.index.max(), freq="D")
data = data.reindex(full_date_range, fill_value=0)

data_filled = data.sort_index()  # Sort data by index
data_filled = data.dropna()  # Remove null values

# Print the filled data
print("Filled Data:")
print(data)

# Descriptive Statistics
print("\nDescriptive Statistics of Fatalities Data:")
print(data.describe())

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(data.index, data, label="Fatalities")
plt.xlabel("Date")
plt.ylabel("Fatalities")
plt.title("Time Series of Fatalities")
plt.legend()
plt.grid()
plt.show()

# Define cluster size for declustering and removing dependency
cluster = "24h"

# Define the threshold obtained from sensitivity analysis
U = 160

# Fit Generalized Pareto Distribution model
try:
    # Step 1: Extract extreme values using the POT method
    extremes = get_extremes(
        ts=data,  # Time series data
        method="POT",  # Peak Over Threshold method
        threshold=U,  
        r=cluster  
    )

    # Check if extremes are empty
    if extremes.empty:
        print("No extreme values found. Try lowering the threshold further.")
    else:
        # Step 2: Perform Extreme Value Analysis (EVA)
        eva = EVA(data=data)  # Create EVA object with the full dataset
        eva.get_extremes(  # Extract extremes
            method="POT",
            threshold=U,  
            r=cluster,
        )
        eva.fit_model(  # Fit GPD model
            model="MLE",  # Maximum Likelihood Estimation
        )
        
        # Extract GPD parameters
        fit_parameters=eva.model.fit_parameters
        Shape = fit_parameters['c']  # Shape parameter (ξ)
        Scale = fit_parameters['scale']  # Scale parameter (σ)
        print(f"\nLocation parameter = {U}") # location parameter is the threshold
        print(f"Scale parameter = {Scale}")
        print(f"Shape parameter = {Shape} >0 indicating that distribution is Frechet type with a heavy tail\n")

        # Plot model diagnostics
        eva.plot_diagnostic()  
        plt.show()  

        # Perform the Kolmogorov-Smirnov (KS) goodness of fit test
        ks_result = eva.test_ks()
        print("Kolmogorov-Smirnov (KS) Test Results:")
        print(ks_result)

        # Calculate VaR (Value at Risk) for a given confidence level
        confidence_level = 0.95 
        period=1 / (1-confidence_level)
        VaR = eva.get_return_value(return_period=period, return_period_size="7D")  
        VaR_value = VaR[0]  
        print(f"Value at Risk (VaR) at {confidence_level * 100}% confidence level and return period of {round(period)} weeks: {VaR_value}")
        
        # Calculate Expected Shortfall (ES) using the closed-form formula
        ES_formula = (VaR_value / (1 - Shape)) + ((Scale - Shape * U) / (1 - Shape))
        print(f"Expected Shortfall (ES) at {confidence_level * 100}% confidence level : {ES_formula}\n")


        

except Exception as e:
    print(f"Error in fitting GPD or plotting: {e}")
