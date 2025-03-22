import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data preparation
years = [2017, 2018, 2019, 2020, 2021, 2022]

# Given data values
computation_capacity = [21.2, 125, 125, 312, 312, 1000]  # Computation Capacity
model_size = [0.34, 0.34, 0.34, 1.5, 175, 1000]  # Model Size
network_bandwidth = [100, 100, 100, 200, 400, 400]  # Network Bandwidth
switch_chip = [12.8, 12.8, 25.6, 25.6, 25.6, 51.2]  # Switch Chip Performance

# Color settings
colors = [
    (169/255, 111/255, 176/255),  # Computation Capacity
    (216/255, 160/255, 199/255),  # Model Size
    (247/255, 167/255, 181/255),  # Network Bandwidth
    (43/255, 48/255, 122/255)     # Switch Chip
]

# Calculate growth rates relative to 2017
def calculate_growth_rate(data):
    base_value = data[0]
    return [(value / base_value - 1) * 100 for value in data]

growth_computation_capacity = calculate_growth_rate(computation_capacity)
growth_model_size = calculate_growth_rate(model_size)
growth_network_bandwidth = calculate_growth_rate(network_bandwidth)
growth_switch_chip = calculate_growth_rate(switch_chip)

# Create a DataFrame
data = {
    'Year': years * 4,
    'Metric': ['Computation Capacity'] * 6 + ['Model Size'] * 6 + ['Network Bandwidth'] * 6 + ['Switch Chip'] * 6,
    'Growth Rate (%)': growth_computation_capacity + growth_model_size + growth_network_bandwidth + growth_switch_chip
}

df = pd.DataFrame(data)

# Set Seaborn style
sns.set(style="whitegrid")

# Create figure and axis
plt.figure(figsize=(3, 2))

# Plot each metric trend with lines connecting the points using Seaborn's lineplot
palette = dict(zip(['Computation Capacity', 'Model Size', 'Network Bandwidth', 'Switch Chip'], colors))
sns.lineplot(x='Year', y='Growth Rate (%)', hue='Metric', data=df, palette=palette, marker='o')

# Add title and labels
plt.title('Growth Trends of Technical Metrics from 2017 to 2022', fontsize=12)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Growth Rate (%)', fontsize=14)

# Show legend
plt.legend(fontsize=8, loc='upper left')

# Set y-axis to logarithmic scale for better display of different magnitude data
plt.yscale('log')

# Adjust layout to prevent clipping of labels
plt.tight_layout()

# Save chart to PDF file
plt.savefig('growth_trends.pdf')

# Close the figure to free up memory
plt.close()
