import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Data for calculation times
scales = ['2k', '4k', '8k', '16k', '32k']
lumoscore_times = [0.042, 0.581, 2.773, 15.213, 45.8]
minirewiring_times = [2.842, 28.581, 147.773, 684.213, 3298]

# Colors for each architecture
colors = {
    'LumosCore': (169/255, 111/255, 176/255),
    'MiniRewiring': (216/255, 160/255, 199/255)
}

# Create a bar plot with logarithmic y-axis
plt.figure(figsize=(8, 3))
bar_width = 0.35
index = np.arange(len(scales))

bars_lumoscore = plt.bar(index, lumoscore_times, bar_width, label='LumosCore', color=colors['LumosCore'], edgecolor='black')
bars_minirewiring = plt.bar(index + bar_width, minirewiring_times, bar_width, label='MiniRewiring', color=colors['MiniRewiring'], edgecolor='black')

plt.xlabel('Cluster Scale', fontsize=15)
plt.ylabel('Calculation Overhead (s)', fontsize=14.5)
plt.xticks(index + bar_width / 2, scales, fontsize=15)
plt.yscale('log')  # Set y-axis to log scale
plt.yticks(fontsize=10)
# Add legend with transparent background
legend = plt.legend(fontsize=15, framealpha=0)  # Set framealpha to 0 for transparency

# Add grid lines
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# Save the plot to a PNG file with transparent background
plt.savefig('calculation_comp.png', bbox_inches='tight', transparent=True, dpi=500)
plt.close()