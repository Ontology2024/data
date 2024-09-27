import matplotlib.pyplot as plt
import pandas as pd

file_path = 'freeze_data.csv'
data = pd.read_csv(file_path)

# Step 4: Visualize the normalized_est_score on a scatter plot based on latitude and longitude
plt.figure(figsize=(10, 6))

# Create scatter plot with color representing the normalized_est_score
scatter = plt.scatter(data['longitude'], data['latitude'], c=data['normalized_est_score'], cmap='viridis', alpha=0.7)

# Add color bar for reference
plt.colorbar(scatter, label='Normalized Estimated Score')

# Label the plot
plt.title('Normalized Estimated Score by Latitude and Longitude')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Display the plot
plt.show()