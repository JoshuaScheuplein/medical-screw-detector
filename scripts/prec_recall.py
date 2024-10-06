import matplotlib.pyplot as plt

# Assume precision and recall are your data arrays
# Replace these with your actual precision and recall values
precision = [0.9, 0.85, 0.8, 0.75, 0.7]
recall = [0.1, 0.2, 0.3, 0.4, 0.5]

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', label='Precision-Recall Curve')

# Add labels and title
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()