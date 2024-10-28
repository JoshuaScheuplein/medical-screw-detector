import tifffile
import matplotlib.pyplot as plt


sample = "Ankle01_2"
# data = tifffile.imread(f"D:/2024-04-Scheuplein-Screw-Detection/{sample}/projections.tiff")
data = tifffile.imread(f"C:/Users/z003y7sd/Desktop/2024-04-Scheuplein-Screw-Detection/{sample}/projections.tiff")
print(data.shape)

plt.imshow(data[10], cmap='gray')
plt.title("Frame Visualization")
plt.colorbar()
plt.show()
