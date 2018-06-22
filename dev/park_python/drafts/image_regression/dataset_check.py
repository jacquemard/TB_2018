import glob
from skimage import io

files = glob.glob("C:/DS/PKLot/PKLot/PKLot/UFPR05_processed_splitted/*/*/*.bmp")

shapes = []

for f in files:
    i = io.imread(f)
    
    if i.shape not in shapes:
        shapes.append(i.shape)
        print(i.shape)