import os
import random
import cv2

# Path to the main folder
main_folder = "hair"

# Subfolder names
subfolders = ["squad", "rectangle", "ovar"]

# Get a random subfolder
random_subfolder = random.choice(subfolders)

# Get a random image from the subfolder
random_image = random.choice(os.listdir(os.path.join(main_folder, random_subfolder)))

# Path to the random image
image_path = os.path.join(main_folder, random_subfolder, random_image)

# Read the image using OpenCV
image = cv2.imread(image_path)


print(image)
# Display the image
cv2.imshow("Random Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
