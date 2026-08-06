
import numpy as np
import matplotlib.pyplot as plt

def draw_circle(image, radius, color=(1, 0, 0)):
    h, w = image.shape[:2]
    cy, cx = h //2, w//2

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    dist = np.sqrt((xx-cx) ** 2 + (yy - cy) ** 2)

    mask = dist <=radius
    image[mask] = color
    return image


#write a function to visualize an image that has a function in it. say a circle function.
image = np.random.rand(100, 200, 3) # create a random image of size 100x100 with 3 color channels (RGB)
image = draw_circle(image, radius=30)


plt.imshow(image)
plt.show()




