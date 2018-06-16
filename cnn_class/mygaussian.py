import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('odlotytest.jpg')
# plt.imshow(img)
# plt.show()

bw = img.mean(axis=2)

# plt.imshow(bw, cmap='gray')
# plt.show()

# W = np.zeros((20, 20))

# for i in range(20):
#     for j in range(20):
#         dist = (i - 9.5)**2 + (j - 9.5)**2
#         W[i, j] = np.exp(-dist / 50)

# plt.imshow(W, cmap='gray')
# # plt.show()

# out = convolve2d(bw, W)
# # plt.imshow(out, cmap='gray')
# # plt.show()

# # print(out.shape)
# # print(bw.shape)

# # out = convolve2d(bw, W, mode='same')
# # plt.imshow(out, cmap='gray')
# # plt.show()

# # print(out.shape)
# # print(bw.shape)

# out3 = np.zeros(img.shape)

# # normalize filter
# W /= W.sum()

# # restrict output to go from 0..1


# for i in range(3):
#     out3[:, :, i] = convolve2d(img[:, :, i], W, mode='same')

# out3 /= out3.max()
# plt.imshow(out3)
# plt.show()


Hx = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
], dtype=np.float32)

Hy = Hx.T

Gx = convolve2d(bw, Hx)

Gy = convolve2d(bw, Hy)

G = np.sqrt(Gx * Gx + Gy * Gy)

# plt.imshow(G, cmap='gray')
# plt.show()

theta = np.arctan2(Gy, Gx)
plt.imshow(G, cmap='gray')
# plt.show()
plt.savefig('odloty.png')
