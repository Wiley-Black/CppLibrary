import numpy as np
import cv2

baboon = cv2.imread(r"data\baboon.png")
#cv2.imshow('image', baboon)

print(f"baboon.shape = {baboon.shape}")
gray = np.zeros((baboon.shape[0], baboon.shape[1]), dtype=np.float)
for yy in range(baboon.shape[0]):
    for xx in range(baboon.shape[1]):
        gray[yy,xx] = (float(baboon[yy,xx,0]) + float(baboon[yy,xx,1]) + float(baboon[yy,xx,2])) / 3.0
#cv2.imshow('gray', gray / 255.0)

print(f"mean value of gray image: {np.mean(gray)}")
ft = np.fft.fft2(gray)# / 255.0)
print(str(ft))

#left off here: need to write this to a file as a float, real and imag images.
ft_real = np.float32(np.real(ft))
ft_imag = np.float32(np.imag(ft))
print(f"real(ft).dtype = {ft_real.dtype}")
cv2.imwrite(r"data\\numpy_fft_baboon_real.tif", ft_real)
cv2.imwrite(r"data\\numpy_fft_baboon_imag.tif", ft_imag)

cv2.imshow('fft of gray', np.real(ft))

cv2.waitKey(0)
cv2.destroyAllWindows()

