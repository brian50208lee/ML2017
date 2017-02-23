import sys
from PIL import Image

# read
image1 = Image.open(sys.argv[1])
image2 = Image.open(sys.argv[2])
image3 = Image.new(image1.mode, image1.size)

# get pixel
pix_im1 = image1.load()
pix_im2 = image2.load()
pix_im3 = image3.load()

# compute diff
img_h, img_w = image3.size
for h in range(img_h):
	for w in range(img_w):
		pix_im3[h,w] = (0,0,0,0) if pix_im1[h,w] == pix_im2[h,w] else pix_im2[h,w]

# save
image3.save('ans_two.png','PNG')
