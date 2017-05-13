import os, sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# image dir
faceExpressionDatabase = sys.argv[1] if len(sys.argv) > 1 else './data/faceExpressionDatabase'
output_folder = sys.argv[2] if len(sys.argv) > 2 else './experiment'


def load_imgs():
	img_set = []
	for file_name in os.listdir(faceExpressionDatabase):
		if file_name.endswith('.bmp'):
			img_path = faceExpressionDatabase + os.sep + file_name
			img = np.array(Image.open(img_path))
			img_set.append(img)
	img_set = np.array(img_set)
	return img_set

def save_imgs(images, figsize, num_per_row, cmap, filename):
	fig = plt.figure(figsize=figsize)
	for idx in range(len(images)):
		img = images[idx]
		ax = fig.add_subplot(len(images) / num_per_row, num_per_row, idx+1)
		ax.imshow(img, cmap=cmap)
		plt.xticks(np.array([]))
		plt.yticks(np.array([]))
		plt.tight_layout()
	fpath = output_folder + os.sep + filename
	fig.savefig(fpath)

class PCA():
	def __init__(self):
		self.eigenvectors = None
		self.mean = None

	def fit(self, dataset):
		self.mean = np.mean(dataset, axis=0)

		norm = (dataset - self.mean)
		cov = np.cov(norm.T)
		eig_val, eig_vec = np.linalg.eigh(cov)

		eig_pairs = [ (np.abs(eig_val[idx]), eig_vec[:,idx]) for idx in range(len(eig_val)) ]
		eig_pairs.sort(reverse=True)

		_, self.eigenvectors = zip(*eig_pairs)
		self.eigenvectors = np.array(self.eigenvectors)

	def transform(self, dataset, new_dim):
		transform_dataset = (dataset - self.mean)
		transform_dataset = transform_dataset.dot(self.eigenvectors[:new_dim].T)
		return transform_dataset

	def inverse_transform(self, dataset):
		data_dim = dataset.shape[-1]
		reconstruct_dataset = dataset.dot(self.eigenvectors[:data_dim])
		reconstruct_dataset += self.mean
		return reconstruct_dataset

	def getEigenvectors(self):
		return self.eigenvectors.copy()

	def getMean(self):
		return self.mean.copy()

def RMSE(X, Y):
    RMSE_value = np.mean((X - Y)**2)
    RMSE_value = np.sqrt(RMSE_value)
    return RMSE_value


def Q1():
	# load image
	imgs = load_imgs().reshape((-1,75,64,64))[:10,:10,:,:].reshape((-1,64*64))

	# pca
	pca = PCA()
	pca.fit(imgs)

	# mean face
	mean_face = pca.getMean().reshape((64,64)).astype(np.uint8)
	Image.fromarray(mean_face).save(output_folder + os.sep + "1_mean_face.png")

	# eigen face
	eigen_faces = pca.getEigenvectors()[:9].reshape((-1,64,64))
	save_imgs(eigen_faces, figsize=(16,16), num_per_row=3, cmap='gray', filename='1_eigen_faces.png')

def Q2():
	# load image
	imgs = load_imgs().reshape((-1,75,64,64))[:10,:10,:,:].reshape((-1,64*64))

	# pca
	pca = PCA()
	pca.fit(imgs)

	# origin faces
	origin_faces = np.reshape(imgs, (-1,64,64))
	save_imgs(origin_faces, figsize=(16,16), num_per_row=10, cmap='gray', filename='1_origin_faces.png')

	# transform face
	transform_imgs = pca.transform(imgs, 5)
	reconsruct_imgs = pca.inverse_transform(transform_imgs).reshape((-1,64,64))
	save_imgs(reconsruct_imgs, figsize=(16,16), num_per_row=10, cmap='gray', filename='1_reconstruct_faces.png')

def Q3():
	# load image
	imgs = load_imgs().reshape((-1,75,64,64))[:10,:10,:,:].reshape((-1,64*64))

	# pca
	pca = PCA()
	pca.fit(imgs)

	# find RMSE rate < 0.01
	for top_k in range(len(imgs)):
		transform_imgs = pca.transform(imgs, top_k)
		reconsruct_imgs = pca.inverse_transform(transform_imgs)
		loss_rate_RMSE = RMSE(imgs, reconsruct_imgs) / 256
		if loss_rate_RMSE < 0.01:
			print "Ans: {}\tloss_rate_RMSE: {}".format(top_k, loss_rate_RMSE)
			break

if __name__ == '__main__':
	Q1()
	Q2()
	Q3()