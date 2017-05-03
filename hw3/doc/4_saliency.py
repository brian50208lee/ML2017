from mytool import parser
from keras.models import load_model
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt


# load model adn predict
model = load_model('../model_best/063834.h5')

# load data
X, Y = parser.parse('../data/train_small.csv')
private_pixels = X/255
private_pixels = [ private_pixels[i].reshape((1, 48, 48, 1)) for i in range(len(private_pixels)) ]
#te_labels = np.argmax(Y[-1000:],axis=1)



def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

input_img = model.input
img_ids = range(20)

for idx in img_ids:
	val_proba = model.predict(private_pixels[idx])
	pred = val_proba.argmax(axis=-1)
	print "idx:{},predict:{}".format(idx,pred)
	target = K.mean(model.output[:, pred])
	grads = K.gradients(target, input_img)[0]
	#grads = normalize(grads)
	fn = K.function([input_img, K.learning_phase()], [target, grads])



	target_value, grads_value = fn([private_pixels[idx], 0])
	heatmap = grads_value.reshape((48,48))
	heatmap -= np.min(heatmap)
	heatmap /= np.max(heatmap)
	#np.set_printoptions(threshold=np.nan)
	#print heatmap

	thres = 0.60
	see = private_pixels[idx].reshape(48, 48).copy()
	see[np.where(heatmap <= thres)] = np.mean(see)

	plt.figure()
	plt.imshow(private_pixels[idx].reshape(48, 48), cmap='gray')
	plt.colorbar()
	plt.tight_layout()
	fig = plt.gcf()
	plt.draw()
	fig.savefig('4_{}.png'.format(idx), dpi=100)

	plt.figure()
	plt.imshow(heatmap, cmap=plt.cm.jet)
	plt.colorbar()
	plt.tight_layout()
	fig = plt.gcf()
	plt.draw()
	fig.savefig('4_{}_heatmap.png'.format(idx), dpi=100)

	plt.figure()
	plt.imshow(see,cmap='gray')
	plt.colorbar()
	plt.tight_layout()
	fig = plt.gcf()
	plt.draw()
	fig.savefig('4_{}_mask.png'.format(idx), dpi=100)

