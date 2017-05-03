from mytool import parser
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
import numpy as np

def main():
    emotion_classifier = load_model('model2.h5')
    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[:])

    input_img = emotion_classifier.input
    name_ls = ["conv2d_33","conv2d_34","conv2d_35","conv2d_36"]
    collect_layers = [ K.function([input_img, K.learning_phase()], [layer_dict[name].output]) for name in name_ls ]


    X, Y = parser.parse('train_small.csv')
    private_pixels = X/255
    private_pixels = [ private_pixels[i].reshape((1, 48, 48, 1)) for i in range(len(private_pixels)) ]

    choose_id = 9
    photo = private_pixels[choose_id]
    for cnt, fn in enumerate(collect_layers):
        nb_filter = layer_dict[name_ls[cnt]].output.shape[-1]
        print "layer_name:{}\tfilter_num:{}".format(name_ls[cnt],nb_filter)
        im = fn([photo, 0]) #get the output of that layer
        fig = plt.figure(figsize=(14, 8))
        nb_filter = im[0].shape[3]
        for i in range(nb_filter):
            ax = fig.add_subplot(nb_filter/16, 16, i+1)
            ax.imshow(im[0][0, :, :, i], cmap='BuGn')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.tight_layout()
        fig.suptitle('Output of layer{} (Given image{})'.format(name_ls[cnt], choose_id))
        img_path = "5_conv_img_{}_{}.png".format(name_ls[cnt], choose_id)
        fig.savefig(img_path)

main()

