import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
import numpy as np

NUM_STEPS = 100
RECORD_FREQ = 10

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-7)

def grad_ascent(num_step,input_image_data,iter_func):
    filter_images = []
    for step in xrange(num_step):
        target_value, grads_value = iter_func([input_image_data])
        #print "step:{}\ttarget_value:{}".format(step,target_value)
        input_image_data += grads_value * 0.1
        if step % RECORD_FREQ == 0:
            filter_images.append([input_image_data.copy().reshape(48,48), target_value])
    return filter_images

def main():
    emotion_classifier = load_model('model2.h5')
    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[:])
    input_img = emotion_classifier.input

    name_ls = ["conv2d_33","conv2d_34","conv2d_35","conv2d_36"]
    #name_ls = layer_dict.keys
    collect_layers = [ layer_dict[name].output for name in name_ls ]

    for cnt, c in enumerate(collect_layers):
        nb_filter = collect_layers[cnt].shape[-1]
        print "layer_name:{}\tfilter_num:{}".format(name_ls[cnt],nb_filter)
        filter_imgs = [[] for i in range(NUM_STEPS//RECORD_FREQ)]
        for filter_idx in range(nb_filter):
            print "filter_idx:{}".format(filter_idx)
            input_img_data = np.random.random((1, 48, 48, 1)) # random noise
            target = K.mean(c[:, :, :, filter_idx])
            grads = normalize(K.gradients(target, input_img)[0])
            iterate = K.function([input_img], [target, grads])

            ###
            #"You need to implement it."
            num_step = NUM_STEPS
            records = grad_ascent(num_step, input_img_data, iterate)
            for i_record in range(len(filter_imgs)):
                filter_imgs[i_record].append(records[i_record])
            ###

        for it in range(NUM_STEPS//RECORD_FREQ):
            fig = plt.figure(figsize=(14, 8))
            for i in range(nb_filter):
                ax = fig.add_subplot(nb_filter/16, 16, i+1)
                ax.imshow(filter_imgs[it][i][0], cmap='BuGn')
                plt.xticks(np.array([]))
                plt.yticks(np.array([]))
                plt.xlabel('{:.3f}'.format(filter_imgs[it][i][1]))
                plt.tight_layout()
            fig.suptitle('Filters of layer {} (# Ascent Epoch {} )'.format(name_ls[cnt], it*RECORD_FREQ))
            img_path = '5_conv_{}_{}.png'.format(name_ls[cnt], it*RECORD_FREQ)
            print "save_ephch:{}".format(it*RECORD_FREQ)
            fig.savefig(img_path)

if __name__ == "__main__":
    main()