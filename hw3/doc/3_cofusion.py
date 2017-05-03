'''
1.pip install sklearn
2.pip install matplotlib
3.echo backend: TkAgg > ~/.matplotlib/matplotlibrc
'''

from mytool import parser
import itertools
import numpy as np
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,title='Confusion matrix',cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



#np.set_printoptions(precision=2)

# load model adn predict
emotion_classifier = load_model('model2.h5')

# load data
X, Y = parser.parse('../data/train.csv')
dev_feats = X[-1000:]/255
te_labels = np.argmax(Y[-1000:],axis=1)

# predict
predictions = emotion_classifier.predict_classes(dev_feats)

# draw confusion
conf_mat = confusion_matrix(te_labels,predictions)
plt.figure()
plot_confusion_matrix(conf_mat, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])
plt.show()
