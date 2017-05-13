import numpy as np
from PIL import Image
import sys, os

faceExpressionDatabase = sys.argv[1] if len(sys.argv) > 1 else './data/hand'

def load_imgs():
    img_set = []
    for file_name in os.listdir(faceExpressionDatabase):
        if file_name.endswith('.png'):
            img_path = faceExpressionDatabase + os.sep + file_name
            img = np.array(Image.open(img_path))
            img_set.append(img)
    img_set = np.array(img_set)
    return img_set

def intrinsic_dimension(X, k1=6, k2=12,estimator='levina',trafo='std'):
    n = X.shape[0]
    if k1 < 1 or k2 < k1 or k2 >= n:
        print "error k1,k2"
        return

    X = X.copy().astype(float)

    # New array with unique rows   
    X = X[np.lexsort(np.fliplr(X).T)]

    if trafo is None:
        pass
    elif trafo == 'var':
        X -= X.mean(axis=0) # broadcast
        X /= X.var(axis=0) + 1e-7 # broadcast
    elif trafo == 'std':
        # Standardization
        X -= X.mean(axis=0) # broadcast
        X /= X.std(axis=0) + 1e-7 # broadcast
    else:
        raise ValueError("Transformation must be None, 'std', or 'var'.")
    
    # Compute matrix of log nearest neighbor distances
    X2 = (X**2).sum(1)
    
    distance = X2.reshape(-1, 1) + X2 - 2*np.dot(X, X.T) #2x br.cast
    distance.sort(1)

    distance[distance<=0] = 1e-7
    knnmatrix = .5 * np.log(distance[:, 1:k2+1])
    
    # Compute the ML estimate
    S = np.cumsum(knnmatrix, 1)
    indexk = np.arange(k1, k2+1) # broadcasted afterwards
    dhat = -(indexk - 2) / (S[:, k1-1:k2] - knnmatrix[:, k1-1:k2] * indexk)
    #print dhat[0]
    if estimator == 'levina':  
        # Average over estimates and over values of k
        no_dims = dhat.mean()
    if estimator == 'mackay':
        # Average over inverses
        dhat **= -1
        dhat_k = dhat.mean(0)
        no_dims = (dhat_k ** -1).mean()

    return no_dims


if __name__ == '__main__':
    data = load_imgs()
    print "data shape:", data.shape
    data = data.reshape((-1, data[0].size))
    id_ = intrinsic_dimension(data, k1=150, k2=480, estimator='levina', trafo='var')
    print "Guess Intrinsic Dimension:",id_

