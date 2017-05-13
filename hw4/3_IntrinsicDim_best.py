import numpy as np
import sys

dataset_path = sys.argv[1] if len(sys.argv) > 1 else './data/data.npz'
output_file_name = sys.argv[2] if len(sys.argv) > 2 else './predict.csv'

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

    #print 'np.mean(distance):',np.mean(distance),'\t','log_normalize:',
    if np.mean(distance) > 0.5:
        #print 'True','\t'
        distance[distance<0] = 1e-7
        knnmatrix = .5 * np.log(distance[:, 1:k2+1])
    else:
        #print 'False','\t'
        # Replace invalid values with a small number
        distance[distance<0] = 1e-7
        knnmatrix = 2.5 * distance[:, 1:k2+1]
    
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

    return no_dims.round()


if __name__ == '__main__':
    data = np.load(dataset_path)
    out = open(output_file_name,'w')
    out.write('SetId,LogDim\n')
    for i in range(200):
        x = data[str(i)][:5000]
        #id_ = simple_predict(x)
        id_ = intrinsic_dimension(x,k1=6, k2=12, estimator='levina', trafo='var')
        print 'id: {}\tguess: {}'.format(i,id_)
        out.write('{},{}\n'.format(i,np.log(id_)))
    out.close()

