import numpy as np
import os, sys
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from sklearn import preprocessing
import matlab.engine

class PerformICA():
    
    ## initialize all the necessary parameters
    #
    def __init__(self, avg = False, whitening = False, method = "gradient_ascent",
                 order = 2, n_iter = 1000, eta = 1.0, seed = 0):
        self.avg = avg
        self.whitening = whitening
        self.method = method
        self.order = order
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        self.mixing_mat = np.random.random((order,order))
        self.n_iter = n_iter
        self.eta = eta

        ## initialize variables for the selected methods
        #
        if method == "gradient_ascent":
            self.hs = np.zeros((self.n_iter, 1))
            self.gs = np.zeros((self.n_iter, 1))

            
        elif method == "maximum_likelihood":
            pass

        else:
            print "Invalid method for BSS!..."
            exit(1)



    ## get signal components
    #
    def est_components(self, unk_sig_1, unk_sig_2 = None):

        self.sig_len = len(unk_sig_1)
        
        if unk_sig_2 is not None:
            cat_sig = np.vstack((unk_sig_1, unk_sig_2))
        else:
            cat_sig = unk_sig_1.copy()

        mix_sig = np.dot(np.transpose(cat_sig), self.mixing_mat)

        self.bss_comp = np.zeros((self.sig_len, self.order))        

        self.est_inv_mixing_matrix(mix_sig)

#         self.bss_comp = np.dot(mix_sig, np.transpose(self.est_mat))
        self.bss_comp = np.dot(mix_sig, self.est_mat)

        ica = FastICA(n_components=2, algorithm = "deflation", max_iter = 10000)
        S_ica = ica.fit_transform(mix_sig)  # Reconstruct signals
        A_ = ica.mixing_  # Get estimated mixing matrix
        fast_ica_bss = np.dot(mix_sig, A_)

        return self.bss_comp, fast_ica_bss

    ## estimate the inverse of mixing matrix to get back unknown signals
    #
    def est_inv_mixing_matrix(self, mix_sig):

        mix_sig = self.preprocess_sig(mix_sig)
        
        ## initialize the mixing matrix
        #
        w = np.eye(self.order)

        ## initialize u the estimated source signals
        #
        u = np.dot(mix_sig, w)

        ## compute gradient ascent
        #
        self.est_mat = self.calc_entropy_gradient_ascent(mix_sig, w)



    def calc_entropy_gradient_ascent(self, in_sig, est_mat):
        
        for i in range(self.n_iter):

            ## get estimated source signals, u.
            u = np.dot(in_sig, est_mat)
            
            ## get maximum entropy signals U = cdf(u)
            max_entropy_u = np.tanh(u)
            
            ## find value of function h
            # h = log(abs(det(w))) + sum(log(eps+1-max_entropy_u[:] ** 2)) /N;

            det_w = abs(np.linalg.det(est_mat))
            h = ( ( 1/float(self.sig_len) ) * sum(sum(max_entropy_u)) + 0.5 * np.log(det_w) )

            ## Find the matrix of gradients
            g = np.linalg.inv(np.transpose(est_mat)) - reduce(np.dot, [(2/float(self.sig_len)), \
                np.transpose(in_sig), max_entropy_u])

            est_mat = est_mat + self.eta * g
            
            ## record h and magnitude of gradients
            #
            self.hs[i] = h
            self.gs[i] = np.linalg.norm(g[:])
            
        ## end of for
        #
        return est_mat
    ## end of method
    #


    def preprocess_sig(self, _sig):
        if self.avg:
            _sig = self.avg_sig(_sig)
        if self.whitening:
            _sig = self.whiten(_sig)

        return _sig

    def avg_sig(self, _sig):
        return _sig - np.mean(_sig)

    def whiten(self, _sig):
        sx = np.dot(np.transpose(_sig), _sig)
        d, v = np.linalg.eig(sx)
        x = reduce(np.dot, [v, np.sqrt(np.linalg.inv(np.diag(d))), np.transpose(v), np.transpose(_sig)])
        
        return np.transpose(x)


class PerformCCA():

    ## initialize all the necessary parameters
    #
    def __init__(self, comp_engine = None, avg = False, whitening = False,
                 method = "canonical", seed = 0):
        self.avg = avg
        print " averaging is: ", avg
        self.whitening = whitening
        self.method = method
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

        ## load the matlab engine if it is passed inthe argument
        #
        if comp_engine is not None:
            self.eng = comp_engine
        else:
            self.eng = None
        ## end of if
        #

    ## end of method
    #

    def est_components(self, set_x, set_y):

        self.sig_mean = np.mean(set_x)
        self.sig_var = np.var(set_x)
        
        if self.eng is not None:

            X = matlab.double(set_x.tolist())
            Y = matlab.double(set_y.tolist())

            [a_c, b_c, r_c, u_c, v_c] = self.eng.canoncorr(X,Y, nargout = 5)

            ## convert all the matlab arrays to a numpy array
            #
            u_c = np.asarray(u_c)
            v_c = np.asarray(v_c)

            ## calculate CCA mean and variances
            #
            self.cca_mean = np.mean(u_c)
            self.cca_var = np.var(u_c)

            return a_c, b_c, r_c, u_c, v_c

        else:
            set_x_sp = self.preprocess_sig(set_x)
            set_y_sp = self.preprocess_sig(set_y)

            if len(set_x_sp.shape) == 1:
                set_x_sp = np.reshape(set_x_sp, (set_x_sp.shape[0],1))
            if len(set_y_sp.shape) == 1:
                set_y_sp = np.reshape(set_y_sp, (set_y_sp.shape[0],1))
        
            p1 = len(set_x_sp)
            p2 = len(set_y_sp[1])

            [q1, r1] = np.linalg.qr(set_x_sp, mode = 'raw')
            rank1 = np.linalg.matrix_rank(r1)
            [q2, r2] = np.linalg.qr(set_y_sp, mode = 'raw')
            rank2 = np.linalg.matrix_rank(r2)
        
            min_rank = min(rank1, rank2)
            l, d, m = np.linalg.svd( np.dot(q1, q2), full_matrices=False )

            A = np.linalg.lstsq(r1, np.transpose(l), rcond=None)
            B = np.linalg.lstsq(r2, np.transpose(m), rcond=None)



    ## TODO: Update this for N-dimensional array, since the input signal sets could be
    ## of any size (especially for CCA)
    #
    def preprocess_sig(self, _sig):
        if self.avg:
            _sig = self.avg_sig(_sig)
        if self.whitening:
            _sig = self.whiten(_sig)

        return _sig

    def avg_sig(self, _sig):
        return _sig - self.sig_mean

    def whiten(self, _sig):
        sx = np.dot(np.transpose(_sig), _sig)
        d, v = np.linalg.eig(sx)
        x = reduce(np.dot, [v, np.sqrt(np.linalg.inv(np.diag(d))), np.transpose(v), np.transpose(_sig)])
        
        return np.transpose(x)

    ## this method rescales the preprocessed and eigen tricked signal to the original level
    #
    def rescale(self, processed_sig):

        processed_sig = np.sqrt(self.sig_var) * processed_sig

        return processed_sig

