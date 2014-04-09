import io
import json
from math import log, exp
import os
from os.path import join
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import time
from sklearn.datasets import make_blobs, make_multilabel_classification, fetch_20newsgroups_vectorized
from sklearn.linear_model import SGDClassifier
import sys
from entropy_maximization.entropy_maximization_sgd_fast import entropy_max_sgd_s

import logging
logger = logging.getLogger(__name__)

class csr_matrix_weights(csr_matrix):
    def __init__(self, csr_sparse_matrix, shape=None, weights=None):
        print 'type of csr_sparse_matrix: {}'.format(type(csr_sparse_matrix))

        if weights is not None:
            csr_matrix.__init__(self,(csr_sparse_matrix.data, csr_sparse_matrix.indices, csr_sparse_matrix.indptr), shape=csr_sparse_matrix.shape, dtype=csr_sparse_matrix.dtype)
            self.weights = np.asarray(weights, dtype=np.float32)
            if self.weights.shape[0] != self.shape[0]:
                raise Exception('problem with size of weights and matrix. Weights %d, matrix %d' % (self.weights.shape[0],self.shape[0]))
        else:
            csr_matrix.__init__(self,csr_sparse_matrix, shape=shape, dtype=csr_sparse_matrix[0].dtype)
            self.weights = None

    def __getitem__(self, indices):
        #print 'indecise : {}'.format(indices)
        weights_local = self.weights
        self.weights = None
        out = csr_matrix.__getitem__(self, indices)
        if not isinstance(out,csr_matrix_weights):
            #raise Exception('type was {}'.format(type(out)))
            out = csr_matrix_weights(out)

        print 'type of csr_sparse_matrix: {}, previous: {}'.format(type(out), type(self))
        self.weights = weights_local

        new_weights = None
        if weights_local is not None :
            if isinstance(indices, tuple):
                indices = indices[0]
            new_weights = weights_local[indices]

        out.weights = new_weights
        if new_weights is not None:
            assert(out.shape[0] == new_weights.shape[0])
        return out
        #return csr_matrix_weights(out, weights=new_weights)




class SGDEntropyMaximization(BaseEstimator, ClassifierMixin):
    def __init__(self, penalty='l2', alpha=0.0001, l1_ratio=0.15, n_iter=5,
                 learning_rate='optimal', eta0=0.001, power_t=0.5,
                 warm_start=False, verbose=False, intercept_decay=0.01,
                 early_stop=False,
                 train_fraction=0.9,
                 patience_increase=1.4,
                 improvement_threshold=0.995, min_epoch=10):
        self.n_iter = n_iter
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.W = None
        self.intercepts = None
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.penalty = penalty
        self.eta0 = eta0
        self.power_t = power_t
        self.warm_start = warm_start
        self.verbose = verbose
        self.early_stop = early_stop
        self.train_fraction = train_fraction
        self.patience_increase = patience_increase
        self.improvement_threshold = improvement_threshold
        self.intercept_decay = intercept_decay
        self.fit_time = None
        self.min_epoch = min_epoch

        if penalty != 'l2':
            raise Exception('other penalty than l2 are not implemented yet')


    def new(self):
        return SGDEntropyMaximization(penalty=self.penalty,
                                      alpha=self.alpha,
                                      l1_ratio=self.l1_ratio,
                                      n_iter=self.n_iter,
                                      learning_rate=self.learning_rate,
                                      eta0=self.eta0,
                                      power_t=self.power_t,
                                      warm_start=self.warm_start,
                                      verbose=self.verbose,
                                      intercept_decay=self.intercept_decay,
                                      early_stop=self.early_stop,
                                      train_fraction=self.train_fraction,
                                      patience_increase=self.patience_increase,
                                      improvement_threshold=self.improvement_threshold,
                                      min_epoch=self.min_epoch)


    def dump(self, folder):
        try:
            os.mkdir(folder)
        except:
            pass

        out = {
            'type': 'SGDEntropyMaximization',
            'penalty': self.penalty,
            'alpha': self.alpha,
            'l1_ratio': self.l1_ratio,
            'n_iter': self.n_iter,
            'learning_rate': self.learning_rate,
            'eta0': self.eta0,
            'power_t': self.power_t,
            'warm_start': self.warm_start,
            'verbose': self.verbose,
            'intercept_decay': self.intercept_decay,
            'fit_time': self.fit_time,
            'early_stop': self.early_stop,
            'train_fraction': self.train_fraction,
            'patience_increase': self.patience_increase,
            'improvement_threshold': self.improvement_threshold,
            'min_epoch': self.min_epoch
        }
        with io.open(join(folder, 'model.json'), 'wb') as model_file:
            json.dump(out, model_file)
        try:
            with io.open(join(folder, 'matrices.npz'), 'wb') as wb_file:
                np.savez(wb_file, **{str(i): param for i, param in enumerate([self.W, self.intercepts])})
        except:
            logger.info( 'cannot dump the matrices of the model in folder %s' % folder)

    @staticmethod
    def load(folder, tree_node_data=None):
        model_file_name = join(folder, 'model.json')
        with open(model_file_name, 'rb') as model_file:
            model = json.load(model_file)

        type = model.get('type')
        if type != 'SGDEntropyMaximization':
            raise Exception('wrong type %s') % type
        out = SGDEntropyMaximization(penalty=model.penalty,
                                     alpha=model.alpha,
                                     l1_ratio=model.l1_ratio,
                                     n_iter=model.n_iter,
                                     learning_rate=model.learning_rate,
                                     eta0=model.eta0,
                                     power_t=model.power_t,
                                     warm_start=model.warm_start,
                                     verbose=model.verbose,
                                     intercept_decay=model.intercept_decay,
                                     early_stop=model.early_stop,
                                     train_fraction=model.train_fraction,
                                     patience_increase=model.patience_increase,
                                     improvement_threshold=model.improvement_threshold,
                                     min_epoch=model.min_epoch)

        out.fit_time = model.get('fit_time')

        try:
            with io.open(join(folder, 'matrices.npz'), 'rb') as wb_file:
                nbz = np.load(wb_file)
                out.W, out.intercepts = [nbz[str(i)] for i in xrange(len(nbz.files))]
        except:
            logger.info( 'cannot load the matrices for the model in folder %s' % folder)

        return out


    def fit(self, X, y, **kwargs):
        y = np.asarray(y, dtype=np.int32)
        if self.warm_start and self.W is not None:
            return self._fit(X, y, W_initial=self.W, intercepts_initial=self.intercepts)
        else:
            return self._fit(X, y)

    def _fit(self, X, y, W_initial=None, intercepts_initial=None):

        min_y = np.min(y)
        max_y = np.max(y)

        if min_y < 0:
            raise Exception('the classes cannot be negative %d' % min_y)

        dtype = X.dtype

        if W_initial is None:
            n_classes = max_y + 1
        else:
            n_classes = W_initial.shape[1] + 1
            if max_y >= n_classes:
                raise Exception(
                    'the data contains classes that are not covered by the model, max_y %d, n_classes %d' % (
                        max_y, n_classes))

        if self.early_stop:
            num_train = int(X.shape[0] * self.train_fraction)
            train_set_x, train_set_y = X[:num_train], y[:num_train]
            valid_set_x, valid_set_y = X[num_train:], y[num_train:]
            X = train_set_x
            y = train_set_y
            n_points_valid = valid_set_x.shape[0]

        #print 'num classes %d' % n_classes
        n_features = X.shape[1]
        n_points = X.shape[0]

        xW = np.zeros(n_classes - 1, dtype=dtype)
        derivative = np.zeros(n_classes - 1, dtype=dtype)

        if W_initial is None:
            W_initial = np.zeros((n_features, n_classes - 1), dtype=dtype)
            intercepts_initial = np.zeros(n_classes - 1, dtype=dtype)
        else:
            if W_initial.shape != (n_features, n_classes - 1):
                raise Exception('the provided W doesnt match the expected size', W_initial.shape, n_features, n_classes)
            if intercepts_initial.shape != (n_classes - 1,):
                raise Exception('the provided intercepts have the wrong size', intercepts_initial.shape, n_classes)

        self.W = W_initial
        self.intercepts = intercepts_initial

        eta = self.eta0
        if self.learning_rate == 'optimal':
            typw = np.sqrt(1.0 / np.sqrt(self.alpha))
            # computing eta0, the initial learning rate TODO is it ok to have only 2 classes here
            tmp = np.zeros(2, dtype=dtype)
            log_soft_max_derivative(np.array([-typw, 0], dtype=dtype), 1, tmp)
            eta0 = typw / max(1.0, np.max(tmp))
            logger.info( 'eta0', eta0)
            # initialize t such that eta at first sample equals eta0
            t = 1.0 / (eta0 * self.alpha)
        else:
            t = 1
        count = 0
        start_t = time.time()
        sumloss = 0

        self.W_scal = 1

        best_accuracy = None
        patience = n_points * self.min_epoch

        for epoch in xrange(self.n_iter):
            logger.info( 'epoch %d' % epoch)
            for row in xrange(n_points):
            # if row % 1000 == 0:
            #   print 'working with row %d' % row

                xRow = self.sparse_vectory_weight_mul(xW, X, row)

                if self.verbose:
                    sumloss += log_soft_max(xW, y[row])

                if self.learning_rate == 'optimal':
                    eta = 1.0 / (self.alpha * t)

                elif self.learning_rate == 'invscaling':
                    eta = self.eta0 / pow(t, self.power_t)

                if self.penalty == 'l2':
                    self.l2_update(eta)

                log_soft_max_derivative(xW, y[row], derivative)
                derivative *= eta

                self.intercept_update(derivative)

                self.weights_update(xRow, derivative)

                t += 1
                count += 1

            if epoch % 5 == 0 and self.early_stop:
                correct_predictions = 0
                xW_valid = np.zeros(n_classes - 1, dtype=dtype)
                for row_validation in xrange(n_points_valid):
                    self.sparse_vectory_weight_mul(xW_valid, valid_set_x, row_validation)
                    m_valid = 0.0
                    guessed_class = n_classes - 1
                    for gi in xrange(n_classes - 1):
                        if xW_valid[gi] > m_valid:
                            guessed_class = gi
                            m_valid = xW_valid[gi]
                    if guessed_class == valid_set_y[row_validation]:
                        correct_predictions += 1
                accuracy_validation = (correct_predictions + 0.0) / n_points_valid

                sys.stdout.write('\nepoch %i, sample processed %i, validation accuracy %f %%, patience %d\n' % (
                    epoch, (count + 1), accuracy_validation * 100., patience))
                sys.stdout.flush()

                if best_accuracy is None or accuracy_validation * self.improvement_threshold > best_accuracy:
                    best_accuracy = accuracy_validation
                    best_W = self.W.copy()
                    best_intercepts = self.intercepts.copy()
                    patience = max(patience, count * self.patience_increase)
                    sys.stdout.write('\r\tpatience updated to %d, iteration %d' % (patience, count))
                    sys.stdout.flush()

                if patience <= count:
                    logger.info( '\ndone because patience %d, iteration %d' % (patience, count))
                    break

            if self.verbose:
                logger.info("Intercept: %s T: %d, Avg. loss: %.6f" % (self.intercepts, count, sumloss / count))

        self.W *= self.W_scal

        if self.early_stop:
            self.W = best_W
            self.intercepts = best_intercepts

        self.fit_time = time.time() - start_t
        if self.verbose:
            logger.info( 'optimization took %d seconds' % self.fit_time)

        return self

    def sparse_vectory_weight_mul(self, xW, X, row):
        xRow = X[row, :]
        xW[:] = (xRow * self.W) * self.W_scal + self.intercepts
        return xRow

    def l2_update(self, eta):
        self.W_scal *= (1.0 - eta * self.alpha)

    def intercept_update(self, derivative):
        self.intercepts += derivative * self.intercept_decay

    def weights_update(self, xRow, derivative):

        for pos in xrange(xRow.indptr[0], xRow.indptr[1]):
            ind = xRow.indices[pos]
            val = xRow.data[pos]
            self.W[ind, :] += derivative * (val / self.W_scal)


    def predict(self, X):
        last_class = self.W.shape[1]

        xW = X * self.W + self.intercepts
        m = np.max(xW, axis=1)
        a = np.argmax(xW, axis=1)

        def add_last_class(clazz, m):
            if m < 0:
                return last_class
            else:
                return clazz

        vfunc = np.vectorize(add_last_class)
        return vfunc(a, m)

    def predict_best_proba(self, x_data, **kwargs):
        xW = x_data * self.W + self.intercepts
        m = np.max(xW, axis=1)
        m = vmax(m)
        diff = xW - m[:, np.newaxis]
        diff = np.exp(diff)
        diff = np.sum(diff, axis=1) + np.exp(-m)
        return np.pow(diff, -1)


class SGDEntropyMaximizationFast(BaseEstimator, ClassifierMixin):
    def __init__(self, penalty='l2', alpha=0.0001, l1_ratio=0.15, n_iter=50,
                 learning_rate='optimal', eta0=0.01, power_t=0.5,
                 warm_start=False, verbose=True, intercept_decay=0.01,
                 early_stop=False,
                 train_fraction=0.9,
                 patience_increase=1.7,
                 improvement_threshold=0.99995, min_epoch=10):
        self.n_iter = n_iter
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.W = None
        self.intercepts = None
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.penalty = penalty
        self.eta0 = eta0
        self.power_t = power_t
        self.verbose = verbose
        self.early_stop = early_stop
        self.train_fraction = train_fraction
        self.patience_increase = patience_increase
        self.improvement_threshold = improvement_threshold
        self.intercept_decay = intercept_decay
        self.fit_time = None
        self.min_epoch = min_epoch

        if penalty != 'l2':
            raise Exception('other penalty than l2 are not implemented yet')


    def new(self):
        return SGDEntropyMaximizationFast(penalty=self.penalty,
                                          alpha=self.alpha,
                                          l1_ratio=self.l1_ratio,
                                          n_iter=self.n_iter,
                                          learning_rate=self.learning_rate,
                                          eta0=self.eta0,
                                          power_t=self.power_t,
                                          verbose=self.verbose,
                                          intercept_decay=self.intercept_decay,
                                          early_stop=self.early_stop,
                                          train_fraction=self.train_fraction,
                                          patience_increase=self.patience_increase,
                                          improvement_threshold=self.improvement_threshold,
                                          min_epoch=self.min_epoch)


    def dump(self, folder):
        try:
            os.mkdir(folder)
        except:
            pass

        out = {
            'type': 'SGDEntropyMaximizationFast',
            'penalty': self.penalty,
            'alpha': self.alpha,
            'l1_ratio': self.l1_ratio,
            'n_iter': self.n_iter,
            'learning_rate': self.learning_rate,
            'eta0': self.eta0,
            'power_t': self.power_t,
            'verbose': self.verbose,
            'intercept_decay': self.intercept_decay,
            'fit_time': self.fit_time,
            'early_stop': self.early_stop,
            'train_fraction': self.train_fraction,
            'patience_increase': self.patience_increase,
            'improvement_threshold': self.improvement_threshold,
            'min_epoch': self.min_epoch
        }
        with io.open(join(folder, 'model.json'), 'wb') as model_file:
            json.dump(out, model_file)
        try:
            with io.open(join(folder, 'matrices.npz'), 'wb') as wb_file:
                np.savez(wb_file, **{str(i): param for i, param in enumerate([self.W, self.intercepts])})
        except:
            logger.info( 'cannot dump the matrices of the model in folder %s' % folder)

    @staticmethod
    def load(folder, tree_node_data=None):
        model_file_name = join(folder, 'model.json')
        with open(model_file_name, 'rb') as model_file:
            model = json.load(model_file)
        type = model.get('type')
        if type != 'SGDEntropyMaximizationFast':
            raise Exception('wrong type %s') % type
        out = SGDEntropyMaximizationFast(penalty=model.get('penalty'),
                                         alpha=model.get('alpha'),
                                         l1_ratio=model.get('l1_ratio'),
                                         n_iter=model.get('n_iter'),
                                         learning_rate=model.get('learning_rate'),
                                         eta0=model.get('eta0'),
                                         power_t=model.get('power_t'),
                                         verbose=model.get('verbose'),
                                         intercept_decay=model.get('intercept_decay'),
                                         early_stop=model.get('early_stop'),
                                         train_fraction=model.get('train_fraction'),
                                         patience_increase=model.get('patience_increase'),
                                         improvement_threshold=model.get('improvement_threshold'),
                                         min_epoch=model.get('min_epoch'))

        out.fit_time = model.get('fit_time')

        try:
            with io.open(join(folder, 'matrices.npz'), 'rb') as wb_file:
                nbz = np.load(wb_file)
                out.W, out.intercepts = [nbz[str(i)] for i in xrange(len(nbz.files))]
        except:
            logger.info( 'cannot load the matrices for the model in folder %s' % folder)

        return out

    def fit(self, X, y, incremental=False, **kwargs):
        y = np.asarray(y, dtype=np.int32)

        start = time.time()

        #print 'd type x is %s' % X.dtype
        #print 'type of entry of x %s' % type(X[0, 0])
        if not incremental or self.W is None:
            self._fit(X, y)
        if (self.early_stop or incremental ) and self.W is not None:
            epochs_to_run = 10
            logger.info( 'running again for %d epochs with all the data' % epochs_to_run)
            default_epochs = self.n_iter
            default_early_stop = self.early_stop
            self.n_iter = epochs_to_run
            self.early_stop = False
            self._fit(X, y, W_initial=self.W, intercepts_initial=self.intercepts)
            self.n_iter = default_epochs
            self.early_stop = default_early_stop
        stop = time.time()
        self.fit_time = stop - start

        return self


    def _fit(self, X, y, W_initial=None, intercepts_initial=None):

        min_y = np.min(y)
        max_y = np.max(y)

        if min_y < 0:
            raise Exception('the classes cannot be negative %d' % min_y)

        dtype = X.dtype

        if W_initial is None:
            n_classes = max_y + 1
        else:
            n_classes = W_initial.shape[1] + 1
            if max_y >= n_classes:
                raise Exception(
                    'the data contains classes that are not covered by the model, max_y %d, n_classes %d' % (
                        max_y, n_classes))

        if self.early_stop:
            num_train = int(X.shape[0] * self.train_fraction)
            train_set_x, train_set_y = X[:num_train], y[:num_train]
            valid_set_x, valid_set_y = X[num_train:], y[num_train:]
            X = train_set_x
            y = train_set_y
            n_points_valid = valid_set_x.shape[0]

        #print 'num classes %d' % n_classes
        n_features = X.shape[1]
        n_points = X.shape[0]

        if W_initial is None:
            W_initial = np.zeros((n_features, n_classes - 1), dtype=dtype)
            intercepts_initial = np.zeros(n_classes - 1, dtype=dtype)
        else:
            if W_initial.shape != (n_features, n_classes - 1):
                raise Exception('the provided W doesnt match the expected size', W_initial.shape, n_features, n_classes)
            if intercepts_initial.shape != (n_classes - 1,):
                raise Exception('the provided intercepts have the wrong size', intercepts_initial.shape, n_classes)

        self.W = W_initial
        self.intercepts = intercepts_initial

        learning_rate_int = 1

        if self.learning_rate == 'optimal':
            typw = np.sqrt(1.0 / np.sqrt(self.alpha))
            # computing eta0, the initial learning rate TODO is it ok to have only 2 classes here
            tmp = np.zeros(2, dtype=dtype)
            log_soft_max_derivative(np.array([-typw, 0], dtype=dtype), 1, tmp)
            eta0 = typw / max(1.0, np.max(tmp))
            #print 'eta0', eta0
            # initialize t such that eta at first sample equals eta0
            t = 1.0 / (eta0 * self.alpha)
            learning_rate_int = 2
        else:
            t = 1
        start_t = time.time()


        if dtype == np.float32:
            logger.info('using floating point algo')
            algo = entropy_max_sgd_s
        else:
            logger.info('using double point algo')
            #algo = entropy_max_sgd
            raise Exception('double is not supported')

        weights = None
        weight_valid = None
        if hasattr(X, 'weights'):
            logger.info( 'using weights in stochastic gradient descent')
            weights = X.weights

            if self.early_stop:
                weight_valid = valid_set_x.weights

        verb = 0
        if self.verbose:
            verb = 1

        if not self.early_stop:
            if weights is not None:
                logger.info( 'type of weight: {}'.format(type(weights)))
                logger.info( 'shape of weights {}, shape of X {}'.format(weights.shape,X.shape))
            algo(self.W, self.intercepts, X.data, X.indices, X.indptr, y, weights, None, None, None, None, None,
                              n_features, n_points, n_classes, 0, np.float32(self.alpha), self.n_iter, verb, learning_rate_int,
                              np.float32(self.eta0), t, np.float32(self.intercept_decay))
        else:
            patience = n_points * self.min_epoch
            algo(self.W, self.intercepts, X.data, X.indices, X.indptr, y, weights, valid_set_x.data,
                              valid_set_x.indices, valid_set_x.indptr, valid_set_y, weight_valid, n_features, n_points, n_classes,
                              n_points_valid, np.float32(self.alpha), self.n_iter, verb, learning_rate_int, np.float32(self.eta0), t,
                              np.float32(self.intercept_decay), 1, self.improvement_threshold, self.patience_increase,patience)

        self.fit_time = time.time() - start_t
        if self.verbose:
            logger.info( 'optimization took %.4f seconds' % self.fit_time)

        return self


    def predict(self, X,  **kwargs):
        last_class = self.W.shape[1]

        #print 'shape X %s, shape W %s, shape intercepts %s' % (X.shape, self.W.shape, self.intercepts.shape)

        xW = X * self.W + self.intercepts
        m = np.max(xW, axis=1)
        a = np.argmax(xW, axis=1)

        def add_last_class(clazz, m):
            if m < 0:
                return last_class
            else:
                return clazz

        vfunc = np.vectorize(add_last_class)
        return vfunc(a, m)

    def predict_best_proba(self, x_data, **kwargs):
        xW = x_data * self.W + self.intercepts
        m = np.max(xW, axis=1)
        m = vmax(m , 0)
        diff = xW - m[:, np.newaxis]
        diff = np.exp(diff)
        diff = np.sum(diff, axis=1) + np.exp(-m)
        return 1.0/diff


vmax = np.vectorize(max)


def log_soft_max_derivative(input, j, derivative):
    m = max(np.max(input), 0)
    derivative[:] = input
    derivative -= m
    derivative[:] = np.exp(derivative)
    derivative *= -1.0 / (np.sum(derivative) + exp(-m))
    if j < input.shape[0]:
        derivative[j] += 1


def log_soft_max(input, j):
    k = 0
    if j < input.shape[0]:
        k = input[j]
    a = input - k
    m = max(np.max(a), -k)
    a -= m
    out = m + np.log(np.sum(np.exp(a)) + exp(-k - m))
    return out


def main():
    alpha = 0.001

    X, y = make_blobs(n_samples=50000, centers=2, random_state=0, cluster_std=0.60)
    #b = fetch_20newsgroups_vectorized()
    #X = b.get('data')
    #y = b.get('target')
    X = csr_matrix(X, dtype=np.float32)
    clf = SGDClassifier(loss="log", penalty="l2", verbose=10, alpha=alpha)
    clf.fit(X, y)

    print 'score other',clf.score(X, y)

    em = SGDEntropyMaximizationFast(verbose=True, alpha=alpha, early_stop=True, n_iter=200, min_epoch=100, learning_rate='fix')

    train_weights = np.asarray([ i for i in xrange(X.shape[0])],dtype=np.float32)

    print train_weights.shape

    Xw = csr_matrix_weights(X, weights=train_weights)

    em.fit(Xw, np.asarray(y, dtype=np.int32))
    print em.W
    print 'score', em.score(X, y)
    print np.min(em.predict_best_proba(X))

if __name__ == "__main__":
    main()
