import random
from sklearn import datasets
import scipy
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.base import BaseEstimator
import sys
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
import convnet

from data import DataProvider, DataProviderException
from sklearn.cross_validation import train_test_split
from optparse import OptionParser
import numpy as np


class ScikitDataProvider(DataProvider):
    def __init__(self, data, batch_range=None, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        if batch_range == None:
            raise Exception('the range is empty')
        if init_batchnum is None or init_batchnum not in batch_range:
            init_batchnum = batch_range[0]

        self.data_dir = None
        self.batch_range = batch_range
        self.curr_epoch = init_epoch
        self.curr_batchnum = init_batchnum
        self.dp_params = dp_params
        self.batch_meta = None
        self.data_dic = None
        self.test = test
        self.batch_idx = batch_range.index(init_batchnum)

        print 'data is: {}'.format(len(data))
        self.X = data[0]
        self.y = data[1]

        self.fraction_test = 0.2

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                test_size=self.fraction_test,
                                                                                random_state=42)


    def get_batch(self, batch_num):
        if batch_num == 1:

            out = [_expand(self.X_train), _adjust_labels(self.y_train)]
        else:
            out = [_expand(self.X_test), _adjust_labels(self.y_test)]
            #print 'bartch shape x: {} , y: {}'.format(out[0].shape, out[1].shape)
        return out

    def get_data_dims(self, idx):
        return self.X.shape[1] if idx == 0 else 1


    def get_num_classes(self):
        return max(self.y) + 1


def _adjust_labels(labels):
    n = labels.shape[0]
    out = np.require(labels.reshape((1, n)), dtype=np.float32, requirements='C')
    print 'shape of labels is {}'.format(out.shape)
    return out


def _expand(matrix):
    if isinstance(matrix, csr_matrix):
        rows, cols = matrix.shape
        print "matrix output has size {}, {}".format(cols,rows)
     
        return [np.require(matrix.data, dtype=np.float32, requirements='C'),np.require( matrix.indices, dtype=np.int32, requirements='C'), np.require(matrix.indptr, dtype=np.int32, requirements='C'), cols, rows]
    else:
        print 'working with a dense matrix'
        out =  np.require(matrix.T, dtype=np.float32, requirements='C')
        print 'the shape is {}'.format(out.shape)
        return out


class ConvNetLearn(BaseEstimator):
    def __init__(self, layer_file, layer_params_file, output_folder="/tmp/convnet", epochs=400):
        print 'initializing the ConvNetLearn'
        self.layer_file = layer_file
        self.layer_params_file = layer_params_file
        self.output_folder = output_folder
        self.dict = {
            '--layer-def': layer_file,
            '--test-range': '2',
            '--data-path': None,
            '--train-range': '1',
            '--save-path': output_folder,
            '--layer-params': layer_params_file,
            '--test-freq': '13',
            '--data-provider': 'dp_scikit',
            '--epochs': epochs}


    def fit(self, X, y, **kwargs):
        print 'about to fit ConvNetLearn'
        op = convnet.ConvNet.get_options_parser()

        op.parse_from_dictionary(self.dict)
        op.eval_expr_defaults()

        class MyConvNet(convnet.ConvNet):
            def init_data_providers(self):
                self.dp_params['convnet'] = self
                try:
                    self.test_data_provider = DataProvider.get_instance([X, y], self.test_batch_range,
                                                                        type=self.dp_type, dp_params=self.dp_params,
                                                                        test=True)
                    self.train_data_provider = DataProvider.get_instance([X, y], self.train_batch_range,
                                                                         self.model_state["epoch"],
                                                                         self.model_state["batchnum"],
                                                                         type=self.dp_type, dp_params=self.dp_params,
                                                                         test=False)
                except DataProviderException, e:
                    print "Unable to create data provider: %s" % e
                    self.print_data_providers()
                    sys.exit()

        model = MyConvNet(op, load_dic=None)
        model.start()

    def predict(self, X):
        pass


def main():
    print 'starting'
    op = OptionParser()

    op.add_option("--layer_def", default='./example-layers/layers-logistic.cfg',
                  action="store", type=str, dest="layer_def",
                  help="Layer definition.")

    op.add_option("--layer_params", default='./example-layers/layer-params-logistic.cfg',
                  action="store", type=str, dest="layer_params",
                  help="The layer parameters file")

    (opts, args) = op.parse_args()

    newsgroup = False
    if newsgroup:
        categories = [
            'alt.atheism',
            'comp.graphics',
            'comp.os.ms-windows.misc',
            'comp.sys.ibm.pc.hardware',
            'comp.sys.mac.hardware',
            'comp.windows.x',
            'misc.forsale',
            'rec.autos',
            'rec.motorcycles',
            'rec.sport.baseball'
        ]
        # Uncomment the following to do the analysis on all the categories
        #categories = None

        print("Loading 20 newsgroups dataset for categories:")
        print(categories)

        data = fetch_20newsgroups(subset='all', categories=categories)
        print("%d documents" % len(data.filenames))
        print("%d categories" % len(data.target_names))

        ###############################################################################
        # define a pipeline combining a text feature extractor with a simple
        # classifier
        pipeline = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer())
        ])

        X = pipeline.fit_transform(data.data, data.target)
        y = data.target
        permutation = range(X.shape[0])
        random.shuffle(permutation)

        X = X[permutation]
        y = y[permutation]

        print "X type is {}".format(type(X))

    else:
        iris = datasets.load_iris()
        X = iris.data  # we only take the first two features.
        y = iris.target
        X = csr_matrix(X)

        print 'type if x is {}'.format(type(X))

    net = ConvNetLearn(layer_file=opts.layer_def, layer_params_file=opts.layer_params, epochs=20)



    net.fit(X, y)


if __name__ == "__main__":
    main()
