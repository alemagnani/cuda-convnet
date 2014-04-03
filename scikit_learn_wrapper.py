import os
import random
from os.path import join
from sklearn import datasets
import scipy
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.base import BaseEstimator
import sys
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
import convnet
from sklearn.linear_model import SGDClassifier
from data import DataProvider, DataProviderException
from sklearn.cross_validation import train_test_split
from optparse import OptionParser
import numpy as np
from entropy_maximization.entropy_maximization_sgd import SGDEntropyMaximizationFast
from gpumodel import IGPUModel
import scikit_data_provider
import shownet







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
        self.last_model = join(self.output_folder,model.save_file)
        print 'last model name {}'.format(self.last_model)
        model.start()

    def predict(self, X):
        op = shownet.ShowConvNet.get_options_parser()

        predict_dict =  {
            '--write-features': 'probs',
            '--feature-path' : '/tmp/feature_path',
            '--test-range': '2',
            '--train-range': '1',
            '-f': self.last_model,
            '--data-provider': 'dp_scikit',
            '--show-preds' : '',
            '--multiview-test': 0,
            '--logreg-name': 'aaa'
            }

        op.parse_from_dictionary(predict_dict)
        load_dic = None
        options = op.options
        if options["load_file"].value_given:
            print 'load file option provided'
            load_dic = IGPUModel.load_checkpoint(options["load_file"].value)
            old_op = load_dic["op"]
            old_op.merge_from(op)
            op = old_op
        op.eval_expr_defaults()


        class MyConvNet(shownet.ShowConvNet):
            def init_data_providers(self):
                self.dp_params['convnet'] = self

            def compute_probs(self, X):
                if not os.path.exists(self.feature_path):
                    os.makedirs(self.feature_path)
                data_point =X.shape[0]
                data = [scikit_data_provider.expand(X),scikit_data_provider.adjust_labels( np.array([0] * data_point,dtype=np.float32))]
                num_ftrs = self.layers[self.ftr_layer_idx]['outputs']

                ftrs = np.zeros((data_point, num_ftrs), dtype=np.single)
                self.libmodel.startFeatureWriter(data + [ftrs], self.ftr_layer_idx)

                self.finish_batch()
                return ftrs


        model = MyConvNet(op, load_dic=load_dic)
        return model.compute_probs(X)


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

    newsgroup = True
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
            ('vect', CountVectorizer(max_features=1024*40,dtype=np.float32)),
            ('tfidf', TfidfTransformer())
        ])

        X = pipeline.fit_transform(data.data)
        X = X.astype(np.float32)
        y = data.target
        permutation = range(X.shape[0])
        random.seed(1)
        random.shuffle(permutation)
        print permutation[0:10]

        #permutation = permutation[0:15]

        X = X[permutation]
        #print X.data

        y = y[permutation]

        print "X type is {}, {}".format(type(X), X.shape)

    else:
        iris = datasets.load_iris()
        X = iris.data  # we only take the first two features.
        y = iris.target
        X = csr_matrix(X)


        print 'type if x is {}'.format(type(X))

    net = ConvNetLearn(layer_file=opts.layer_def, layer_params_file=opts.layer_params, epochs=100)
    #net = SGDEntropyMaximizationFast(verbose=True, alpha= 0.001, early_stop=False, n_iter=10, min_epoch=10, learning_rate='fix')

    #print 'fitting'
    #net =  SGDClassifier(loss="hinge", penalty="l2",  n_iter=50, verbose=10)
    #print 'done fitting'

    net.fit(X, y)

    probs = net.predict(X)
    print probs




if __name__ == "__main__":
    main()
