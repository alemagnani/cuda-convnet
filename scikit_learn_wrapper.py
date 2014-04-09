import json
import os
import random
from os.path import join
import io
import shutil
from time import asctime, localtime, time
from sklearn import datasets
import scipy
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
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
#from entropy_maximization.entropy_maximization_sgd import SGDEntropyMaximizationFast
from gpumodel import IGPUModel
from layer import MyConfigParser
from ordereddict import OrderedDict
import scikit_data_provider
import shownet

import logging


logger = logging.getLogger(__name__)





class ConvNetLearn(BaseEstimator, ClassifierMixin):
    def __init__(self, layer_file, layer_params_file, output_folder="/tmp/convnet", epochs=400, fraction_test=0.01, mcp_layers=None, mcp_params=None, last_model=None):
        print 'initializing the ConvNetLearn'
        self.layer_file = layer_file
        self.layer_params_file = layer_params_file
        self.last_model = last_model
        self.fraction_test = fraction_test

        if mcp_layers is None:
            self.mcp_layers = MyConfigParser(dict_type=OrderedDict)
            self.mcp_layers.read([layer_file])

        else:
            self.mcp_layers = mcp_layers

        if mcp_params is None:
            self.mcp_params = MyConfigParser(dict_type=OrderedDict)
            self.mcp_params.read([layer_params_file])
        else:
            self.mcp_params = mcp_params

        self.output_folder = output_folder
        self.epochs = epochs
        self.dict = {
            '--layer-def': '',
            '--test-range': '2',
            '--data-path': None,
            '--train-range': '1',
            '--save-path': output_folder,
            '--layer-params': '',
            '--test-freq': '13',
            '--data-provider': 'dp_scikit',
            '--epochs': epochs}


    def new(self):
        return ConvNetLearn(layer_file=self.layer_file,layer_params_file=self.layer_params_file,output_folder=self.output_folder, epochs=self.epochs, mcp_layers=self.mcp_layers, mcp_params=self.mcp_params, fraction_test=self.fraction_test)


    def dump(self, folder):
        try:
            os.mkdir(folder)
        except:
            pass
        new_layer_file = join(folder, 'layer.cfg')
        new_layer_param_file = join(folder, 'layer_params.cfg')
        new_last_model = join(folder,'last_model')
        out = {
            'type': 'ConvNetLearn',
            'output_folder': self.output_folder,
            'epochs': self.epochs,
            'layer_file': new_layer_file,
            'layer_params_file': new_layer_param_file,
            'last_model': new_last_model,
            'fraction_test' : self.fraction_test
        }
        with io.open(join(folder, 'model.json'), 'wb') as model_file:
            json.dump(out, model_file)
        try:
            shutil.copyfile(self.layer_file, new_layer_file)
            shutil.copyfile(self.layer_params_file, new_layer_param_file)
            if self.last_model is not None:
                shutil.rmtree(new_last_model,ignore_errors=True)
                shutil.copyfile(self.last_model, new_last_model)

        except:
            logger.info( 'cannot copu stuff into  %s' % folder)

    @staticmethod
    def load(folder, tree_node_data=None):
        model_file_name = join(folder, 'model.json')
        with open(model_file_name, 'rb') as model_file:
            model = json.load(model_file)
        type = model.get('type')
        if type != 'ConvNetLearn':
            raise Exception('wrong type %s') % type
        out = ConvNetLearn(layer_file=model.get('layer_file'),layer_params_file=model.get('layer_params_file'),output_folder=model.get('output_folder'), epochs=model.get('epochs'), last_model=model.get('last_model'), fraction_test=model.get('fraction_test'))
        return out



    def fit(self, X, y, **kwargs):
        print 'about to fit ConvNetLearn'
        op = convnet.ConvNet.get_options_parser()

        op.parse_from_dictionary(self.dict)
        op.eval_expr_defaults()

        num_classes = y.max()+1
        logger.info('num classes {}'.format(num_classes))


        #we adjust the number of classes dynamically
        for name in self.mcp_layers.sections():
            if self.mcp_layers.has_option(name,'outputs'):
                if self.mcp_layers.get(name, 'outputs') == 'num_classes':
                    self.mcp_layers.set( name, 'outputs', value='{}'.format(num_classes))
        ##################




        print 'X shape {}, y shape {}'.format( X.shape,y.shape)
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=self.fraction_test,random_state=42)


        out_train = [scikit_data_provider.expand(X_train), scikit_data_provider.adjust_labels(y_train)]
        out_test = [scikit_data_provider.expand(X_test), scikit_data_provider.adjust_labels(y_test)]


        class MyConvNet(convnet.ConvNet):

            def __init__(self, op, load_dic, mcp_layers, mcp_params, fraction_test):
                  self.layer_def_dict = mcp_layers
                  self.fraction_test = fraction_test
                  self.epoch_count = 0

                  self.layer_params_dict = mcp_params
                  convnet.ConvNet.__init__(self,op,load_dic=load_dic,initialize_from_file=False)

            def get_data_dims(self, idx):
                return X.shape[1] if idx == 0 else 1

            def get_num_classes(self):
                return num_classes

            def get_next_batch(self, train=True):
                if train:
                    self.epoch_count += 1
                    return [self.epoch_count, 1 ,out_train]
                else:
                    return [self.epoch_count, 1, out_test]



            def init_data_providers(self):
                self.dp_params['convnet'] = self
                self.epoch_count = 0

        model = MyConvNet(op, load_dic=None, mcp_layers=self.mcp_layers, mcp_params=self.mcp_params, fraction_test=self.fraction_test)

        self.last_model = join(self.output_folder, model.save_file)
        print 'last model name {}'.format(self.last_model)
        model.start()

    def predict_proba(self, X):
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
                data_point =X.shape[0]
                data = [scikit_data_provider.expand(X), scikit_data_provider.adjust_labels( np.array([0] * data_point,dtype=np.float32))]
                num_ftrs = self.layers[self.ftr_layer_idx]['outputs']

                ftrs = np.zeros((data_point, num_ftrs), dtype=np.single)
                self.libmodel.startFeatureWriter(data + [ftrs], self.ftr_layer_idx)

                self.finish_batch()

                return ftrs


        model = MyConvNet(op, load_dic=load_dic)
        probs =  model.compute_probs(X)
        model.cleanup()
        return probs
    def predict(self,X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_best_proba(self, X, **kwargs):
        probs = self.predict_proba(X)
        return np.max(probs, axis=1)




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

    score = net.score(X, y)
    print 'score is {}'.format(score)



if __name__ == "__main__":
    main()
