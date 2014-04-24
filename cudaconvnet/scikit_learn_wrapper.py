import json
import os
import random
from os.path import join
import io
import shutil
from sklearn import datasets
from scipy.sparse import  csr_matrix
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import convnet
from sklearn.cross_validation import train_test_split
from optparse import OptionParser
import numpy as np
from gpumodel import IGPUModel
from layer import MyConfigParser
from ordereddict import OrderedDict
import scikit_data_provider
import shownet

import logging
import pylab as pl

logger = logging.getLogger(__name__)

class InMemorySplitDataProvider:
    def __init__(self,X, y, fraction_test=0.05):

        if fraction_test > 0.0:
            if y is not None:
                print 'X shape {}, y shape {}'.format( X.shape,y.shape)
                X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=fraction_test,random_state=42)
            else:
                print 'X shape {}'.format( X.shape)
                X_train, X_test = train_test_split(X, test_size=fraction_test,random_state=42)
                y_train = None
        else:
            X_train = X
            y_train = y
            X_test = None
            y_test = None

        self.x_size = X_train.shape[1]
        if y is not None:
            self.num_classes = np.max(y_train)+1
        else:
            self.num_classes = None

        if y is not None:
            self.out_train = [scikit_data_provider.expand(X_train), scikit_data_provider.adjust_labels(y_train)]
        else:
            self.out_train = [scikit_data_provider.expand(X_train), None]
        if X_test is not None:
            self.out_test = [scikit_data_provider.expand(X_test), scikit_data_provider.adjust_labels(y_test)]
        else:
            self.out_test = None
        self.epoch_count = 0
        self.epoch_count_test = 0

    def get_num_test_batches(self):
        return 1

    def get_next_batch(self, train=True):
                if train:
                    self.epoch_count += 1
                    return [self.epoch_count, 1 ,self.out_train]
                else:
                    self.epoch_count_test += 1
                    return [self.epoch_count_test, 1, self.out_test]
    def get_data_dims(self, idx):
        return self.x_size if idx == 0 else 1

    def get_num_classes(self):
        return self.num_classes

    def init_data_providers(self):
            self.epoch_count = 0
            self.epoch_count_test = 0

class ConvNetLearn(BaseEstimator, ClassifierMixin):
    def __init__(self, layer_file, layer_params_file, output_folder="/tmp/convnet", epochs=400, fraction_test=0.01, mcp_layers=None, mcp_params=None, last_model=None, init_states_models=None):
        print 'initializing the ConvNetLearn'
        self.layer_file = layer_file
        self.layer_params_file = layer_params_file
        self.last_model = last_model
        self.fraction_test = fraction_test
        self.init_states_models = init_states_models

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
            '--test-freq': '10',
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



    def fit(self, X, y, use_starting_point=True, **kwargs):
        print 'about to fit ConvNetLearn'
        if use_starting_point and self.last_model is not None:
            self.dict['-f']=self.last_model


        op = convnet.ConvNet.get_options_parser()
        op.parse_from_dictionary(self.dict)

        load_dic = None
        options = op.options
        if options["load_file"].value_given:
            print 'load file option provided'
            load_dic = IGPUModel.load_checkpoint(options["load_file"].value)

            name_to_weights = {}
            if self.init_states_models is not None:
                name_to_weights = {}
                for init_model in self.init_states_models:
                    load_dic_local = IGPUModel.load_checkpoint(init_model)

                    for k, v in load_dic_local['model_state'].iteritems():
                        if k == 'layers':
                            for elem in v:
                                name = elem.get('name')
                                weights =  elem.get('weights')
                                if weights is not None:
                                    print 'adding weights for layer {}'.format(name)
                                    if name not in name_to_weights:
                                        name_to_weights[name] = {}
                                    name_to_weights[name]['weights'] = weights
                                    name_to_weights[name]['biases'] = elem.get('biases')
                                    name_to_weights[name]['weightsInc'] = elem.get('weightsInc')
                                    name_to_weights[name]['biasesInc'] = elem.get('biasesInc')




            if len(name_to_weights) > 0:
                print 'layer names with init arrays: {}'.format(name_to_weights.keys())

                for k, v in load_dic['model_state'].iteritems():
                    if k == 'layers':
                        for elem in v:
                            name = elem.get('name')
                            print 'name of layer to possibly be updated {}'.format(name)
                            weights =  elem.get('weights')
                            if weights is not None:
                                if name in name_to_weights:
                                    print 'changing init point of model for layer {}'.format(name)
                                    coefs_name = name_to_weights.get(name)
                                    if coefs_name is None or 'weights' not in coefs_name:
                                        raise Exception('coeef names doent have weights for {}, coef names fields: {}'.format(name, coefs_name.keys()))
                                    elem['weights'] = coefs_name['weights']
                                    elem['biases'] = coefs_name['biases']
                                    elem['weightsInc'] = coefs_name['weightsInc']
                                    elem['biasesInc'] = coefs_name['biasesInc']




            old_op = load_dic["op"]
            old_op.merge_from(op)
            op = old_op
        op.eval_expr_defaults()


        try:
            self.dict.pop('-f')
        except:
            pass


        if hasattr(X, 'get_next_batch'):
            data_provider = X
        else:
            data_provider = InMemorySplitDataProvider(X,y,fraction_test=self.fraction_test)

        data_provider.init_data_providers()

        num_classes = data_provider.get_num_classes()
        logger.info('num classes {}'.format(num_classes))


        #we adjust the number of classes dynamically
        for name in self.mcp_layers.sections():
            if self.mcp_layers.has_option(name,'outputs'):
                if self.mcp_layers.get(name, 'outputs') == 'num_classes':
                    self.mcp_layers.set( name, 'outputs', value='{}'.format(num_classes))
        ##################


        class MyConvNet(convnet.ConvNet):

            def __init__(self, op, load_dic, mcp_layers, mcp_params, fraction_test):
                  self.layer_def_dict = mcp_layers
                  self.layer_params_dict = mcp_params
                  convnet.ConvNet.__init__(self,op,load_dic=load_dic,initialize_from_file=False)
                  self.test_one = True
                  self.epoch = 1
                  self.max_filesize_mb = 5000

            def get_data_dims(self, idx):
                return data_provider.get_data_dims(idx)

            def get_num_classes(self):
                return data_provider.get_num_classes()

            def get_next_batch(self, train=True):
                return data_provider.get_next_batch(train)

            def get_num_test_batches(self):
                return data_provider.get_num_test_batches()

            def init_data_providers(self):
                data_provider.init_data_providers()

        model = MyConvNet(op, load_dic=load_dic, mcp_layers=self.mcp_layers, mcp_params=self.mcp_params, fraction_test=self.fraction_test)

        self.last_model = join(self.output_folder, model.save_file)
        print 'last model name {}'.format(self.last_model)
        model.start()

    def predict_proba(self, X, train=True):
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

        if hasattr(X, 'get_next_batch'):
            data_provider = X
        else:
            data_provider = InMemorySplitDataProvider(X,None,fraction_test=0.0)

        data_provider.init_data_providers()

        class MyConvNet(shownet.ShowConvNet):
            def init_data_providers(self):
                self.dp_params['convnet'] = self

            def compute_probs(self, X):
                out = None

                while True:
                    data_all = data_provider.get_next_batch(train=train)
                    epoch, batch = data_all[0], data_all[1]
                    if epoch != 1:
                        break
                    print 'working on epoch: {}, batch: {}'.format(epoch, batch)
                    data = data_all[2]
                    if isinstance(data[0], list):
                         data_point = data[0][4]

                    else:
                        data_point = data[0].shape[1]
                    print 'data points {}'.format(data_point)
                    num_ftrs = self.layers[self.ftr_layer_idx]['outputs']

                    ftrs = np.zeros((data_point, num_ftrs), dtype=np.single)
                    self.libmodel.startFeatureWriter(data + [ftrs], self.ftr_layer_idx)

                    self.finish_batch()
                    if out is None:
                        out = ftrs
                    else:
                        out = np.vstack((out,ftrs))
                return out
        model = MyConvNet(op, load_dic=load_dic)
        probs = model.compute_probs(data_provider)
        model.cleanup()
        return probs

    def plot_filters(self, data_provider,show_filters='conv1',  output_file='/tmp/filters.png'):
        op = shownet.ShowConvNet.get_options_parser()

        predict_dict =  {
            '--show-filters' : show_filters,
            '--test-range': '2',
            '--train-range': '1',
            '--show-preds' : 'probs',
            '-f': self.last_model,
            '--data-provider': 'dp_scikit',
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
            def get_data_dims(self, idx):
                return data_provider.get_data_dims(idx)

            def get_num_classes(self):
                return data_provider.get_num_classes()

            def get_next_batch(self, train=True):
                return data_provider.get_next_batch(True)

            def get_num_test_batches(self):
                return data_provider.get_num_test_batches()

            def get_plottable_data(self, data):
                return data_provider.get_plottable_data(data)

            def init_data_providers(self):
                data_provider.init_data_providers()

            def get_label_names(self):
                return data_provider.get_label_names()

        model = MyConvNet(op, load_dic=load_dic)
        model.plot_filters()
        pl.savefig(output_file)
        model.cleanup()

    def plot_predictions(self, data_provider, output_file='/tmp/predictions.png', train=True, only_errors=True):
        op = shownet.ShowConvNet.get_options_parser()

        local_train = train
        predict_dict =  {
            '--write-features': 'probs',
            '--feature-path' : '/tmp/feature_path',
            '--test-range': '2',
            '--train-range': '1',
            '--show-preds' : 'probs',
            '-f': self.last_model,
            '--data-provider': 'dp_scikit',
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
            def get_data_dims(self, idx):
                return data_provider.get_data_dims(idx)

            def get_num_classes(self):
                return data_provider.get_num_classes()

            def get_next_batch(self, train=True):
                return data_provider.get_next_batch(local_train)

            def get_num_test_batches(self):
                return data_provider.get_num_test_batches()

            def get_plottable_data(self, data):
                return data_provider.get_plottable_data(data)

            def init_data_providers(self):
                data_provider.init_data_providers()

            def get_label_names(self):
                return data_provider.get_label_names()



        model = MyConvNet(op, load_dic=load_dic)
        model.only_errors = only_errors
        model.plot_predictions()
        pl.savefig(output_file)
        model.cleanup()

    def predict(self,X, train=False):
        probs = self.predict_proba(X, train=train)
        print 'size of probs {}'.format(probs.shape)
        out =  np.argmax(probs, axis=1)
        print 'size of out {}'.format(out.shape)
        return out

    def predict_best_proba(self, X, train=False, **kwargs):
        probs = self.predict_proba(X, train=train)
        return np.max(probs, axis=1)

    def score(self, X, y, train=False, type='accuracy'):
        if hasattr(X, 'get_next_batch'):
            data_provider = X
        else:
            data_provider = InMemorySplitDataProvider(X, y, fraction_test=0.0)
            train = True


        y = None
        data_provider.init_data_providers()
        while True:
                    data_all = data_provider.get_next_batch(train=train)
                    epoch, batch = data_all[0], data_all[1]
                    print 'epoch is {}'.format(epoch)
                    if epoch != 1:
                        break
                    print 'working on epoch: {}, batch: {}'.format(epoch, batch)
                    y_local = data_all[2][1].T
                    print 'y_local shape {}'.format(y_local.shape)
                    if y is None:
                        y = y_local
                    else:
                        y = np.vstack((y, y_local))
        print 'shape y {}'.format(y.shape)
        predictions = self.predict(X, train=train)
        if type == 'accuracy':
            return accuracy_score(y, predictions)
        elif type == 'f1macro':
            return sklearn.metrics.f1_score(y,predictions,average='macro')
        elif type == 'f1':
            return sklearn.metrics.f1_score(y,predictions,average='weighted')
        else:
            raise Exception('unrecognized type: {}'.format(type))


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

    net = ConvNetLearn(layer_file=opts.layer_def, layer_params_file=opts.layer_params, epochs=200)
    #net = SGDEntropyMaximizationFast(verbose=True, alpha= 0.001, early_stop=False, n_iter=10, min_epoch=10, learning_rate='fix')

    #print 'fitting'
    #net =  SGDClassifier(loss="hinge", penalty="l2",  n_iter=50, verbose=10)
    #print 'done fitting'

    net.fit(X, y)

    score = net.score(X, y)
    print 'score is {}'.format(score)



if __name__ == "__main__":
    main()
