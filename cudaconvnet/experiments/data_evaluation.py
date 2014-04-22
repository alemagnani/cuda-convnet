import gzip
from optparse import OptionParser
import os
from os.path import join
import cPickle
import shutil
import numpy as np
from sklearn import preprocessing
from cudaconvnet import ConvNetLearn

from tree_data import TreeNodeData


def main():
    op = OptionParser()

    op.add_option("--batch_folder", default='/Users/alessandro/Desktop/autotagData/PROD/productType/test1/batches',
                  action="store", type=str, dest="batch_folder",
                  help="Product data batch folder .")


    op.add_option("--product_type_file", default='/Users/alessandro/Desktop/autotagData/PROD/productType/test1/product_type.json',
                  action="store", type=str, dest="product_type_file",
                  help="Product type  file.")

    op.add_option("--output_folder", default='/Users/alessandro/Desktop/autotagData/PROD/productType/test1/batches_staged',
                  action="store", type=str, dest="output_folder",
                  help="Location of the output")


    (opts, args) = op.parse_args()


    output_folder = opts.output_folder
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)



class MultiModelEvaluation:
    def __init__(self, model_location_map):
        self.model_location_map = model_location_map


    def predict(self, data_provider):
        prediction_by_id = {}
        for id, model_location in self.model_location_map.iteritems():
            model, label_encoder = self.load_model(model_location)
            predictions = model.predict(data_provider)
            prediction_by_id[id] = label_encoder.inverse_transform(predictions)

        initial_predicitons = prediction_by_id.get(None)
        if initial_predicitons is None:
            raise Exception('the initial predictions is None no rootmodel specified probably')
        self._recursive_predictions(initial_predicitons, None, [i for i in xrange(len(initial_predicitons))],prediction_by_id)
        return initial_predicitons

    def _recursive_predictions(self, predictions, id, indices, predictions_by_id):
        indices_by_id = {}
        for i in indices:
            current_id = predictions[i]
            if current_id != id and current_id in predictions_by_id:
                if current_id not in indices_by_id:
                    indices_by_id[current_id] = []
                indices_by_id[current_id].append(i)
                predictions[i] = predictions_by_id[current_id][i]
        for child_id, child_indices in indices_by_id.iteritems():
            self._recursive_predictions(predictions, child_id, child_indices, predictions_by_id)




    def load_model(self, model_location):
        cuda_cond_data = join(model_location, 'model')
        with open(join(model_location,'label_encoder.p')) as encoder_file:
            label_encoder = cPickle.load(encoder_file)

        net = ConvNetLearn(layer_file=None, layer_params_file=None, epochs=None, fraction_test=None, last_model=cuda_cond_data)
        return net, label_encoder