from optparse import OptionParser
from os.path import join

import numpy as np
from scipy.sparse import csr_matrix
from sklearn import preprocessing

from experiments.tree_data import TreeNodeData
from cudaconvnet.collect_images import ImageDataProvider
from scikit_learn_wrapper import ConvNetLearn


def main():
    op = OptionParser()

    op.add_option("--data_folder", default='/mnt/image_data',
                  action="store", type=str, dest="data_folder",
                  help="Product data file.")

    op.add_option("--batch_folder", default='/mnt/image_data/footwear_1361407141225_1361407159889_75',
                  action="store", type=str, dest="batch_folder",
                  help="Product data batch folder .")

    op.add_option("--product_type_file", default='/mnt/image_data/product_type.json',
                  action="store", type=str, dest="product_type_file",
                  help="Product type  file.")


    op.add_option("--layer_def", default='./example-layers/layers-80sec.cfg',
                  action="store", type=str, dest="layer_def",
                  help="Layer definition.")

    op.add_option("--layer_params", default='./example-layers/layer-params-80sec.cfg',
                  action="store", type=str, dest="layer_params",
                  help="The layer parameters file")

    op.add_option("--previous_model", default=None,
                  action="store", type=str, dest="previous_model",
                  help="Location of the model")


    (opts, args) = op.parse_args()


    scenario_folder = opts.data_folder

    batch_folder = opts.batch_folder

    tree_data = TreeNodeData(opts.product_type_file)

    with open(join(scenario_folder, 'matrices.npz'), 'rb') as wb_file:
            nbz = np.load(wb_file)
            X_train_data, X_train_indices, X_train_indptr, X_train_shape, y_train, X_test_data, X_test_indices, X_test_indptr, X_test_shape, y_test, products_ids_train, products_ids_test, product_image_urls_train, product_image_urls_test = [
                nbz[str(i)] for i in xrange(len(nbz.files))]
            X_train = csr_matrix((X_train_data, X_train_indices, X_train_indptr),
                                 shape=(X_train_shape[0], X_train_shape[1]))
            X_test = csr_matrix((X_test_data, X_test_indices, X_test_indptr), shape=(X_test_shape[0], X_test_shape[1]))

            print 'X_train %s, y_train %s, X_test %s, y_test %s' % (
            X_train.shape, len(y_train), X_test.shape, len(y_test))

    y_train = tree_data.l0_transform(y_train)

    le = preprocessing.LabelEncoder()
    le.fit(y_train)

    labelEncoder = LabelL0Encoder(tree_data,le )

    data_provider = ImageDataProvider(batch_folder, 'train', test_interval=15, label_transformer=labelEncoder, max_batch=3)


    num_classes = data_provider.get_num_classes()
    print 'num classes: {}'.format(num_classes)

    net = ConvNetLearn(layer_file=opts.layer_def, layer_params_file=opts.layer_params, epochs=2, fraction_test=0.05, last_model=opts.previous_model)

    #net.plot_predictions(data_provider,opts.previous_model)

    net.fit(data_provider, None, use_starting_point=True)

    sc = net.score(ImageDataProvider(batch_folder, 'train', test_interval=0, label_transformer=labelEncoder,max_batch=30), None)
    print 'score is {}'.format(sc)

class LabelL0Encoder:
    def __init__(self, tree_data, labelEncoder):
        self.tree_data = tree_data
        self.labelEncoder = labelEncoder

    def transform(self,y):
        l0 = self.tree_data.l0_transform(y)
        #print 'l0 {}'.format(l0)
        return self.labelEncoder.transform(l0)

    def get_num_classes(self):
        return len(list(self.labelEncoder.classes_))

    def get_label_names(self):
        return list(self.labelEncoder.classes_)

if __name__ == "__main__":
    main()