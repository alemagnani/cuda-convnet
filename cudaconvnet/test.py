from optparse import OptionParser
from os.path import join
from scipy.sparse import csr_matrix
from sklearn import preprocessing
from scikit_learn_wrapper import ConvNetLearn
from tree_data import TreeNodeData
import numpy as np

def main():
    op = OptionParser()

    op.add_option("--data_folder", default='/home/ubuntu/test1',
                  action="store", type=str, dest="data_folder",
                  help="Product data file.")

    op.add_option("--product_type_file", default='/home/ubuntu/test1/product_type.json',
                  action="store", type=str, dest="product_type_file",
                  help="Product type  file.")


    op.add_option("--layer_def", default='./example-layers/layers-multi-perceptron-2.cfg',
                  action="store", type=str, dest="layer_def",
                  help="Layer definition.")

    op.add_option("--layer_params", default='./example-layers/layer-params-multi-perceptron-2.cfg',
                  action="store", type=str, dest="layer_params",
                  help="The layer parameters file")


    (opts, args) = op.parse_args()


    scenario_folder = opts.data_folder

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
    y_test = tree_data.l0_transform(y_test)

    le = preprocessing.LabelEncoder()
    le.fit(y_train)

    y_train_transformed = le.transform(y_train)
    y_test_transformed = le.transform(y_test)

    num_classes = np.max(y_train_transformed)+1
    print 'num classes: {}'.format(num_classes)

    net = ConvNetLearn(layer_file=opts.layer_def, layer_params_file=opts.layer_params, epochs=20,fraction_test=0.05)

    net.fit(X_train, y_train_transformed)

    score = net.score(X_test, y_test_transformed)

    print 'test score over {} points is: {}'.format(len(y_test),score)

if __name__ == "__main__":
    main()