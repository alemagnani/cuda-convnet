from optparse import OptionParser
from os.path import join
import numpy as np
from scipy.sparse import csr_matrix
from sklearn import preprocessing
from cudaconvnet import ConvNetLearn
import json

# Simple caching
def cache(function):
    cache = {}

    def wrapper(*args):
        if args in cache:
            return cache[args]
        else:
            rv = function(*args)
            cache[args] = rv
        return rv

    return wrapper

class TreeNodeData(object):
    def __init__(self,tree_filename,id_to_node=None):
        self.id_to_node = {}
        self.id_to_depth = {} #cache of depth
        if tree_filename is not None:
            with open(tree_filename, 'rb') as inputfile:
                for line in inputfile:
                    type = json.loads(line)
                    id = type.get('id')
                    parentId = type.get('parentId')
                    if parentId is not None:
                        if len(parentId) == 0:
                            type['parentId'] = None
                    if not id:
                        raise Exception('problem with id in row %s' % line)
                    self.id_to_node[id] = type
        elif id_to_node is not None:
            self.id_to_node = id_to_node
        else:
            raise Exception("cannot have both filename and id to none empty")

    def __iter__(self):
        return self.id_to_node.itervalues()

    def size(self):
        return len(self.id_to_node)

    def get_tree_node(self, id):
        return self.id_to_node.get(id)

    def get_tree_node_name(self,id):
        tmp = self.get_tree_node(id)
        if tmp is None:
            print 'problem of getting tree node name of node %s' % id
            return None
        else:
            return tmp.get('name')

    def get_path_name(self, tree_node_id):
        path = self.get_path(tree_node_id)
        return [self.get_tree_node(id).get('name') for id in path]

    @cache
    def get_path(self,tree_node_id):
        path = []
        path.append(tree_node_id)
        node = self.get_tree_node(tree_node_id)
        if node is None:
            raise Exception('the id %s was not found' % tree_node_id)
        parentId = node.get('parentId')
        while parentId is not None :
            path.insert(0,parentId)
            node = self.get_tree_node(parentId)
            if not node:
                raise Exception('the id %s was not found' % parentId)
            parentId = node.get('parentId')
        return path

    def get_children_ids(self, tree_node_id):
        #inefficient code TODO
        out = []
        for node in self:
            parentId = node.get('parentId')
            if parentId is not None and parentId == tree_node_id:
                out.append(node.get('id'))
        return out

    def get_all_children_ids(self, tree_node_id):
        out = []
        for children_id in self.get_children_ids(tree_node_id):
            out.append(children_id)
            out.extend(self.get_all_children_ids(children_id))
        return out

    def get_depth(self,tree_node_id):
        depth = self.id_to_depth.get(tree_node_id)
        if depth:
            return depth
        depth = 0
        node = self.get_tree_node(tree_node_id)
        if not node:
            raise Exception('the id %s was not found' % id)
        parentId = node.get('parentId')
        while parentId:
            depth += 1
            node = self.get_tree_node(parentId)
            if not node:
                raise Exception('the id %s was not found' % parentId)
            parentId = node.get('parentId')
        self.id_to_depth[tree_node_id] = depth
        return depth

    def l0_transform(self, y):
        out = []
        for id in y:
            try:
                l0 = self.get_path(id)[0]
            except:
                print 'cannot find id {}'.format(id)
                l0 = 'unknown'
            out.append(l0)
        return out

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