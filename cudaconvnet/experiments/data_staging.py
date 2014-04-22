import gzip
from optparse import OptionParser
import os
from os.path import join
import cPickle
import shutil
import numpy as np
import scipy
from sklearn import preprocessing

from tree_data import TreeNodeData


def main():
    op = OptionParser()

    op.add_option("--batch_folder", default='/data/sgeadmin/productTypeTest/batches',
                  action="store", type=str, dest="batch_folder",
                  help="Product data batch folder .")

    op.add_option("--product_type_file",
                  default='/data/sgeadmin/productTypeTest/product_type.json',
                  action="store", type=str, dest="product_type_file",
                  help="Product type  file.")

    op.add_option("--output_folder",
                  default='/data/sgeadmin/productTypeTest/batches_staged',
                  action="store", type=str, dest="output_folder",
                  help="Location of the output")

    (opts, args) = op.parse_args()

    output_folder = opts.output_folder
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    batch_folder = opts.batch_folder

    tree_data = TreeNodeData(opts.product_type_file)

    splits = find_splits(tree_data, max=400)

    print 'size of split is {}'.format(len(splits))

    create_split(splits, batch_folder, 'train', output_folder)
    create_split(splits, batch_folder, 'test', output_folder)


def create_split(splits, batch_folder, prefix, output_folder, batch_size=2048):


    staging_mappings = []
    for id, id_mapping in splits:
        if id is None:
            id = 'root'
        staging_mappings.append(StagingMapping(id, id_mapping, prefix, join(output_folder, id), batch_size=batch_size))

    count = 1
    while True:
        fname = join(batch_folder, '{}_{}.p.gz'.format(prefix, count))
        print 'reading file {}'.format(fname)
        count += 1
        if not os.path.isfile(fname):
            break
        with gzip.open(fname, 'rb') as f:
            X_batch, y_batch, image_data = cPickle.load(f)
            # print 'shape x batch {}, shape y batch {}, shape image data {}, type X {}'.format(X_batch.shape,
            #                                                                                   y_batch.shape,
            #                                                                                   image_data.shape,
            #                                                                                   type(X_batch))
            for staging_mapping in staging_mappings:
                staging_mapping.add_batch(X_batch, y_batch, image_data)

    for staging_mapping in staging_mappings:
        staging_mapping.finish()

    for staging_mapping in staging_mappings:
        if os.path.isdir(staging_mapping.output_folder):
            fname = '{}_{}.p.gz'.format('train', 'mean')
            try:
                shutil.copyfile(join(batch_folder, fname), join(staging_mapping.output_folder, fname))
            except:
                pass


class StagingMapping:
    def __init__(self, id, id_mapping, prefix, output_folder, batch_size=2048):
        self.id = id
        self.id_mapping = id_mapping
        self.prefix = prefix
        self.output_folder = output_folder
        self.batch_size = batch_size

        self.X = None
        self.y = None
        self.img = None
        self.count = 1
        self.output_count = 1
        self.distinct_label = set()

    def add_batch(self, X_batch, y_batch, image_data):
        # print 'shape x batch {}, shape y batch {}, shape image data {}, type X {}'.format(X_batch.shape, y_batch.shape,
        #                                                                                   image_data.shape,
        #                                                                                   type(X_batch))
        indices = [i for i in xrange(y_batch.shape[0]) if y_batch[i] in self.id_mapping]
        #print 'length indices {}'.format(len(indices))

        if len(indices) > 0:
            for p in indices:
                self.distinct_label.add(y_batch[p])
            if self.X is None:
                self.X = X_batch[indices]
                self.y = y_batch[indices]
                self.img = image_data[indices]
            else:
                pass
                # print 'shape x batch indices {}, X {}, type X {} , type X batch {}'.format(X_batch[indices].shape,
                #                                                                            self.X.shape, type(self.X),
                #                                                                            type(X_batch[indices]))
                self.X = scipy.sparse.vstack((self.X, X_batch[indices]), format='csr')
                self.y = np.vstack((self.y.reshape((self.y.shape[0], 1)), y_batch[indices].reshape((len(indices), 1))))
                self.img = np.vstack((self.img, image_data[indices]))
            #print 'shape x {}, shape y {}, shape image data {}'.format(self.X.shape, self.y.shape, self.img.shape)
        if self.X is not None and self.X.shape[0] >= self.batch_size:
            if not os.path.isdir(self.output_folder):
                os.mkdir(self.output_folder)
            with gzip.open(join(self.output_folder, '{}_{}.p.gz'.format(self.prefix, self.output_count)), 'wb') as f:
                new_y = self.y[:self.batch_size].reshape((self.batch_size,))
                #print 'new y shape {}'.format(new_y.shape)

                cPickle.dump([self.X[:self.batch_size], new_y, self.img[:self.batch_size]], f)
                if self.X.shape[0] > self.batch_size:
                    self.X = self.X[self.batch_size:]
                    self.y = self.y[self.batch_size:]
                    self.img = self.img[self.batch_size:]
                else:
                    self.X = None
                    self.y = None
                    self.img = None

            self.output_count += 1

    def finish(self):
        if self.X is not None:
            if not os.path.isdir(self.output_folder):
                os.mkdir(self.output_folder)
            with gzip.open(join(self.output_folder, '{}_{}.p.gz'.format(self.prefix, self.output_count)), 'wb') as f:
                cPickle.dump([self.X, self.y.reshape((self.y.shape[0],)), self.img], f)

        if self.prefix == 'train' and len(self.distinct_label) > 0:
            #print 'encoding {} labels'.format(len(self.distinct_label))
            le = preprocessing.LabelEncoder()
            le.fit([label for label in self.distinct_label])
            with open(join(self.output_folder, 'label_encoder.p'), 'wb') as f:
                cPickle.dump(le, f)


def find_splits(tree_node_data, max=300):
    m = {}
    splits = []
    for root_id in tree_node_data.find_root_ids():
        m[root_id] = root_id
        find_split_recursive(root_id, tree_node_data, splits, max=max)
        for child_child in tree_node_data.get_all_children_ids(root_id):
                m[child_child] = root_id
    splits.append((None, m))
    return splits


def find_split_recursive(id, tree_node_data, splits, max=300):
    children = tree_node_data.get_all_children_ids(id)
    if len(children) == 0:
        return
    print 'finding splits for {}'.format(id)
    if len(children) < max:
        print 'stopping at id {} with size of children {}'.format(id, len(children))
        m = {child: child for child in children}
        m[id] = id
        splits.append((id, m))
        return
    else:
        print 'expanding node {} with {} children ----------------------------------------------------'.format(id, len(
            children))
        m = {id: id}
        for direct_child in tree_node_data.get_children_ids(id):
            find_split_recursive(direct_child, tree_node_data, splits, max=max)
            m[direct_child] = direct_child
            for child_child in tree_node_data.get_all_children_ids(direct_child):
                m[child_child] = direct_child
        splits.append((id, m))


if __name__ == "__main__":
    main()