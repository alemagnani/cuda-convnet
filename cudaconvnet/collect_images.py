import base64
import cStringIO
import gzip
from optparse import OptionParser
import os
import urllib

from PIL import Image
import cPickle
import datetime
from joblib import Parallel, delayed
import math
import numpy as np
from os.path import join
from pymongo import MongoClient
from scipy.sparse import csr_matrix
from scikit_data_provider import expand, adjust_labels
import pylab as pl
import random as nr

def transform(items, array_extractor, n_jobs=5, **kwargs):
    list_arrays = Parallel(n_jobs=n_jobs)(
        delayed(array_extractor)(item, **kwargs) for item in items)
    good_indices = [i for i, image in enumerate(list_arrays) if image is not None]

    return [list_arrays[i] for i in good_indices], good_indices
    #return np.array(list_arrays[good_indices],dtype=np.float32), good_indices


def process(image):
    image = np.array(image)  # 32 x 32 x 3
    image = np.rollaxis(image, 2)  # 3 x 32 x 32
    image = image.reshape(-1)  # 3072
    return image


#the image is deformed to fit the size given
def load_resize(file, size):
    #file = cStringIO.StringIO(urllib.urlopen(url).read())

    im = Image.open(file, mode='r')
    im = im.convert('RGB')
    out = im.resize(size, Image.ANTIALIAS)

    return out


def process_product_image(file, size=(32, 32)):
    try:
        im = load_resize(file, size)
    except Exception as e:
        #print 'cannot load image from {} because {}'.format(image_url,e)
        return None
    return process(im)


def process_batch(files, size=(32, 32), n_jobs=5):
    return transform(files, process_product_image, n_jobs=n_jobs, size=size)


class ImageDataProvider:
    def __init__(self, folder, prefix, test_interval=25, label_transformer=None, max_batch=None, crop_image=False, border_size=4, flip_image=True, bool_text=False):
        self.test_interval = test_interval
        self.current_train_index = 1
        self.current_test_index = self.test_interval
        self.prefix = prefix
        self.folder = folder
        self.epoch_count_train = 1
        self.epoch_count_test = 1
        mean_file = join(folder, '{}_{}.p.gz'.format('train', 'mean'))  # the mean can only be coming from the train data
        with gzip.open(mean_file, 'rb') as mf:
            self.mean = cPickle.load(mf)
        self.label_transformer = label_transformer
        self.test_batches = None


        self.img_size = int(math.sqrt(self.mean.shape[0]/3))
        print 'image size: {}'.format(self.img_size)
        self.max_batch = max_batch

        fname = join(self.folder, '{}_{}.p.gz'.format(self.prefix, '1'))
        with gzip.open(fname, 'rb') as f:
                X_batch, y_batch, image_data = cPickle.load(f)
        self.text_size = X_batch.shape[1]
        self.crop_image = crop_image
        self.border_size = border_size
        self.flip_image = flip_image
        self.bool_text = bool_text


    def get_num_test_batches(self):
        if self.test_batches is not None:
            return self.test_batches
        file_pos = self.test_interval

        while (True):
            fname = join(self.folder, '{}_{}.p.gz'.format(self.prefix, file_pos))
            if os.path.isfile(fname):
                file_pos += self.test_interval
            else:
                break

        file_pos -= self.test_interval
        self.test_batches = file_pos / self.test_interval
        print 'the number of test batches is {}'.format(self.test_batches)
        return self.test_batches

    def get_plottable_data(self, data):
        return np.require(
            (data + self.mean).T.reshape(data.shape[1], 3, self.img_size, self.img_size).swapaxes(1, 3).swapaxes(1,
                                                                                                                 2) / 255.0,
            dtype=np.single)


    def get_next_batch(self, train=True):
        if train:
            fname = join(self.folder, '{}_{}.p.gz'.format(self.prefix, self.current_train_index))
            if not os.path.isfile(fname) or (self.max_batch is not None and self.current_train_index >= self.max_batch):
                self.current_train_index = 1
                self.epoch_count_train += 1
                fname = join(self.folder, '{}_{}.p.gz'.format(self.prefix, self.current_train_index))
            with gzip.open(fname, 'rb') as f:
                X_batch, y_batch, image_data = cPickle.load(f)
            image_data = image_data - self.mean.T
            epoch = self.epoch_count_train
            if not (self.test_interval is None or self.test_interval <= 1):
                batch = (self.test_interval - 1) * (
                self.current_train_index / self.test_interval) + self.current_train_index % self.test_interval
            else:
                batch = self.current_train_index
            transform_y = self.label_transformer.transform(y_batch)
            #print 'image data shape: {}, X_batch data shape: {}'.format(image_data.shape,X_batch.shape)
            out = [epoch, batch, [expand(image_data,crop_image=self.crop_image,border_size=self.border_size,image_size=self.img_size,flip_image=self.flip_image), adjust_labels(transform_y), expand(X_batch)]]
            self.current_train_index += 1
            if not (self.test_interval is None or self.test_interval <= 1):
                if self.current_train_index % self.test_interval == 0:
                    self.current_train_index += 1
            return out
        else:
            if self.test_interval is None or self.test_interval <= 1:
                return None
            fname = join(self.folder, '{}_{}.p.gz'.format(self.prefix, self.current_test_index))

            if not os.path.isfile(fname):
                self.current_test_index = self.test_interval
                self.epoch_count_test += 1
                fname = join(self.folder, '{}_{}.p.gz'.format(self.prefix, self.current_test_index))

            with gzip.open(fname, 'rb') as f:
                X_batch, y_batch, image_data = cPickle.load(f)
            image_data = image_data - self.mean.T

            epoch = self.epoch_count_test
            batch = self.current_test_index / self.test_interval
            #print 'image data shape: {}, X_batch data shape: {}'.format(image_data.shape,X_batch.shape)
            transform_y = self.label_transformer.transform(y_batch)
            out = [epoch, batch, [expand(image_data,crop_image=self.crop_image,border_size=self.border_size,image_size=self.img_size,flip_image=self.flip_image), adjust_labels(transform_y), expand(X_batch)]]
            self.current_test_index += self.test_interval

            return out

    def get_label_names(self):
        return self.label_transformer.get_label_names()

    def get_data_dims(self, idx):
        if idx == 0:

            if self.crop_image:
                crop_size = self.img_size - 2*self.border_size
                return crop_size * crop_size * 3
            else:
                return self.img_size * self.img_size * 3
        elif idx == 1:
            return 1
        elif idx == 2:
            return self.text_size
        else:
            raise Exception('data index out of dound {}'.format(idx))

    def get_num_classes(self):
        return self.label_transformer.get_num_classes()

    def init_data_providers(self):
        self.epoch_count_train = 1
        self.epoch_count_test = 1
        self.current_train_index = 1
        self.current_test_index = self.test_interval





def main():
    op = OptionParser()

    op.add_option("--data_folder", default='/data/sgeadmin/productTypeTest',
                  action="store", type=str, dest="data_folder",
                  help="Product data file.")

    op.add_option("--output_folder_images", default='/data/sgeadmin/productTypeTest/images',
                  action="store", type=str, dest="output_folder_images",
                  help="Output folder.")

    op.add_option("--output_folder", default='/data/sgeadmin/productTypeTest/batches',
                  action="store", type=str, dest="output_folder",
                  help="Output folder.")

    op.add_option("--mongo_url", default='starcluster-gw-vip.prod.aws.adchemy.net',
                  action="store", type=str, dest="mongo_url",
                  help="Mongo url.")

    (opts, args) = op.parse_args()

    scenario_folder = opts.data_folder
    output_folder = opts.output_folder

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    size = (64, 64)
    batch_size = 1024 * 2
    n_jobs = 15

    with open(join(scenario_folder, 'matrices.npz'), 'rb') as wb_file:
        nbz = np.load(wb_file)
        X_train_data, X_train_indices, X_train_indptr, X_train_shape, y_train, X_test_data, X_test_indices, X_test_indptr, X_test_shape, y_test, products_ids_train, products_ids_test, product_image_urls_train, product_image_urls_test = [
            nbz[str(i)] for i in xrange(len(nbz.files))]
        X_train = csr_matrix((X_train_data, X_train_indices, X_train_indptr),
                             shape=(X_train_shape[0], X_train_shape[1]))
        X_test = csr_matrix((X_test_data, X_test_indices, X_test_indptr), shape=(X_test_shape[0], X_test_shape[1]))

        print 'X_train %s, y_train %s, X_test %s, y_test %s' % (
            X_train.shape, len(y_train), X_test.shape, len(y_test))

    image_storage = ImageStorage(output_folder=opts.output_folder_images, url=opts.mongo_url)
    download_files = False
    if download_files:

        image_storage.download_images(product_image_urls_train)
        image_storage.download_images(product_image_urls_test)
    else:
        print 'creating batches'
        product_image_files_train = [image_storage.get_image_location(url) for url in product_image_urls_train]
        print 'startgin batch creation for {} filesin train'.format(len(product_image_urls_train))
        create_batch(output_folder, X_train, y_train, product_image_files_train, batch_size=batch_size, size=size,
               n_jobs=n_jobs, prefix='train')
        print 'startgin batch creation for {} files in test'.format(len(product_image_urls_test))
        product_image_files_test = [image_storage.get_image_location(url) for url in product_image_urls_test]
        create_batch(output_folder, X_test, y_test, product_image_files_test, batch_size=batch_size, size=size,
                  n_jobs=n_jobs, prefix='test')




IMAGE_URL='image_url'
IMAGE_LOCATION = 'file_name'

class ImageStorage(object):

    def __init__(self, url='rsrch-mysql-db01.dev.adchemy.colo', port=27017, db='image_storage',
                 collection='image_metadata', output_folder='/tmp/images/'):
        self.client = MongoClient(url, port)
        self.db = self.client[db]
        self.collection = self.db[collection]
        self.collection_name = collection
        self.db_name = db
        self.url = url
        self.port = port
        self.output_folder = output_folder

        self.collection.ensure_index(IMAGE_URL, name=IMAGE_URL, unique=True)
        self.collection.ensure_index(IMAGE_LOCATION, name=IMAGE_LOCATION)

    def clear(self):
        self.collection.remove()

    def get_image_location(self, url):
        doc = self.collection.find_one({IMAGE_URL : url})
        if doc is None:
            return None
        return doc.get(IMAGE_LOCATION)

    def download_image(self,url):
        self._download_images_batch([url], n_jobs=1)

    def download_images(self, urls, n_jobs=10):
        i = 0
        batch_size = 2000
        while i < len(urls):
            print 'totally processed {}'.format(i)
            end = min(i+ batch_size, len(urls))
            self._download_images_batch(urls[i: end], n_jobs=n_jobs)
            i = end

    def _download_images_batch(self, urls, n_jobs=5):
        FORMAT = '%Y%m%d%H%M%S'
        data_folder = 'data_{}'.format(datetime.datetime.now().strftime(FORMAT))
        new_folder = join(self.output_folder, data_folder)
        if not os.path.isdir(new_folder):
            os.mkdir(new_folder)
        not_available_urls = [ url for url in urls if (url is not None and len(url) > 0 and not self.collection.find_one({IMAGE_URL : url}))]
        print 'urls are {}, the one to download are {}'.format(len(urls), len(not_available_urls))
        downloaded = Parallel(n_jobs=n_jobs)( delayed(_download_image)( url, new_folder) for url in not_available_urls)
        for url, full_path in downloaded:
            self.collection.update({IMAGE_URL : url}, {IMAGE_URL : url, IMAGE_LOCATION : full_path}, upsert=True )

def _download_image(  url, output_folder):
            full_path = join(output_folder, get_file_name(url))
            #print 'full path is {} for image {}'.format(full_path, url)
            try:
                urllib.urlretrieve(url, full_path)
                return url, full_path

            except Exception as e:
                #print 'cannot download image {} because {}'.format(url, e)
                return url, None



def get_file_name(url):
    return base64.urlsafe_b64encode(url)


def compute_means_save(output_folder, batch_means, prefix):
    mean = np.mean(batch_means, axis=0, dtype=np.float32)
    mean = mean.reshape((-1, 1))
    print 'shape mean of means {}'.format(mean.shape)
    with gzip.open(join(output_folder, '{}_{}.p.gz'.format(prefix, 'mean')), 'wb') as f:
        cPickle.dump(mean, f)


def create_batch(output_folder, X, y, image_files, batch_size=100, size=(32, 32), n_jobs=5, prefix='batch'):
    batch_means = []
    left_over_indices = []
    left_over_images = []
    indices = []
    batch_files = []
    batch_index = 1
    for i, image_file in enumerate(image_files):
        if image_file is not None and len(image_file) > 0:
            indices.append(i)
            batch_files.append(image_file)
            if len(batch_files) == batch_size:
                image_batch, good_indices = process_batch(batch_files, size=size, n_jobs=n_jobs)
                indices = [indices[index] for index in good_indices]

                assert (len(image_batch) == len(indices))

                #plot_some_images(np.array(image_batch,dtype=np.float32),size=size[0],labels=[y[lp] for lp in indices])


                left_over_indices.extend(indices)
                left_over_images.extend(image_batch)
                indices = []
                batch_files = []

        if len(left_over_indices) >= batch_size:
            bi = left_over_indices[0:batch_size]
            bimage = left_over_images[0:batch_size]
            print 'size bi {}, size bimage {}'.format(len(bi), len(bimage))
            left_over_indices = left_over_indices[batch_size:]
            left_over_images = left_over_images[batch_size:]

            X_batch = X[bi]
            y_batch = y[bi]
            print 'saving batch {} of size {}, processed a total of {} images'.format(batch_index, len(bi), i)

            image_data = np.array(bimage, dtype=np.float32)
            mean = np.mean(image_data, axis=0, dtype=np.float32)
            print 'shape mean {}'.format(mean.shape)
            batch_means.append(mean)

            assert (X_batch.shape[0] == len(y_batch) )
            assert (X_batch.shape[0] == image_data.shape[0] )
            with gzip.open(join(output_folder, '{}_{}.p.gz'.format(prefix, batch_index)), 'wb') as f:
                cPickle.dump([X_batch, y_batch, image_data], f)
            compute_means_save(output_folder, batch_means, prefix)

            batch_index += 1

    if len(left_over_indices) > 0:
        bi = left_over_indices
        bimage = left_over_images
        print 'size bi {}, size bimage {}'.format(len(bi), len(bimage))

        X_batch = X[bi]
        y_batch = y[bi]

        image_data = np.array(bimage, dtype=np.float32)
        mean = np.mean(image_data, axis=0, dtype=np.float32)
        print 'shape mean {}'.format(mean.shape)
        batch_means.append(mean)

        print 'saving batch {} of size {}, processed a total of {} images'.format(batch_index, len(bi), i)
        with gzip.open(join(output_folder, '{}_{}.p.gz'.format(prefix, batch_index)), 'wb') as f:
            cPickle.dump([X_batch, y_batch, image_data], f)


def plot_some_images(image_data, size, labels):
    fig = pl.figure(1)
    print 'shape images data {}'.format(image_data.shape)
    plottable_data = np.require(
        image_data.reshape(image_data.T.shape[1], 3, size, size).swapaxes(1, 3).swapaxes(1, 2) / 255.0, dtype=np.single)
    for img_idx in xrange(10):
        pl.title(labels[img_idx])
        img = plottable_data[img_idx, :, :, :]
        pl.imshow(img, interpolation='nearest')
        pl.show()


if __name__ == "__main__":
    main()

