from optparse import OptionParser
import os
from os.path import join
import cPickle
import shutil
import traceback
import sys
from scikit_learn_wrapper import ConvNetLearn
from collect_images import ImageDataProvider
import numpy as np

class LabelTransformer:
    def __init__(self, label_encoder):
        self.labelEncoder = label_encoder
    def transform(self,y):
        return np.array([_transform(elem,self.labelEncoder) for elem in y])

    def get_num_classes(self):
        return len(list(self.labelEncoder.classes_))

    def get_label_names(self):
        return list(self.labelEncoder.classes_)

def _transform(y, encoder):
    try:
        return encoder.transform([y])[0]
    except:
        return -1

def run_single_model(batch_folder, output_folder, layer_file, layer_params_file, epochs=5, last_model=None, max_batch=None, init_states_models=None, crop_image=False, border_size=8):

    with open(join(batch_folder, 'label_encoder.p'), 'rb') as encoder_file:
        label_encoder = cPickle.load(encoder_file)

    label_transformer = LabelTransformer(label_encoder)
    print 'labels: {}'.format(label_transformer.get_label_names())
    data_provider = ImageDataProvider(batch_folder, 'train', test_interval=15, label_transformer=label_transformer, max_batch=max_batch,  crop_image=crop_image, border_size=border_size)

    num_classes = data_provider.get_num_classes()
    print 'num classes: {}'.format(num_classes)

    net = ConvNetLearn(layer_file=layer_file, layer_params_file=layer_params_file, epochs=epochs, fraction_test=0.05, last_model=last_model, init_states_models=init_states_models)

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)


    fit_model = True
    data_provider_test = ImageDataProvider(batch_folder, 'test', test_interval=0, label_transformer=label_transformer, max_batch=max_batch, crop_image=crop_image, border_size=border_size)
    if fit_model:
        net.fit(data_provider, None, use_starting_point=True)



        score_test = net.score(data_provider_test, None, train=True)
        score_test_of_train = net.score(data_provider, None, train=False)
        score_test_f1macro = net.score(data_provider_test, None, train=True, type='f1macro')
        score_test_of_train_f1macro = net.score(data_provider, None, train=False, type='f1macro')
        score_test_f1 = net.score(data_provider_test, None, train=True, type='f1')
        score_test_of_train_f1 = net.score(data_provider, None, train=False, type='f1')
        print 'score on test data is {}'.format(score_test)
        print 'score on test part of the train data is {}'.format(score_test_of_train)
        print 'score f1 macro on test data is {}'.format(score_test_f1macro)
        print 'score f1 macro on test part of the train data is {}'.format(score_test_of_train_f1macro)
        print 'score f1 on test data is {}'.format(score_test_f1)
        print 'score f1 on test part of the train data is {}'.format(score_test_of_train_f1)

        model_folder = join(output_folder,'model')
        if os.path.exists(model_folder):
            shutil.rmtree(model_folder)
        shutil.copytree(net.last_model, model_folder)
        shutil.copy(join(batch_folder, 'label_encoder.p'), join(output_folder, 'label_encoder.p'))
    else:
        # try:
        #     print 'plotting filters'
        #     net.plot_filters(output_file=join(output_folder,'filters.jpg'), data_provider=data_provider)
        # except:
        #     pass
        try:
            print 'plotting predictions'
            net.plot_predictions(data_provider_test,join(output_folder,'predictions.jpg'), only_errors=True)
            print 'done plottign predictions'
        except Exception as e:
            print 'problem plotting predictions: {}'.format(e)

            traceback.print_exc(file=sys.stdout)
            pass




def main():
    op = OptionParser()

    op.add_option("--batch_folder", default='/mnt/image_data/batches_staged/footwear_1361407141225_1361407159889_75',
                  action="store", type=str, dest="batch_folder",
                  help="Product data batch folder .")



    op.add_option("--output_folder", default='/mnt/image_data/models/footwear_1361407141225_1361407159889_75',
                  action="store", type=str, dest="output_folder",
                  help="Location of the output")

    #op.add_option("--layer_def", default='./layers/layers-image-text.cfg',action="store", type=str, dest="layer_def",help="Layer definition.")
    #op.add_option("--layer_def", default='./layers/layers-text.cfg',action="store", type=str, dest="layer_def",help="Layer definition.")
    op.add_option("--layer_def", default='./layers/layers-image.cfg',action="store", type=str, dest="layer_def",help="Layer definition.")
    #op.add_option("--layer_def", default='./layers/layers-text-simple.cfg',action="store", type=str, dest="layer_def",help="Layer definition.")

    #op.add_option("--layer_params", default='./layers/layer-params-image-text.cfg',action="store", type=str, dest="layer_params",help="The layer parameters file")
    #op.add_option("--layer_params", default='./layers/layer-params-text.cfg',action="store", type=str, dest="layer_params",help="The layer parameters file")
    op.add_option("--layer_params", default='./layers/layer-params-image.cfg',action="store", type=str, dest="layer_params",help="The layer parameters file")
    #op.add_option("--layer_params", default='./layers/layer-params-text-simple.cfg',action="store", type=str, dest="layer_params",help="The layer parameters file")


    op.add_option("--last_model", default=None,
                  action="store", type=str, dest="last_model",
                  help="Last model location")

    op.add_option("--init_models", default=None,
                  action="store", type=str, dest="init_models",
                  help="Models used to init the matrices in the last_model. This is used to load  multiple models into a bigger more complex one. (come separated)")

    (opts, args) = op.parse_args()


    output_folder = opts.output_folder
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    init_model_string = opts.init_models
    init_states_models = None
    if init_model_string is not None:
        init_states_models = init_model_string.split(',')

    crop_image = True
    border_size = 8


    run_single_model(opts.batch_folder, opts.output_folder, opts.layer_def, opts.layer_params,epochs=30, last_model=opts.last_model, init_states_models=init_states_models, crop_image=crop_image, border_size=border_size)

if __name__ == "__main__":
        main()