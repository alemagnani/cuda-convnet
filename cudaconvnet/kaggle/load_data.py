from optparse import OptionParser
import numpy as np
import cPickle
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer


def load_data_raw(file, output_file, max_features=100000):
    distinct_indices = set()
    max_index = None
    total_labels_count = 0
    total_labels = []
    total_indices = []
    ptr = [0]
    total_values = []
    count = 1

    for labels, indices, values in iterate_file(file):

            if labels is None:
                continue
            if len(labels) == 0:
                raise Exception('probelm with labels in line {}'.format(line))
            total_labels.extend(labels)
            for label in labels:
                total_indices.extend(indices)
                total_values.extend(values)
                new_position = ptr[-1] + len(values)
                ptr.append(new_position)

            total_labels_count += len(labels)
            distinct_indices.update(indices)
            m_local = np.max(indices)
            if max_index is None or m_local > max_index:
                max_index = m_local
            count +=1
            if count % 10000 ==0:
                print 'processed {} lines'.format(count)

            if count == 200000:
                break

    matrix = csr_matrix( (total_values,total_indices,ptr), shape=(total_labels_count,max_index + 1), dtype=np.int8)
    print 'distinct indices {}, max index {}, total labels {}'.format(len(distinct_indices),max_index, total_labels_count)
    with open(output_file,'wb') as out:
        cPickle.dump([matrix, total_labels], out)


def load_data(file, output_file, vect_output_file, max_features=100000):

    vect = CountVectorizer(max_features=max_features, max_df=0.2,  dtype=np.float32)

    vect.fit(iterate_file_fake_doc(file))

    print 'size of features {}'.format(len(vect.get_feature_names()))

    matrix = vect.transform(iterate_file_fake_doc(file,iterate_label=True))

    total_labels = [label for label in iterate_file_label(file)]

    assert(len(total_labels) == matrix.shape[0])
    print 'dtype {}, {}'.format(matrix.dtype, type(matrix))

    with open(output_file,'wb') as out:
        cPickle.dump([matrix, total_labels], out)

    with open(vect_output_file,'wb') as out:
        cPickle.dump(vect, out)


def iterate_file(file):
    with open(file, 'rb') as f:
        for line in f:
            labels, indices, values = process_line(line)
            yield labels, indices, values

def iterate_file_label(file):
     count = 0
     for labels, indices, values in iterate_file(file):
            if labels is None:
                continue
            count +=1
            if count % 10000 == 0:
                print 'processed {}'.format(count)

            for label in labels:
                yield label

def iterate_file_fake_doc(file, iterate_label=False):
        count = 0
        for labels, indices, values in iterate_file(file):
            if labels is None:
                continue
            count +=1
            if count % 10000 == 0:
                print 'processed {}'.format(count)


            fake_doc = []
            for i in xrange(len(indices)):
                fake_doc.extend( [str(indices[i])] * values[i])
            fake_doc = ' '.join(fake_doc)

            for label in labels:
                yield fake_doc
                if not iterate_label:
                    break



def process_line(line):
    parts = line.split(' ')
    lp = len(parts)
    if lp == 0 or lp ==1:
        return None, None, None
    labels = []
    indices = []
    values = []
    for part in parts:
        if ':' in part:
            pair = part.split(':')
            indices.append(int(pair[0]))
            v = int(pair[1])

            values.append(v)
        elif ',' in part:
            #print 'part {} obtained {}'.format(part, part[:-1])
            labels.append(int(part[:-1]))
        else:
            labels.append(int(part))
    return labels, indices, values

def main():

    op = OptionParser()

    op.add_option("--train_data", default='/Users/alessandro/Desktop/kaggle/train.csv',
                  action="store", type=str, dest="train_data",
                  help="Train data.")
    op.add_option("--output_data", default='/Users/alessandro/Desktop/kaggle/train_matrix.p',
                  action="store", type=str, dest="output_data",
                  help="Train data output.")

    op.add_option("--vectorizer_file", default='/Users/alessandro/Desktop/kaggle/vectorizer.p',
                  action="store", type=str, dest="vectorizer_file",
                  help="Vectorizer file.")

    (opts, args) = op.parse_args()
    load_data(opts.train_data, opts.output_data, vect_output_file=opts.vectorizer_file)

if __name__ == "__main__":
    main()