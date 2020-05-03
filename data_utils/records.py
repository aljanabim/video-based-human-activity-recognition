"""This is all bullshit, but it's necessary according to
   https://www.tensorflow.org/tutorials/load_data/tfrecord#walkthrough_reading_and_writing_image_data"""

import tensorflow as tf

def _tensor_to_bytes_feature(value):
    if tf.executing_eagerly():  # eager tensors have numpy() instead of numpy
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.numpy().tostring()]))
    else:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.numpy.tostring()]))


def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _serialize_sample(tensor, label):
    # print(frame)
    feature = {'tensor': _tensor_to_bytes_feature(tensor),
               'label': _tensor_to_bytes_feature(label)}
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


def _tf_serialize_sample(tensor, label):
    tf_string = tf.py_function(
        _serialize_sample,
        (tensor, label),
        tf.string)
    return tf.reshape(tf_string, ())


def serialize_dataset(dataset, output_directory, name):
    print("Starting serialization")
    serialized_dataset = dataset.map(_tf_serialize_sample)
    print("Serialization complete")
    file_name = "{}/{}.tfrecord".format(output_directory, name)
    writer = tf.data.experimental.TFRecordWriter(output_directory)
    writer.write(serialized_dataset)
    print("TFRecord '{}.tfrecord' created".format(name))
