import tensorflow as tf
from google.protobuf import text_format
from object_detection import exporter
from object_detection.protos import pipeline_pb2

slim = tf.contrib.slim
flags = tf.app.flags

def main(_):

  input_shape = None
  input_type = "image_tensor"
  pipeline_config_path = "config/ssd_mobilenet_v1_pets.config"
  trained_checkpoint_prefix = "training/model.ckpt-200933"
  output_directory = "output"
  
  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
  with tf.gfile.GFile(pipeline_config_path, 'r') as f:
    text_format.Merge(f.read(), pipeline_config)
  if input_shape:
    input_shape = [
        int(dim) if dim != '-1' else None
        for dim in input_shape.split(',')
    ]
  else:
    input_shape = None
  exporter.export_inference_graph(input_type, pipeline_config,
                                  trained_checkpoint_prefix,
                                  output_directory, input_shape)


if __name__ == '__main__':
  tf.app.run()
