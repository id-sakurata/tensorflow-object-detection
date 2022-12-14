import functools
import json
import os
import tensorflow as tf

from object_detection.legacy import trainer
from object_detection.builders import input_reader_builder
from object_detection.builders import model_builder
from object_detection.utils import config_util

tf.logging.set_verbosity(tf.logging.INFO)



def main(_):
  master ='' #Name of the TensorFlow master to use.
  task = 0 #task id
  num_clones = 1 #Number of clones to deploy per worker.
  clone_on_cpu = False #Force clones to be deployed on CPU.  Note that even if
                       #set to False (allowing ops to run on gpu), some ops may
                       #still be run on the CPU if they have no GPU kernel.
  worker_replicas = 1 #Number of worker+trainer replicas.
  ps_tasks = 0 #Number of parameter server tasks. If None, does not use a parameter server.
  train_dir = 'training/' #Directory to save the checkpoints and training summaries.
  pipeline_config_path = 'config/ssd_mobilenet_v1_pets.config' #Path to a pipeline_pb2.TrainEvalPipelineConfig config file. If provided, other configs are ignored
  train_config_path = '' #Path to a train_pb2.TrainConfig config file.
  input_config_path = '' #Path to an input_reader_pb2.InputReader config file.
  model_config_path = '' #Path to a model_pb2.DetectionModel config file.

  #assert train_dir, '`train_dir` is missing.'
  if task == 0: tf.gfile.MakeDirs(train_dir)
  if pipeline_config_path:
    configs = config_util.get_configs_from_pipeline_file(
        pipeline_config_path)
    if task == 0:
      tf.gfile.Copy(pipeline_config_path,
                    os.path.join(train_dir, 'pipeline.config'),
                    overwrite=True)
  else:
    configs = config_util.get_configs_from_multiple_files(
        model_config_path=model_config_path,
        train_config_path=train_config_path,
        train_input_config_path=input_config_path)
    if task == 0:
      for name, config in [('model.config', model_config_path),
                           ('train.config', train_config_path),
                           ('input.config', input_config_path)]:
        tf.gfile.Copy(config, os.path.join(train_dir, name),
                      overwrite=True)

  model_config = configs['model']
  train_config = configs['train_config']
  input_config = configs['train_input_config']

  model_fn = functools.partial(
      model_builder.build,
      model_config=model_config,
      is_training=True)

  create_input_dict_fn = functools.partial(
      input_reader_builder.build, input_config)

  env = json.loads(os.environ.get('TF_CONFIG', '{}'))
  cluster_data = env.get('cluster', None)
  cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None
  task_data = env.get('task', None) or {'type': 'master', 'index': 0}
  task_info = type('TaskSpec', (object,), task_data)

  # Parameters for a single worker.
  ps_tasks = 0
  worker_replicas = 1
  worker_job_name = 'lonely_worker'
  task = 0
  is_chief = True
  master = ''

  if cluster_data and 'worker' in cluster_data:
    # Number of total worker replicas include "worker"s and the "master".
    worker_replicas = len(cluster_data['worker']) + 1
  if cluster_data and 'ps' in cluster_data:
    ps_tasks = len(cluster_data['ps'])

  if worker_replicas > 1 and ps_tasks < 1:
    raise ValueError('At least 1 ps task is needed for distributed training.')

  if worker_replicas >= 1 and ps_tasks > 0:
    # Set up distributed training.
    server = tf.train.Server(tf.train.ClusterSpec(cluster), protocol='grpc',
                             job_name=task_info.type,
                             task_index=task_info.index)
    if task_info.type == 'ps':
      server.join()
      return

    worker_job_name = '%s/task:%d' % (task_info.type, task_info.index)
    task = task_info.index
    is_chief = (task_info.type == 'master')
    master = server.target

  trainer.train(create_input_dict_fn, model_fn, train_config, master, task,
                num_clones, worker_replicas, clone_on_cpu, ps_tasks,
                worker_job_name, is_chief, train_dir)


if __name__ == '__main__':
  tf.app.run()
