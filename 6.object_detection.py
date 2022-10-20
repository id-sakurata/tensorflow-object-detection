import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import warnings
import glob

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from utils import label_map_util
from utils import visualization_utils as vis_util

warnings.filterwarnings("ignore")

pathfile = input("Masukkan File: ")
pathfile = pathfile.replace("'", "")
pathfile = pathfile.strip()
#------------------------------------------------------------------------------

NUM_CLASSES = 2
PATH_TO_CKPT = 'models/frozen_inference_graph.pb'
PATH_TO_LABELS = 'models/object-detection.pbtxt'

GetDir = os.path.dirname(os.path.realpath(__file__))
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Running the tensorflow session
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.compat.v1.GraphDef()
  with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
    with tf.compat.v1.Session(graph=detection_graph) as sess:

      isExist = os.path.exists(pathfile)
      if isExist == False:
        print('File tidak ditemukan.')
        exit()

      image_np = cv2.imread(pathfile)
      filename = os.path.basename(pathfile)
      height, width, channels = image_np.shape
      image_bw = image_np.copy()

      image_np_expanded = np.expand_dims(image_bw, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections], feed_dict={image_tensor: image_np_expanded})
      ops = tf.compat.v1.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(tensor_name)
      if 'detection_masks' in tensor_dict:
        detection_boxes = tf.compat.v1.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.compat.v1.squeeze(tensor_dict['detection_masks'], [0])
        real_num_detection = tf.compat.v1.cast(tensor_dict['num_detections'][0], tf.compat.v1.int32)
        detection_boxes = tf.compat.v1.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.compat.v1.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.compat.v1.cast(tf.compat.v1.greater(detection_masks_reframed, 0.5), tf.compat.v1.uint8)
        tensor_dict['detection_masks'] = tf.compat.v1.expand_dims(detection_masks_reframed, 0)
      image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')
      output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image_bw, 0)})
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
      
      #---- output process-------------------
      max_boxes_to_draw = output_dict['detection_boxes'].shape[0]
      print('')
      for i in range(min(max_boxes_to_draw, output_dict['detection_boxes'].shape[0])):
        if output_dict['detection_classes'][i] in category_index.keys():
          class_name = category_index[output_dict['detection_classes'][i]]['name']
          akurasi = output_dict['detection_scores'][i]
          ymin = boxes[0, i, 0]
          xmin = boxes[0, i, 1]
          ymax = boxes[0, i, 2]
          xmax = boxes[0, i, 3]
          label = f'{class_name} = {round(akurasi*100,2)}%'
          (xminn, xmaxx, yminn, ymaxx) = (xmin * width, xmax * width, ymin * height, ymax * height)
          image_np = cv2.rectangle(image_np, (int(xminn), int(yminn)), (int(xmaxx), int(ymaxx)), (0, 255, 0), 4)
          
          textX = int(xminn) + int(xmaxx/4)
          textY = int(yminn) + int(ymaxx/2)
          text_size, _ = cv2.getTextSize(text=label, fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=2, thickness=2)
          text_w, text_h = text_size
          image_np = cv2.rectangle(image_np, (textX - 50, textY - 70), (textX + text_w + 50, textY + text_h), (0, 0, 0), -1)
          image_np = cv2.putText(img=image_np, text=label, org=(textX, textY), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=2, color=(255, 255, 255),thickness=2)
          
          print(label)
          break

      # cv2.imwrite(f'{GetDir}/outputs/{filename}', image_np)

      scale_percent = 15 # percent of original size
      width = int(image_np.shape[1] * scale_percent / 100)
      height = int(image_np.shape[0] * scale_percent / 100)
      dim = (width, height)
      resized = cv2.resize(image_np, dim, interpolation = cv2.INTER_AREA)
      cv2.imshow('win', resized);
      #pindahkan gambar setelah terproses ke folder developer
cv2.waitKey(0)