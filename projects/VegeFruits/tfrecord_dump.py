# 2021/09/15 arai.toshiyuki@safie.jp
# tfrecord_image_extractor.py
import os
import sys
from pprint import pprint
import cv2
import numpy as np
import tensorflow as tf
import warnings
import traceback
from LabelMapReader import LabelMapReader
warnings.filterwarnings('ignore', category = FutureWarning)  #tf 1.14 and np 1.17 are clashing: temporary solution
def cv_bbox(image, bbox, color = (255, 255, 255), line_width = 2):
    """
    use opencv to add bbox to an image
    assumes bbox is in standard form x1 y1 x2 y2
    """
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, line_width)
    return
def parse_record(data_record):
    """
    parse the data record from a tfrecord file, typically pulled from an iterator,
    in this case a one_shot_iterator created from the dataset.
    """
    #2021/08/23 arai: image/filename is not required. It's OK if it's missing.
    #2021/10/21 arai: Added 'image/filename':         tf.io.FixedLenFeature([], tf.string)
    feature = {'image/encoded': tf.io.FixedLenFeature([], tf.string),
                'image/object/class/label': tf.io.VarLenFeature(tf.int64),
                'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
                'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
                'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
                'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
                'image/filename':         tf.io.FixedLenFeature([], tf.string)
                }
    return tf.io.parse_single_example(data_record, feature)
# arai
# Modified to write annotated images to the output_images_dir
# class_labels = {1: 'excluded', 2: 'car', 3: 'unictruck', 4: 'crane', 5: 'vantruck',
#  6: 'dump', 7: 'deepdump', 8: 'flattruck', 9: 'mixer', 10: 'trailer', 11: 'others', 12: 'tank'}
def extract_image_from_tfrecord(file_path, class_labels, output_images_dir, stride = 1, verbose = 1):
    """
    peek at the data using opencv and tensorflow tools.
    Inputs:
        file_path: path to tfrecord file (usually has 'record' extension)
        class_labels: dictionary of labels with name:number pairs (start with 1)
        stride (default 1): how many records to jump (you might have thousands so skip a few)
        verbose (default 1): display text output if 1, display nothing except images otherwise.
    """
    dataset = tf.data.TFRecordDataset([file_path])
    record_iterator = iter(dataset)
    num_records = dataset.reduce(np.int64(0), lambda x, _: x + 1).numpy()
    print("----- Num records {}".format(num_records))
    if verbose:
        print(f"\nGoing through {num_records} records with a stride of {stride}.")
        print("Enter 'n' to bring up next image in record.\n")
    for im_ind in range(num_records):
        #Parse and process example
        parsed_example = parse_record(record_iterator.get_next())
        encoded_image = parsed_example['image/encoded']
        image_np = tf.image.decode_image(encoded_image, channels=3).numpy()
        filename = ""
        try:
          filename = parsed_example['image/filename']
          filename = "{}".format(filename)
          filename = filename.strip('b').strip("'")
          print("=== filename {}".format(filename))
        except:
          traceback.print_exc()
        labels =  tf.sparse.to_dense( parsed_example['image/object/class/label'], default_value=0).numpy()
        x1norm =  tf.sparse.to_dense( parsed_example['image/object/bbox/xmin'], default_value=0).numpy()
        x2norm =  tf.sparse.to_dense( parsed_example['image/object/bbox/xmax'], default_value=0).numpy()
        y1norm =  tf.sparse.to_dense( parsed_example['image/object/bbox/ymin'], default_value=0).numpy()
        y2norm =  tf.sparse.to_dense( parsed_example['image/object/bbox/ymax'], default_value=0).numpy()
        num_bboxes = len(labels)
        #% Process and display image
        height, width = image_np[:, :, 1].shape
        image_copy    = image_np.copy()
        image_rgb     = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
        print(" ---- class_label {}".format(class_labels))
        if num_bboxes > 0:
            x1 = np.int64(x1norm*width)
            x2 = np.int64(x2norm*width)
            y1 = np.int64(y1norm*height)
            y2 = np.int64(y2norm*height)
            for bbox_ind in range(num_bboxes):
                    bbox = (x1[bbox_ind], y1[bbox_ind], x2[bbox_ind], y2[bbox_ind])
                    print("------ labels {}".format(labels))
                    category = labels[bbox_ind] # -1
                    print(" bbox_ind {}".format(bbox_ind))
                    #index = bbox_ind
                    print("------ category {}".format(category))
                    label_name = class_labels[category]
                    #list(class_labels.keys())[list(class_labels.values()).index(labels[bbox_ind])]
                    print(" category_id {}  label_name {}".format(category, label_name))
                    label_position = (bbox[0] + 5, bbox[1] + 20)
                    color = (255, 0, 0) #BRG BLUE
                    if category == 0:
                       color = (0, 0, 255) #RED
                    """
                    cv_bbox(image_rgb, bbox, color = color, line_width = 2)
                    cv2.putText(image_rgb,
                                label_name,
                                label_position,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, color, 2); #scale, color, thickness
                    """

        if verbose:
            print(f"\nImage {im_ind}")
            #print(f"    {fname}")
            print(f"    Height/width: {height, width}")
            print(f"    Num bboxes: {num_bboxes}")
        #cv2.imshow("bb data", image_rgb)
        output_image_file = os.path.join(output_images_dir, filename)
        print("=== Saved image file {}".format(output_image_file))
        cv2.imwrite(output_image_file, image_rgb , [cv2.IMWRITE_JPEG_QUALITY, 100])
def get_class_labels_from(label_map):
  class_labels = {}
  reader = LabelMapReader()
  dic, classes, = reader.read(label_map)
  print(dic)
  print(classes)
  #dic = {(i + 1): classes[i] for i in range(0, len(classes))}
  #print("---- dic {}".format(dic))
  """
  dic =   {1: 'excluded',
           2: 'car',
           3: 'unictruck',
           4: 'crane',
           5: 'vantruck',
           6: 'dump',
           7: 'deepdump',
           8: 'flattruck',
           9: 'mixer',
           10: 'trailer',
           11: 'others',
           12: 'tank'}
  """
  #return classes
  return dic
# Modified to take two arguments tfrecod_file label_map.pbtxt
# python tfrecord_image_extractor_tf2.py ./tfrecord/test/512x512_kajima.tfrecord ./dataset/label_map.pbtxt  ./dataset/test
if __name__ == '__main__':
  #tf.disable_eager_execution()
  try:
    if len(sys.argv) <4:
      raise Exception("Usage: python tfrecord_image_extractor_tf2.py ./tfrecord/valid/512x512_kajima.tfrecord ./dataset/label_map.pbtxt  ./output_images_dir")
    if len(sys.argv) >=4:
      tfrecord_file     = sys.argv[1]
      label_map_pbtxt   = sys.argv[2]
      output_images_dir = sys.argv[3]
      if not os.path.exists(tfrecord_file):
        raise Exception(" Not found {}".format(tfrecord_file))
      if not os.path.exists(label_map_pbtxt):
        raise Exception(" Not found {}".format(label_map_pbtxt))
      if not os.path.exists(output_images_dir):
        os.makedirs(output_images_dir)
      class_labels = get_class_labels_from(label_map_pbtxt)
      print("---- class_labels {}".format(class_labels))
      verbose = 1
      stride  = 1
      extract_image_from_tfrecord(tfrecord_file, class_labels, output_images_dir, stride = stride, verbose = verbose)
  except Exception as ex:
    traceback.print_exc()
