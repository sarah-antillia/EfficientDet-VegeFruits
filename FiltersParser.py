#******************************************************************************
#
#  Copyright (c) 2020 Antillia.com TOSHIYUKI ARAI. ALL RIGHTS RESERVED.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#******************************************************************************
# 
# EfficientDet 
# FiltersParser.py
import os

## coco classes
COCO_CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

class FiltersParser:

  # Specify a str_filters string like this "[person,motorcycle]" ,
  # If needed, please specify your own classes.
  # default classes = COCO_CLASSES
  def __init__(self, classes=COCO_CLASSES):
      print("== FiltersParser ")
      
      self.str_filters  = None
      self.classes      = classes
      self.filters  = []


  def parse(self, filters):
      self.str_filters = filters
      if filters == None:
        return filters
        
      self.filters = []
      if self.str_filters != None:
          tmp = self.str_filters.strip('[]').split(',')
          if len(tmp) > 0:
              for e in tmp:
                  e = e.lstrip()
                  e = e.rstrip()
                  if e in self.classes :
                    self.filters.append(e)
                  else:
                    print("Invalid label(class)name {}".format(e))
      return self.filters

  #2020/07/31 Updated 
  def get_ouput_filename(self, input_image_filename, image_out_dir):
        rpos  = input_image_filename.rfind("/")
        fname = input_image_filename
        if rpos >0:
            fname = input_image_filename[rpos+1:]
        else:
            rpos = input_image_filename.rfind("\\")
            if rpos >0:
                fname = input_image_filename[rpos+1:]
          
        
        abs_out  = os.path.abspath(image_out_dir)
        if not os.path.exists(abs_out):
            os.makedirs(abs_out)

        filname = ""
        if self.str_filters is not None:
            filname = self.str_filters.strip("[]").replace("'", "").replace(", ", "_")
            if len(filname) != 0:
              filname += "_"
        
        output_image_filename = os.path.join(abs_out, filname + fname)
        return output_image_filename

