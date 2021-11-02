#******************************************************************************
#
#  Copyright (c) 2020-2021 Antillia.com TOSHIYUKI ARAI. ALL RIGHTS RESERVED.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#******************************************************************************

import os
import sys


class DetectResultsWriter(object):

  def __init__(self, output_image_path):
    self.output_image_path = output_image_path
 

  def write(self, detected_objects, objects_stats):
    OBJECTS = "_objects"
    STATS   = "_stats"
    CSV     = ".csv"
    SEP     = ","
    NL      = "\n"
    detected_objects_path = self.output_image_path + OBJECTS + CSV
    objects_stats_path    = self.output_image_path + STATS   + CSV

    #2020/08/15 atlan: save the detected_objects as csv file
    print("=== Writing detected_objects to: {}".format(detected_objects_path))
    with open(detected_objects_path, mode='w') as f:
      #2020/09/15 Write a header(title) line of csv.
      header = "id, class, score, x, y, w, h" + NL
      f.write(header)

      for item in detected_objects:
        line = str(item).strip("()").replace("'", "") + NL
        f.write(line)
       
    #2020/08/15 atlan: save the detected_objects as csv file
    print("=== Writing objects_stats to: {}".format(objects_stats_path))

    #print("=== objects_stats {}".format(objects_stats))

    with open(objects_stats_path, mode='w') as s:
      #2020/09/15 Write a header(title) line of csv.
      header = "id, class, count" + NL
      s.write(header)
          
      for (k,v) in enumerate(objects_stats.items()):
        (name, value) = v
        line = str(k +1) + SEP + str(name) + SEP + str(value) + NL
        s.write(line)
