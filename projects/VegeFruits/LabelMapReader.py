# LabelMapReader.py
# 2021/09/27
#
# arai.toshiyuki@safie.jp
#

import os
import sys
import traceback

class LabelMapReader:

  def __init__(self):
    pass

  def read(self, label_map_file):
    id    = None
    name  = None
    items = {}
    classes = []
    with open(label_map_file, "r") as f:
        for line in f:
            line.replace(" ", "")
            if "id" in line:
                id = int(line.split(":")[1].replace(",", "").strip() )

            elif "name" in line:
                name = line.split(":")[1].replace(",", "").strip()
                name = name.replace("'", "").replace("\"", "")

            if id is not None and name is not None:
                classes.append(name)
                items[id]   = name  #2021/10/29 Modified
                #items[name] = id
                id = None
                name = None

    return items, classes



if __name__ == "__main__":
  label_map = "./dataset/label_map.pbtxt"

  try:
     reader = LabelMapReader()
     items, classes = reader.read(label_map)
     print("--- items   {}".format(items))
     print("--- classes {}".format(classes))


     for i in items:
       print("i {} name  {}".format(i, items[i]) )

     for i in range(20):
       try:
         class_name = items[i]
         print("index {}  class {}".format(i, class_name))
       except:
         traceback.print_exc()


  except Exception as ex:
    traceback.print_exc()