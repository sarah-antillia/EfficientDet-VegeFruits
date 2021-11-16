#******************************************************************************
#
#  Copyright (c) 2020-2021 Antillia.com TOSHIYUKI ARAI. ALL RIGHTS RESERVED.
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

#2021/11/12 toshiyuki.arai

"""
  Modified the following line in read metho.
            # The following line will cause an error if line contained the line 'name: "Maximum_Width_in_Meters",', 
            # because "id" in "Width".
            #if "id" in line:
            if line.startswith("id:"):
            

"""
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
          try:
            line.replace(" ", "")
            # The following line will cause an error if line contained the line 'name: "Maximum_Width_in_Meters",', 
            # because "id" in "Width".
            #if "id" in line:
            if "id:" in line:
                id = int(line.split(":")[1].replace(",", "").strip() )
                
            #elif "name" in line:  
            elif "name:" in line:
                name = line.split(":")[1].replace(",", "").strip()
                name = name.replace("'", "").replace("\"", "")
                
            if id is not None and name is not None:
                classes.append(name)
                #items[name] = id 2021/09/20
                items[id] = name

                id = None
                name = None
          except:
            traceback.print_exc()
            
    return items, classes



if __name__ == "__main__":
  label_map = "./projects/BloodCells/train/label_map.pbtxt"

  try:
     reader = LabelMapReader()
     items, classes = reader.read(label_map)
     print(items)
     print(classes)

  except Exception as ex:
    traceback.print_exc()
