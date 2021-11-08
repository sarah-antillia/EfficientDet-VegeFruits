# Copyright 2021 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# This is based on the code in the following site:
# https://github.com/kimyoon-young/centerNet-deep-sort/blob/master/tools/cocoeval.py

# CategoriedCocoEvaluator.py

import os
import numpy as np
import datetime
import time
import traceback
import pprint
from collections import defaultdict
import pycocotools.mask as maskUtils

import copy
from pycocotools.cocoeval import COCOeval

class CategoriedCocoEvaluator(COCOeval):
    # Constructor
    
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm', label_map=None, eval_dir=None):
         print("=== CategoriedCocoEvaluator")
         super().__init__(cocoGt, cocoDt, iouType)
         self.label_map  = label_map
         self.eval_dir   = eval_dir
            
    # Override summarize.
    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            #print('==== _summerize ... {}'.format(iouThr))

            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            s = None
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1

            else:
                mean_s = np.mean(s[s>-1])

            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))

            # return mean_s
            # Modified to return meas_s and s, which is a precision or recall depending on ap
            
            return mean_s, s


        # Write ap, ar, and f to a logfile for each epoch.

        def _write_categorized_ap_ar_f(ap, ar):
            #This computes ap, ar an f for each category, and write them to each category_ap_ar_f.csv file
            #
            # See also: https://github.com/kimyoon-young/centerNet-deep-sort/blob/master/tools/cocoeval.py
            p = self.params

            #print("=== _write_ap_ar_f ")
            num_classes = len(p.catIds)
            avg_ap = 0.0

            #print("=== num_classes:{} label_map:{}".format(num_classes, self.label_map))
            try: 
                SEP = ","
                NL  = "\n"
                # NOTE: We get epoch from the environment variable.
                epoch = os.environ['epoch']
                for i in range(0, num_classes):
                    category = self.label_map[i+1]

                    filepath = os.path.join(self.eval_dir, category + "_ap_ar_f.csv")
                    if os.path.exists(filepath) == False:
                        with open(filepath, "w") as file:
                            head = "epoch, ap, ar, f" + NL 
                            file.write(head)
                    #Open the filepath with append mode
                    with open(filepath, "a") as file:
                        map = np.mean(ap[:,:,i,:])
                        mar = np.mean(ar[:,i,:])
                        f   = 2.0*(map * mar)/(map + mar)

                        map = round(map, 3)
                        mar = round(mar, 3)
                        f   = round(f,   3)

                        line = str(epoch)+ SEP + str(map) + SEP + str(mar) + SEP + str(f) + NL
                        file.write(line)
                        print("=== epoch:{}, category:{} ap:{}, ar:{}, f:{}".format(epoch, category, map, mar, f))
                        
            except Exception as ex:
                traceback.print_exc()


        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0],ap   = _summarize(1)
            stats[1],_    = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2], _   = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[3], _   = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[4], _   = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[5], _   = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])

            stats[6],ar   = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7],_    = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8], _   = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9], _   = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[10],_   = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[11],_   = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])

            _write_categorized_ap_ar_f(ap, ar)
            return stats

        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0], _ = _summarize(1, maxDets=20)
            stats[1], _ = _summarize(1, maxDets=20, iouThr=.5)
            stats[2], _ = _summarize(1, maxDets=20, iouThr=.75)
            stats[3], _ = _summarize(1, maxDets=20, areaRng='medium')
            stats[4], _ = _summarize(1, maxDets=20, areaRng='large')
            stats[5], _ = _summarize(0, maxDets=20)
            stats[6], _ = _summarize(0, maxDets=20, iouThr=.5)
            stats[7], _ = _summarize(0, maxDets=20, iouThr=.75)
            stats[8], _ = _summarize(0, maxDets=20, areaRng='medium')
            stats[9], _ = _summarize(0, maxDets=20, areaRng='large')
            return stats

        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()

    def __str__(self):
        self.summarize()