#!/usr/bin/env bash

# use this script in the destination folder.

wget http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
tar -xf VOCtrainval_11-May-2012.tar
mkdir PascalVOC12
mv VOCdevkit/VOC2012/* PascalVOC12
cd PascalVOC12
wget http://cs.jhu.edu/~cxliu/data/SegmentationClassAug.zip
wget http://cs.jhu.edu/~cxliu/data/SegmentationClassAug_Visualization.zip
wget http://cs.jhu.edu/~cxliu/data/list.zip
unzip SegmentationClassAug.zip
unzip SegmentationClassAug_Visualization.zip
unzip list.zip
mv list splits
