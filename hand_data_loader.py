from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import xml.etree.ElementTree as ET


class HandBboxDataset(Dataset):
    """Hand Bounding box dataset."""

    def __init__(self, annot_dir, transform=None):
        """
        Args:
            annot_dir (string): Directory with all the xml files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.annot_dir = annot_dir
        self.transform = transform


        self.all_xmls = []
        for r, d, f in os.walk(self.annot_dir):
            for file in f:
                if file.endswith(".xml"):
                    self.all_xmls.append(os.path.join(r, file))
        
    def __len__(self):
        return len(self.all_xmls)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        xml = self.all_xmls[idx]
        
        record = {}
        height, width = (240,320)
        
        tree = ET.parse(xml)
        root = tree.getroot()
        path = root.find('path').text
        record["file_name"] = path
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        objs = []
        image = io.imread(path)
        
        for boxes in root.iter('object'):
            flg = False
            obj = {"category_id": 1}
            for box in boxes.findall("bndbox"):
                flg = True
                ymin = int(float(box.find("ymin").text))
                xmin = int(float(box.find("xmin").text))
                ymax = int(float(box.find("ymax").text))
                xmax = int(float(box.find("xmax").text))
                obj = {
                "bbox": [xmin, ymin, xmax, ymax],
                "category_id": 0,
            }
                
            if flg == True:
                objs.append(obj)
        record["annotations"] = objs
        record["image"] = image


        if self.transform:
            record = self.transform(record)

        return record