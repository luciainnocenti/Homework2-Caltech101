
from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')

        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''

        #Generate a list containing all the classes available in the dataset and a list of index for the classes
        classes, class_to_idx = self._find_classes(self.root)

        self.loader = loader
        self.classes = classes
        self.class_to_idx = class_to_idx

        self.classes_to_idx.remove( classes.index('BACKGROUND_Google'))
        self.classes.remove('BACKGROUND_Google') 

        #Open and read the file containing all the elements of the split set
        pathsplit = root + split + ".txt"
        f = open(pathsplit, 'r')
        lines = f.readlines()

        items = []
        items_as_string = []
        #Generate a list of elements name
        for line in Lines:
            if(line.split("/")[0] != "BACKGROUND_Google") :
              items_as_string.append(line)
              items.append(pil_loader( root + "/101_ObjectCategories/" + line))
        f.close()

def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''
        #Find the string (label + img_name) corrispondent to index passed as input
        item = self.items_as_string[index]
        
        #Divide the item into label (that is the class) and image name
        label, image_name = item.split("/")  

        #By the index, access directly the img file
        image = self.items[index]    

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        #By the label, return the index of the class in the list of all the classes for the dataset
        target = self.classes.index(label)

        return image, target

def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.items)
        return length


