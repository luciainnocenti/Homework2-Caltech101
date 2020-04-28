
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

        #Generate a list containing all the classes available in the dataset and a list of index for the classes
        classes = self._find_classes(self.root + "/101_ObjectCategories")

        self.classes = classes
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
  

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        return classes

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
