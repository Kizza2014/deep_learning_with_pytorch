from torch.utils.data import Dataset
import os
from xml.etree import ElementTree as ET
from PIL import Image
import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image
import torchvision

class ImageDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.image_files = self._filter_images_with_single_object()

    def _filter_images_with_single_object(self):
        valid_image_files = []
        for file in os.listdir(self.image_dir):
            if os.path.isfile(os.path.join(self.image_dir, file)):
                image_name = file
                annotation_name = os.path.splitext(image_name)[0] + '.xml'
                annotation_path = os.path.join(self.annotation_dir, annotation_name)

                if self._count_object_in_annotation(annotation_path) <= 1:
                    valid_image_files.append(image_name)
                else:
                    print(f'Image {image_name} has more than 1 object and will be excluded from dataset.')
        return valid_image_files
    
    def _count_object_in_annotation(self, annotation_path):
        try:
            tree = ET.parse(annotation_path)
            root = tree.getroot()
            count = 0
            for obj in root.findall('object'):
                count += 1
            return count
        except FileNotFoundError:
            return 0

    def _parse_annotation(self, annotation_path, normalize=True):
        try:
            tree = ET.parse(annotation_path)
            root = tree.getroot()
            image_width = int(root.find('size/width').text)
            image_height = int(root.find('size/height').text)
            label = None
            bbox = None
            for obj in root.findall('object'):
                name = obj.find('name').text
                # we only consider image with 1 object at the moment
                if label is None:
                    label = name
                    break

            xmin = int(obj.find('bndbox/xmin').text)
            xmax = int(obj.find('bndbox/xmax').text)
            ymin = int(obj.find('bndbox/ymin').text)
            ymax = int(obj.find('bndbox/ymax').text)

            if normalize:
                bbox = [xmin / image_width, ymin / image_height,
                        xmax / image_width, ymax / image_height]
            else:
                bbox = [xmin, ymin, xmax, ymax]

            # convert label to numerical representation
            label_num = 0 if label == 'cat' else 1 if label == 'dog' else -1
            return label_num, torch.tensor(bbox, dtype=torch.float32 if normalize else torch.uint8)
        except FileNotFoundError:
            print(f'File {annotation_path} not found.')

    def _get_item_size(self, idx):
        try:
            annotation_name = os.path.splitext(self.image_files[idx])[0] + '.xml'
            annotation_path = os.path.join(self.annotation_dir, annotation_name)
            tree = ET.parse(annotation_path)
            root = tree.getroot()
            image_width = int(root.find('size/width').text)
            image_height = int(root.find('size/height').text)
            return image_height, image_width
        except FileNotFoundError:
            print(f'File {annotation_path} not found.')

    def get_item_with_bbox(self, idx, to_PIL_image=False):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        img = read_image(img_path)

        # bbox
        annotation_name = os.path.splitext(self.image_files[idx])[0] + '.xml'
        annotation_path = os.path.join(self.annotation_dir, annotation_name)
        bbox = self._parse_annotation(annotation_path, normalize=False)[1]
        bbox.unsqueeze_(0)

        img = draw_bounding_boxes(img, bbox, width=5, colors="green", labels=['ground truth'])

        if to_PIL_image:
            img = torchvision.transforms.ToPILImage()(img)
        return img
    
    def get_item_with_predicted_bbox(self, idx, predicted_bbox, include_gt_bbox=False, to_PIL_image=False):
        if include_gt_bbox:
            img = self.get_item_with_bbox(idx)
        else:
            img_path = os.path.join(self.image_dir, self.image_files[idx])
            img = read_image(img_path)

        # draw predicted bbox
        image_height, image_width = self._get_item_size(idx)
        x_min = int(predicted_bbox[0] * image_width)
        y_min = int(predicted_bbox[1] * image_height)
        x_max = int(predicted_bbox[2] * image_width)
        y_max = int(predicted_bbox[3] * image_height)
        box = torch.tensor([x_min, y_min, x_max, y_max]).unsqueeze_(0)

        img = draw_bounding_boxes(img, box, width=5, colors='red', labels=['predicted'])

        if to_PIL_image:
            img = torchvision.transforms.ToPILImage()(img)
        return img        

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        annotation_name = os.path.splitext(image_name)[0] + '.xml'
        annotation_path = os.path.join(self.annotation_dir, annotation_name)

        # load image
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        label, bbox = self._parse_annotation(annotation_path)
        return image, label, bbox