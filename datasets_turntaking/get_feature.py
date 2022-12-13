
import matplotlib.pyplot as plt
import numpy as np
# from google.colab.patches import cv2_imshow
from tqdm import tqdm

# import some common libraries
import numpy as np
import os, json, cv2, random
from PIL import Image
import torch
import torch.nn as nn
import gc

# # Some basic setup:
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# # import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.structures import Boxes
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.layers.mask_ops import _do_paste_mask
from detectron2.modeling.meta_arch import GeneralizedRCNN

# 6DRepNet for headpose

import sys
 
# adding Folder_2/subfolder to the system path
# sys.path.insert(0, '/content/drive/MyDrive/CMU Courses/11785 Introduction to Deep Learning/Project/SixDRepNet/')
sys.path.insert(0, '/content/drive/MyDrive/Project/SixDRepNet/')

from sixdrepnet.regressor import SixDRepNet_Detector
 
from sixdrepnet.model import SixDRepNet
import sixdrepnet.utils as utils

# import prepare_detectron_model

class Get_Feature(nn.Module):
      
    def __init__(self):
        super().__init__()

        # Create models
        # Weights are automatically downloaded
        
        
        # self.sixdrepnet = SixDRepNet.sixdrepnet.SixDRepNet()
        self.sixdrepnet = SixDRepNet_Detector()

        self.model_detectron, self.predictor = self.create_detectron_model()

    ### create model and predictor
    def create_detectron_model(self, threshold = 0.3):
        cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        model_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        cfg.merge_from_file(model_zoo.get_config_file(model_path))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # set threshold for this model

        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_path)
        
        # build predictor (for corner images)
        predictor = DefaultPredictor(cfg)

        # build model (for batch inference in closeup images)
        model_detectron = build_model(cfg)

        # get pretrained weight from model zoo
        DetectionCheckpointer(model_detectron).load(model_zoo.get_checkpoint_url(model_path)) 
        model_detectron.eval() # set to eval

        return model_detectron, predictor


    def onlykeep_person_class(self, outputs):
        '''
        function to get only boxes of `person` class in detectron
            outputs: outputs from Detectron model
        '''
        cls = outputs['instances'].pred_classes
        scores = outputs["instances"].scores
        boxes = outputs['instances'].pred_boxes

        # index to keep whose class == 0
        indx_to_keep = (cls == 0).nonzero().flatten().tolist()

        # only keeping index  corresponding arrays

        boxes1 = Boxes(torch.tensor(np.take(boxes.tensor.cpu().numpy(), indx_to_keep, axis=0)))

        #   # create new instance obj and set its fields
        #   obj = detectron2.structures.Instances(image_size=(image.shape[0], image.shape[1]))
        #   obj.set('pred_boxes',boxes1)
        return boxes1

    def crop_face_detectron_batchver(self, final_shape, image):
        """
        function for cropping picture with face in the center (if face is detected), else center crop
        Input:
            detect_faces: function for face detection, will return face frame if face are detected
            final_shape: image output shape (H,W) channel will be added automatically
            image: numpy array of images

        """
        print('Running crop face')
        num_batches, data_len, width, height, channel = image.shape   # Get dimensions
        print(image.shape)
        # # preprocessing for images
        # image_trans = torch.from_numpy(image).permute(0,1,4,2,3)
        # images_input = [{f'image': image_trans[i]} for i in range(len(image_trans))]
        
        
        images_batch = []
        for b in range(num_batches):

            images_per_batch = []

            for i in range(data_len):

                images_per_batch.append({'image':image[b][i].permute(2, 0, 1)})
                # images_per_batch.append({'image':torch.from_numpy(np.transpose(image[b][i], (2, 0, 1)))})
                # in_img = self.predictor.aug.get_transform(image[b][i]).apply_image(image[b][i])
                # images_per_batch.append({'image':torch.from_numpy(np.transpose(in_img, (2, 0, 1))), 'height': height, 'width': width})
                
            images_batch.append(images_per_batch)
        print('1. Finished preprocessing')
        # print(len(images_batch[0]))
        # print(len(images_batch[1]))

        # run thru detectron
        batch_s = 64

        # num_mini_batches = data_len // batch_s
        gc.collect()
        torch.cuda.empty_cache()
        self.model_detectron.eval()
        
        detected_faces_batch= []
        updated_images_batch = []
        for b in range(num_batches):
            # print(b)
            detected_faces_list = []
            updated_images_list = []
            images_input = images_batch[b]
            i=0
            while i < data_len: 
                # print(i)
                with torch.no_grad():
                    im = images_input[i:i+batch_s]
                    output = self.model_detectron(im)
                    
                    # print(output)
                    detected_faces_list.append(output)
                    updated_images_list.append(im)
                    del im, output
                    torch.cuda.empty_cache()
                    i += batch_s
            # print('u_im',len(updated_images_list))
            detected_faces_batch.append(detected_faces_list)
            updated_images_batch.append(updated_images_list)

        # print('detected_faces', len(detected_faces_batch[1]))
        # print('updated_images', len(updated_images_batch[1]))
        print('2. Finished prediction by Detectron2')
        
        # detected_faces_batch is a list of list of list of dictionary  shape:(num_batches x num_mini_batches x (64 x 16)), each element in list is `instances` dictionary output from detectron
    
        del detected_faces_list, updated_images_list

        # get face crop (have to loop over the entire batch because the detectron output is dictionary)

        face_array_batch = []

        for outer_b in range(num_batches):
            print(f'\tprocessing batch {outer_b +1}')    
            face_array_list = []
            detected_faces_list = detected_faces_batch[outer_b]
            updated_images_list = updated_images_batch[outer_b]
            # if outer_b == 1:
                # print(detected_faces_list)
            for b in range(len(detected_faces_list)):
                # print('counting',outer_b, b )
                
                for i in range(len(detected_faces_list[b])):
                    detected_faces = detected_faces_list[b][i]
                    image_to_crop = updated_images_list[b][i]['image'].permute(1,2,0).numpy()
                    # print('image shape' , image_to_crop.shape)
                    if len(detected_faces['instances'].pred_boxes)> 0: 
                        
                        boxes = self.onlykeep_person_class(detected_faces) # get only person class
                        # print(boxes)
                        if len(boxes)> 1:
                            # boxes = detected_faces['instances'].pred_boxes
                            boxes = boxes.tensor
                            # print(boxes.shape)
                            # Crop faces and plot
                            if (boxes[0][2] -  boxes[0][0]) > (boxes[1][2] - boxes[1][0]):
                                pass
                            else:
                                boxes = boxes[1:]
                        if len(boxes)> 0:

                            for n, face_rect in enumerate(boxes):
                                # print('face_rect', face_rect)
                                face_rect = face_rect.cpu().detach().numpy()
                                mid_x = face_rect[0] + (face_rect[2] - face_rect[0])//2 # midpoint
                                mid_y = face_rect[1] + (face_rect[3] - face_rect[1])//2 # midpoint
                                left = mid_x - (final_shape[0]//2)
                                top = mid_y - (final_shape[1]//2)
                                
                                # c = mid_x + (final_shape[0]//2)
                                # d = mid_y + (final_shape[1]//2)
                                right = left + final_shape[0]
                                bottom = top + final_shape[1]
                                face_rect_new = (left, top, right, bottom)
                                # print(face_rect_new)
                                # # face = image[bottom:right, b:d]
                                # ori_face = image[face_rect[0]:face_rect[2] , face_rect[1]: face_rect[3]]
                                # Image.fromarray(face)
                                # plt.subplot(1, len(detected_faces), n+1)
                                # plt.axis('off')
                                # plt.imshow(face)
                                
                                face = Image.fromarray(image_to_crop).crop(face_rect_new)
                                
                                face_array = np.array(face)
                                # print(face_array.shape)
                                # plt.imshow(face)
                                face_array_list.append(face_array)
                        
                                break
                        else:
                            # if cannot detect face , crop center
        

                            new_width, new_height = final_shape
                            left = (width - new_width)//2
                            top = (height - new_height)//2
                            right = (width + new_width)//2
                            bottom = (height + new_height)//2
                            face_rect_new = (left, top, right, bottom)
                            # print(face_rect_new)

                            # Crop the center of the image
                            face = Image.fromarray(image_to_crop).crop(face_rect_new)
                            face_array = np.array(face)
                            # print(face_array.shape)
                            # plt.imshow(face)
                            face_array_list.append(face_array)
                        
                    else:
                        # if cannot detect face , crop center

                        new_width, new_height = final_shape
                        left = (width - new_width)//2
                        top = (height - new_height)//2
                        right = (width + new_width)//2
                        bottom = (height + new_height)//2
                        face_rect_new = (left, top, right, bottom)
                        # print(face_rect_new)

                        # Crop the center of the image
                        face = Image.fromarray(image_to_crop).crop(face_rect_new)
                        face_array = np.array(face)
                        # print(face_array.shape)
                        # plt.imshow(face)
                        face_array_list.append(face_array)
            # print(face_array_list)
            face_array_stack = np.stack(face_array_list)            
            face_array_batch.append(face_array_stack)

        face_array_batch = np.stack(face_array_batch)

        face_tensor = torch.from_numpy(face_array_batch)
        print('finished getting face tensor')
        return face_tensor




    def corner_batchver(self, image, return_pooled_only = True):
        '''
        function to get corner feature
            
            image: numpy array of images
            return_pooled_only: if true return only pooled tensor
        '''

        print(image.shape)
        num_batches, data_len, height, width, channel = image.shape
        
        # preprocessing for images, different from closeup

        images_batch = []
        for b in range(num_batches):

            images_per_batch = []

            for i in range(data_len):
                in_img = self.predictor.aug.get_transform(image[b][i]).apply_image(image[b][i])
                images_per_batch.append({'image':torch.from_numpy(np.transpose(in_img, (2, 0, 1))), 'height': height, 'width': width})
                # images_per_batch.append({'image':in_img.permute(2, 0, 1), 'height': height, 'width': width})
                
            images_batch.append(images_per_batch)
            
        batch_s = 16
        # num_batches = data_len // batch_s
        
        
        updated_images_list = []
        detector_results_list = []
        i = 0
        pbar = tqdm(total = num_batches)
        pooled_stack_list = []
        detector_results_batch = []
        for b in range(num_batches):

            images_input = images_batch[b]
            pooled_list = []
            
            i = 0 # added here
            while i < data_len: 

                with torch.no_grad():

                    # minibatch the image
                    #print(i)
                    #print(np.array(images_input[i:i+batch_s]).shape)
                    images = self.predictor.model.preprocess_image(images_input[i:i+batch_s])
                    updated_images_list.append(images_input[i:i+batch_s])
                    features = self.predictor.model.backbone(images.tensor)
                    proposals, _ = self.predictor.model.proposal_generator(images, features)
                    detector_results, _ = self.predictor.model.roi_heads(images, features, proposals)
                    detector_results = GeneralizedRCNN._postprocess(detector_results, images_input[i:i+batch_s], images.image_sizes)

                    # print(detector_results)
                    # print(len(detector_results))
                    
                    # pbar.update(batch_s)
                    
                    detector_results_list.append(detector_results)
                    

                    # for each detector result, create a pooled vector
                    for box in range(len(detector_results)):
                        
                        # print(f'processing batch {b+1}')
                        
                        #person = detector_results[b].pred_classes == 0
                        person = detector_results[box]['instances'].pred_classes == 0

                        
                        N = len(detector_results[box]['instances'].pred_masks[person])
                        if N > 0: # if there is a person detected we add value to the zero array

                            masks = detector_results[box]['instances'].pred_masks[person, ..., :, :]
                            scores = detector_results[box]['instances'].scores[person]
                            img_masks = torch.sum(masks * scores.unsqueeze(-1).unsqueeze(-1).expand_as(masks), dim=0)
                        else:
                            print('batch', b, 'seq_len',i ,'b', box)
                            print(detector_results[box])
                        
                        pooled = torch.nn.functional.max_pool2d(img_masks.unsqueeze(0).unsqueeze(0), 4)
                        pooled = torch.squeeze(pooled)
                        pooled_flat = torch.flatten(pooled)
                        # print(pooled_flat.shape)
                        pooled_flat = pooled_flat.cpu().detach()
                        pooled_list.append(pooled_flat)
                        del pooled, pooled_flat
                    i += batch_s
            pbar.update(b)
                    
            # stack the list, shape = (batchsize x (image H x W)) image HxW should be 6336
            pooled_stack = torch.stack(pooled_list) 
            pooled_stack_list.append(pooled_stack)
            detector_results_batch.append(detector_results_list)
            
        pbar.close()
        pooled_batch = torch.stack(pooled_stack_list)

        if return_pooled_only:
            return pooled_batch 
        else:
            return detector_results_batch, pooled_batch

    def get_closeup_feature(self, images, final_shape = (200,200)):
        '''
        function to get closeup feature
            images: list of  torch.tensor (1024 x H x W x 3)
            final shape: image crop size (H x W)
        returns:
            feature tensor: torch.tensor shape of (batch_size x 1024 x 1 x 6)
        '''
        images = torch.stack(images)
        
        # gete face frop
        
        face_tensor = self.crop_face_detectron_batchver(final_shape, images)
        # print('face_tensor',face_tensor.shape)
        # get x_tensor, if GPU good, try predict_batch method
        feature_tensor = self.sixdrepnet.predict_batch(face_tensor)
        del face_tensor
        torch.cuda.empty_cache()
        return feature_tensor
    
    def get_corner_feature(self, images): 
        '''
        function to get corner feature
            images: list of numpy array (torch.tensor does not work in one of the detectron code)
        returns:
            featur tensor: torch.tensor shape of (batch_size x 1024 x 6336)
        '''
        images = np.stack(images)

        feature_tensor = self.corner_batchver(images)  # input must be numpy array
        del images
        return feature_tensor




