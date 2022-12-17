from argparse import ArgumentParser
import os
from os.path import expanduser, join, exists
from os import listdir, cpu_count
import re
import shutil
import librosa
import math
import gc
from typing import Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from datasets import load_from_disk, load_dataset
from datasets import Sequence
import pytorch_lightning as pl
import numpy as np
import cv2 as cv
import ffmpeg
import logging


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
            
            image: torch tensor (batch_size x 1024)
            return_pooled_only: if true return only pooled tensor
        '''

       
        print(image.shape)
        num_batches, data_len, height, width, channel = image.shape
        
        # convert to numpy
        image_numpys = image.numpy()

        # preprocessing for images, different from closeup

        images_batch = []
        for b in range(num_batches):

            images_per_batch = []

            for i in range(data_len):
                image_array = image_numpys[b][i].copy()
                in_img = self.predictor.aug.get_transform(image_array).apply_image(image_array)
                image_tensor =  torch.from_numpy(np.transpose(in_img, (2, 0, 1)))
                images_per_batch.append({'image': image_tensor, 'height': height, 'width': width})
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
                    features = self.predictor.model.backbone(images.tensor.type(torch.cuda.FloatTensor)) 
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
            images: torch.tensor after pad_sequence (batch_size x 1024 x H x W x 3)
            final shape: image crop size (H x W)
        returns:
            feature tensor: torch.tensor shape of (batch_size x 1024 x 1 x 6)
        '''
        # images = torch.stack(images) # we do not need to stack agin 
        
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
            images: torch.tensor after pad_sequence (batch_size x 1024 x H x W x 3)
        returns:
            featur tensor: torch.tensor shape of (batch_size x 1024 x 6336)
        '''
        # images = np.stack(images)

        feature_tensor = self.corner_batchver(images)  
        del images
        return feature_tensor

# newly added <---


# Original -->

CACHE_PATH = join(expanduser("~"), ".cache/datasets_turntaking/conversational")
class ConversationalDM2(pl.LightningDataModule):
    
    def __init__(
        self,
        tokenizer,
        datasets='ami',
        datasets_subset = 'headset-single',
        savepath='/ocean/projects/cis220078p/stomita/project/dataset',
        tensorpath = '/ocean/projects/cis220078p/stomita/project/dataset_tensor/',
        videodirpath = '/ocean/projects/cis220078p/yasano/amicorpus/',
        batch_size=16,
        max_length=1024,
        num_workers=5,
        pin_memory=True,
        overwrite=False,
        include_dialog=False,
        load_from_cache_file=True,
        num_proc=5,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.pin_memory = pin_memory
        self.num_workers = num_workers

        # `datasets` parameters
        self.load_from_cache_file = load_from_cache_file
        self.num_proc = num_proc
        self.include_dialog = include_dialog

        # Datasets
        self.datasets = datasets
        self.datasets_subset = datasets_subset

        self.savepath = savepath
        self.tensor_path = tensorpath
        self.videodir_path = videodirpath
        self.overwrite = overwrite

        # Get Features
        self.get_feature = Get_Feature()
        

    def get_split_path(self, split):
        pass
        # return join(self.savepath, split)

    def filter_empty_turns(self, examples):
        """
        return only dialogs with no empty turns
        """
        for utterance in examples["dialog"]:
            if utterance == "" or not re.search(r"\w", utterance):  # utt is empty
                return False
        return True

    def get_video(self, word_id, video_type): 
        path_video = self.videodir_path + word_id + '/video/'
        counter = 0
        # sometimes videocapture failed to get video, but somehow loading video multiple times works
        # so here video will be loaded until loaded correctly
        while True:
            video = cv.VideoCapture(path_video + word_id + '.' + video_type + '.avi', cv.CAP_ANY)
            if video.isOpened():
                break
            if counter > 100:
                print(f'loading video {path_video + word_id + "." + video_type + ".avi"}')
                raise RuntimeError(f'loading video {path_video + word_id + "." + video_type + ".avi"}, unable to load video')
            counter += 1
        return video


    def get_frame(self, video, word_start_times, word_end_times, fps=25, frame_definiton='end'):
        """
        inputs: 
          video (cv.VideoCapture) : a video to get a frame 
          word_start_times (float): the starting time of a word
          word_end_times   (float): the ending time of a word
          fps (int): fps of video. In AMI corpus, it is 25.
          frame_definiton (str, 'end' or 'middle'): if 'end', it will return the last frame. 
                                                    if 'middle', it will return the frame in the middle. 
        """
        # convert times to ms scale
        time_start = float(1000*word_start_times)
        time_end = float(1000*word_end_times)
        
        # if we get middle time 
        if frame_definiton == 'middle':
          time_end = (time_start+time_end)/2

        # set time to the ending time
        video.set(cv.CAP_PROP_POS_MSEC, time_end-fps)
        available, frame = video.read()
        if available:
          return frame
        else:
          # return zero array with shape of frame in AMI
          return np.zeros([288, 352, 3])


    def encode(self, examples):
        return self.tokenizer(examples)

    def prepare_data(self, skip_preprocessing = True):
        if not skip_preprocessing:
            # in PSC you need to activate ffmpeg module
            os.system('module load ffmpeg')
            
            if not os.path.exists(self.savepath):
                datasets = load_dataset(self.datasets, self.datasets_subset)
                for split in ["train", "validation", "test"]:
                    dataset = datasets[split]
                    if split == 'train':
                        dataset = dataset.select([i for i in range(114)])
                    elif split == 'validation':
                        dataset = dataset.select([i for i in range(12)])
                    elif split == 'test':
                        dataset = dataset.select([i for i in range(11)])
                    dataset = dataset.map(
                        self.encode,
                        num_proc=self.num_proc,
                    )
                    dataset.save_to_disk(os.path.join(self.savepath, split))

            for split in ["train", "validation", "test"]:
                tensor_path = self.tensor_path + split
                dataset = load_from_disk(os.path.join(self.savepath, split))
                for i in range(len(dataset)):
                    for j in range(len(dataset[i]['word'])):
                        save_dict_name = split + str(i) + 'part' + str(j) + '.npz'
                        data_path = os.path.join(tensor_path, save_dict_name)
                        if not os.path.exists(data_path):
                            dataset = load_from_disk(os.path.join(self.savepath, split))
                            words = dataset[i]['word'][j]
                            speakers = dataset[i]['speaker_ids'][j]
                            starting_time_list = list(dataset[i]['word_start_times'][j])
                            ending_time_list = list(dataset[i]['word_start_times'][j])
                            word_ids_list = list(dataset[i]['word_ids'][j])
                            frame_closeup1_list = []
                            frame_closeup2_list = []
                            frame_closeup3_list = []
                            frame_closeup4_list = []
                            frame_corner_list = []
                                
                            prev_word_ids = word_ids_list[0]
                            video = self.get_video(prev_word_ids, 'Closeup1')
                            for k in range(len(word_ids_list)):
                                if prev_word_ids != word_ids_list[k]:
                                    prev_word_ids =word_ids_list[k]
                                    video = self.get_video(prev_word_ids, 'Closeup1')
                                frame_closeup1 = self.get_frame(video, starting_time_list[k], ending_time_list[k], frame_definiton='end') 
                                frame_closeup1_list.append(frame_closeup1)
                                del frame_closeup1
                            del video
                            gc.collect()
                            print('finish Closeup1')

                            prev_word_ids = word_ids_list[0]
                            video = self.get_video(prev_word_ids, 'Closeup2')
                            for k in range(len(word_ids_list)):
                                if k % 500 == 0:
                                    print(k)
                                if prev_word_ids != word_ids_list[k]:
                                    prev_word_ids = word_ids_list[k]
                                    video = self.get_video(prev_word_ids, 'Closeup2')
                                frame_closeup2 = self.get_frame(video, starting_time_list[k], ending_time_list[k], frame_definiton='end') 
                                frame_closeup2_list.append(frame_closeup2)
                                del frame_closeup2
                            del video
                            gc.collect()
                            print('finish Closeup2')
                                
                            prev_word_ids = word_ids_list[0]
                            video = self.get_video(prev_word_ids, 'Closeup3')
                            for k in range(len(word_ids_list)):
                                if prev_word_ids != word_ids_list[k]:
                                    prev_word_ids = word_ids_list[k]
                                    video = self.get_video(prev_word_ids, 'Closeup3')
                                frame_closeup3 = self.get_frame(video, starting_time_list[k], ending_time_list[k], frame_definiton='end') 
                                frame_closeup3_list.append(frame_closeup3)
                                del frame_closeup3
                            del video
                            gc.collect()
                            print('finish Closeup3')
                                
                            prev_word_ids = word_ids_list[0]
                            video = self.get_video(prev_word_ids, 'Closeup4')
                            for k in range(len(word_ids_list)):
                                if prev_word_ids != word_ids_list[k]:
                                    prev_word_ids = word_ids_list[k]
                                    video = self.get_video(prev_word_ids, 'Closeup4')
                                frame_closeup4 = self.get_frame(video, starting_time_list[k], ending_time_list[k], frame_definiton='end') 
                                frame_closeup4_list.append(frame_closeup4)
                                del frame_closeup4
                            del video
                            gc.collect()
                            print('finish Closeup4')
                                
                            prev_word_ids = word_ids_list[0]
                            if prev_word_ids[:2] == 'ES':
                                video = self.get_video(prev_word_ids, 'Corner')
                            elif prev_word_ids[:2] == 'IS':
                                video = self.get_video(prev_word_ids, 'C')
                            elif prev_word_ids[:2] == 'TS':
                                if (prev_word_ids == 'TS3008b') | (prev_word_ids == 'TS3008c') | (prev_word_ids == 'TS3008d'):
                                    video = self.get_video(prev_word_ids, 'Overview3')
                                else:
                                    video = self.get_video(prev_word_ids, 'Overview2')

                            for k in range(len(word_ids_list)):
                                if prev_word_ids != word_ids_list[k]:
                                    prev_word_ids = word_ids_list[k]
                                    if prev_word_ids[:2] == 'ES':
                                        video = self.get_video(prev_word_ids, 'Corner')
                                    elif prev_word_ids[:2] == 'IS':
                                        video = self.get_video(prev_word_ids, 'C')
                                    elif prev_word_ids[:2] == 'TS':
                                        if (prev_word_ids == 'TS3008b') | (prev_word_ids == 'TS3008c') | (prev_word_ids == 'TS3008d'):
                                            video = self.get_video(prev_word_ids, 'Overview3')
                                        else:
                                            video = self.get_video(prev_word_ids, 'Overview2')
                                frame_corner = self.get_frame(video, starting_time_list[k], ending_time_list[k], frame_definiton='end') 
                                frame_corner_list.append(frame_corner)
                                del frame_corner
                            del video
                            gc.collect()

                            input_ids = np.array(words)
                            speaker_ids = np.array(speakers)                        
                            closeup1 = np.array(frame_closeup1_list)
                            closeup2 = np.array(frame_closeup2_list)
                            closeup3 = np.array(frame_closeup3_list)
                            closeup4 = np.array(frame_closeup4_list)
                            corner = np.array(frame_corner_list)
                                
                            del words, speakers, frame_closeup1_list, frame_closeup2_list, frame_closeup3_list, frame_closeup4_list, frame_corner_list 
                            gc.collect()
                                                        
                            np.savez_compressed(data_path, input_ids = input_ids, 
                                                speaker_ids = speaker_ids, closeup1 = closeup1,
                                                closeup2 = closeup2, closeup3 = closeup3, 
                                                closeup4 = closeup4, corner = corner)
                            
                            del input_ids, speaker_ids, closeup1, closeup2, closeup3, closeup4, corner
                            gc.collect()
                
        train_dir_path = os.path.join(self.tensor_path, 'train')
        # so far we moved validation data to different folder
        #val_dir_path = os.path.join(self.tensor_path, 'validation')
        val_dir_path = '/ocean/projects/cis220078p/yasano/amicorpus/validation'
        test_dir_path = os.path.join(self.tensor_path, 'test')
        
        dataset_list_train = self.get_dset_paths(train_dir_path)
        dataset_list_val = self.get_dset_paths(val_dir_path)
        dataset_list_test = self.get_dset_paths(test_dir_path)
        
        self.train_dset = dataset_list_train
        self.val_dset = dataset_list_val
        self.test_dset = dataset_list_test

    def get_dset_paths(self, directory):
        """
        Args:
            directory (str): name of the directory
        Outputs:
            file_path_list (list): list of absolute path in the directory
        """
        file_path_list = []
        for dirpath,_,filenames in os.walk(directory):
            for f in filenames:
                file_path_list.append(os.path.join(dirpath, f))
        return file_path_list
                

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        pass
        """
        if stage == "fit" or stage is None:
            self.train_dset = load_from_disk(self.get_split_path("train"))
            self.val_dset = load_from_disk(self.get_split_path("validation"))

        if stage == "test":
            self.test_dset = load_from_disk(self.get_split_path("test"))
        """

    def collate_fn(self, batch):
        # 'batch' will be iterable of path name
        # load data here
        batch_dict = [np.load(path) for path in batch]
        path_dict = [path for path in batch]
        
        input_word = [torch.tensor(b["input_ids"]) for b in batch_dict] # list of tensor(1024)
        input_speaker = [torch.tensor(b["speaker_ids"]) for b in batch_dict] # list of tensor(1024)
        input_closeup1 = [torch.tensor(b['closeup1']) for b in batch_dict] # list of tensor(1024 * H * W * 3)
        input_closeup2 = [torch.tensor(b['closeup2']) for b in batch_dict] # list of tensor(1024 * H * W * 3)
        input_closeup3 = [torch.tensor(b['closeup3']) for b in batch_dict] # list of tensor(1024 * H * W * 3)
        input_closeup4 = [torch.tensor(b['closeup4']) for b in batch_dict] # list of tensor(1024 * H * W * 3)
        input_corner = [torch.tensor(b['corner']) for b in batch_dict] # list of tensor(1024 * H * W * 3)


        # before padding everything, create original attention_mask without padding
        attention_mask_list = [torch.ones_like(word) for word in input_word]
        
        # in case all tensor in the batch is shorter than 1024, padding the first entity 
        if len(input_word[0]) != self.max_length:
            input_word[0] = torch.nn.functional.pad(input_word[0], (0, self.max_length-len(input_word[0])), 'constant', self.tokenizer.tokenizer.pad_token_id)
            input_closeup1[0] = torch.nn.functional.pad(input_closeup1[0].permute([1,2,3,0]), (0, self.max_length-len(input_word[0])), 'constant', 0).permute([3,0,1,2])
            input_closeup2[0] = torch.nn.functional.pad(input_closeup2[0].permute([1,2,3,0]), (0, self.max_length-len(input_word[0])), 'constant', 0).permute([3,0,1,2])
            input_closeup3[0] = torch.nn.functional.pad(input_closeup3[0].permute([1,2,3,0]), (0, self.max_length-len(input_word[0])), 'constant', 0).permute([3,0,1,2])
            input_closeup4[0] = torch.nn.functional.pad(input_closeup4[0].permute([1,2,3,0]), (0, self.max_length-len(input_word[0])), 'constant', 0).permute([3,0,1,2])
            input_corner[0] = torch.nn.functional.pad(input_corner[0].permute([1,2,3,0]), (0, self.max_length-len(input_word[0])), 'constant', 0).permute([3,0,1,2])

        # debug
        debug = True
        if debug:
            print('debug:') 
            for i in range(len(input_closeup1)):
                print(path_dict[i])
                print(input_closeup1[i].shape)
        
        # pad_sequence to input_word
        input_word_pad = pad_sequence(input_word, batch_first = True, padding_value=self.tokenizer.tokenizer.pad_token_id)
        input_closeup1_pad = pad_sequence(input_closeup1, batch_first = True, padding_value=0)
        input_closeup2_pad = pad_sequence(input_closeup2, batch_first = True, padding_value=0)
        input_closeup3_pad = pad_sequence(input_closeup3, batch_first = True, padding_value=0)
        input_closeup4_pad = pad_sequence(input_closeup4, batch_first = True, padding_value=0)
        input_corner_pad = pad_sequence(input_corner, batch_first = True, padding_value=0)
        
#         #           --------------------------------------------------------    Get  Feature    -----------------------------------------------------
#         ### get features from corner, shape = batch_size x 1024 x 6336
#         feature_corner = self.get_feature.get_corner_feature(input_corner_pad)        
        
#         ### get features from closeup, shape = batch_size x 1024 x 6
#         feature_closeup1 = self.get_feature.get_closeup_feature(input_closeup1_pad)
#         feature_closeup2 = self.get_feature.get_closeup_feature(input_closeup2_pad)
#         feature_closeup3 = self.get_feature.get_closeup_feature(input_closeup3_pad)
#         feature_closeup4 = self.get_feature.get_closeup_feature(input_closeup4_pad)
        
#         # ------------------------------------------------
        
        
        if debug:
            print('closeup features:')
            print(input_closeup1_pad.shape)
            print(input_closeup2_pad.shape)
            print(input_closeup3_pad.shape)
            print(input_closeup4_pad.shape)
            print('corner features:')
            print(input_corner_pad.shape)
            


        # since padding_mode = 'replicate' didn't work, let's do it manually...
        # create a tensor to store the result
        input_speaker_pad = torch.zeros_like(input_word_pad)
        for i in range(len(input_speaker)):
          input_speaker_element = torch.nn.functional.pad(input_speaker[i], (0, self.max_length-len(input_speaker[i])), 'constant', input_speaker[i][-1].item())
          input_speaker_pad[i] = input_speaker_element

        # create a tensor to store the result
        attention_mask = torch.zeros_like(input_word_pad)
        for i in range(len(attention_mask_list)):
          attention_mask_element = torch.nn.functional.pad(attention_mask_list[i], (0, self.max_length-len(attention_mask_list[i])), 'constant', 0)
          attention_mask[i] = attention_mask_element
        
        del input_word, input_speaker, input_closeup1, input_closeup2, input_closeup3, input_closeup4, input_corner
        gc.collect()
        
        return {'input_ids': input_word_pad, 'speaker_ids': input_speaker_pad, 'attention_mask': attention_mask,
                'closeup1': input_closeup1_pad, 'closeup2': input_closeup2_pad, 'closeup3': input_closeup3_pad, 'closeup4': input_closeup4_pad,
                'corner': input_corner_pad}
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dset,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dset,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        n_cpus = cpu_count()
        parser.add_argument(
            "--datasets", type=str, nargs="+", default="ami"
        )
        parser.add_argument("--savepath", default=None, type=str)
        parser.add_argument("--overwrite", default=False, type=bool)
        parser.add_argument(
            "--max_length",
            default=1024,
            type=int,
            help="maximum length of sequences (applied in `collate_fn`)",
        )
        # arguments for `datasets` library
        parser.add_argument("--batch_size", default=4, type=int)
        parser.add_argument("--num_workers", default=n_cpus, type=int)
        parser.add_argument("--pin_memory", default=True, type=bool)
        parser.add_argument("--load_from_cache_file", default=True, type=bool)
        parser.add_argument("--num_proc", default=n_cpus, type=int)
        parser.add_argument("--tensorpath", default='/ocean/projects/cis220078p/stomita/project/dataset_tensor/')
        
        return parser




