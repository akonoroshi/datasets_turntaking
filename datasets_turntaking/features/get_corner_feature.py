# import some common libraries
import numpy as np
import os, json, cv2, random
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from detectron2.modeling.meta_arch import GeneralizedRCNN

def corner_batchver(predictor, image, return_pooled_only = True):
    '''
    function to get corner feature
        predictor: Detectron predictor object
        iamige: numpy array of images
        return_pooled_only: if true return only pooled tensor
    '''
    data_len, height, width, channel = image.shape
    
    # preprocessing for images, different from closeup
    images_input = []
    for i in range(data_len):
        in_img = predictor.aug.get_transform(image[i]).apply_image(image[i])
        images_input.append({'image':torch.from_numpy(np.transpose(in_img, (2, 0, 1))), 'height': height, 'width': width})
    
    batch_s = 16
    num_batches = data_len // batch_s
    
    i = 0
    updated_images_list = []
    pooled_list = []
    detector_results_list = []

    pbar = tqdm(total = data_len)
    while i < data_len: 

        with torch.no_grad():

            # minibatch the image
            #print(i)
            #print(np.array(images_input[i:i+batch_s]).shape)
            images = predictor.model.preprocess_image(images_input[i:i+batch_s])
            updated_images_list.append(images_input[i:i+batch_s])
            features = predictor.model.backbone(images.tensor)
            proposals, _ = predictor.model.proposal_generator(images, features)
            detector_results, _ = predictor.model.roi_heads(images, features, proposals)
            detector_results = GeneralizedRCNN._postprocess(detector_results, images_input[i:i+batch_s], images.image_sizes)

            # print(detector_results)
            # print(len(detector_results))
            
            pbar.update(batch_s)

            detector_results_list.append(detector_results)
            

            # for each detector result, create a pooled vector
            for b in range(len(detector_results)):
                
                # print(f'processing batch {b+1}')
                
                #person = detector_results[b].pred_classes == 0
                person = detector_results[b]['instances'].pred_classes == 0

                
                N = len(detector_results[b]['instances'].pred_masks[person])
                if N > 0: # if there is a person detected we add value to the zero array

                    masks = detector_results[b]['instances'].pred_masks[person, ..., :, :]
                    scores = detector_results[b]['instances'].scores[person]
                    img_masks = torch.sum(masks * scores.unsqueeze(-1).unsqueeze(-1).expand_as(masks), dim=0)
                else:
                    print('batch_id',i ,'b', b)
                    print(detector_results[b])
                
                pooled = torch.nn.functional.max_pool2d(img_masks.unsqueeze(0).unsqueeze(0), 4)
                pooled = torch.squeeze(pooled)
                pooled_flat = torch.flatten(pooled)
                # print(pooled_flat.shape)
                pooled_list.append(pooled_flat)
            i += batch_s
            pbar.update(batch_s)
                
    # stack the list, shape = (batchsize x (image H x W)) image HxW should be 6336
    pooled_stack = torch.stack(pooled_list) 
    pbar.close()

    if return_pooled_only:
        return pooled_stack
    else:
        return detector_results_list, pooled_stack, updated_images_list
