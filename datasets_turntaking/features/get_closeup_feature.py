import os 
import torch
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from google.colab.patches import cv2_imshow
from PIL import Image
from tqdm import tqdm
from detectron2.structures import Boxes


# 6DRepNet for headpose
import sixdrepnet

def onlykeep_person_class(outputs):
    '''
    function to get only boxes of `person` class in detectron
        outputs: putputs from Detectron model
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



def crop_face_detectron_batchver(model_detectron, final_shape, image):
    """
    function for cropping picture with face in the center (if face is detected), else center crop
    Input:
        detect_faces: function for face detection, will return face frame if face are detected
        final_shape: image output shape (H,W) channel will be added automatically
        image: numpy array of images

    """
    print('Running crop face')
    data_len, width, height, channel = image.shape   # Get dimensions

    # preprocessing for images
    image_trans = torch.from_numpy(image).permute(0,3,1,2)
    images_input = [{f'image': image_trans[i]} for i in range(len(image_trans))]
    print('1. Finished preprocessing')
        
    # run thru detectron
    batch_s = 16
    num_batches = data_len // batch_s
    detected_faces_list = []
    i = 0


    updated_images_list = []
    while i < data_len: 
        with torch.no_grad():
            output = model_detectron(images_input[i:i+batch_s])
            detected_faces_list.append(output)
            updated_images_list.append(images_input[i:i+batch_s])
            torch.cuda.empty_cache()
            i += batch_s
    print('2. Finished prediction by Detectron2')
    
    # detected_faces_list is a list of list tensor (64 x 16), each element in list is `instances` dictionary output from detectron
   

    # get face crop (have to loop over because the detectron output is dictionary)

    face_array_list = []
    for b in range(len(detected_faces_list)):
        print(f'processing batch{b+1}')
        
        for i in range(len(detected_faces_list[b])):
            detected_faces = detected_faces_list[b][i]
            image_to_crop = updated_images_list[b][i]['image'].permute(1,2,0).numpy()
            print('image shape' , image_to_crop.shape)
            if len(detected_faces['instances'].pred_boxes)> 0: 
                
                boxes = onlykeep_person_class(detected_faces, image_to_crop) # get only person class
                print(boxes)
                if len(boxes)> 1:
                    # boxes = detected_faces['instances'].pred_boxes
                    boxes = boxes.tensor
                    print(boxes.shape)
                    # Crop faces and plot
                    if (boxes[0][2] -  boxes[0][0]) > (boxes[1][2] - boxes[1][0]):
                        pass
                    else:
                        boxes = boxes[1:]
                if len(boxes)> 0:

                    for n, face_rect in enumerate(boxes):
                        print('face_rect', face_rect)
                        face_rect = face_rect.cpu().numpy()
                        mid_x = face_rect[0] + (face_rect[2] - face_rect[0])//2 # midpoint
                        mid_y = face_rect[1] + (face_rect[3] - face_rect[1])//2 # midpoint
                        left = mid_x - (final_shape[0]//2)
                        top = mid_y - (final_shape[1]//2)
                        
                        # c = mid_x + (final_shape[0]//2)
                        # d = mid_y + (final_shape[1]//2)
                        right = left + final_shape[0]
                        bottom = top + final_shape[1]
                        face_rect_new = (left, top, right, bottom)
                        print(face_rect_new)
                        # # face = image[bottom:right, b:d]
                        # ori_face = image[face_rect[0]:face_rect[2] , face_rect[1]: face_rect[3]]
                        # Image.fromarray(face)
                        # plt.subplot(1, len(detected_faces), n+1)
                        # plt.axis('off')
                        # plt.imshow(face)
                        
                        face = Image.fromarray(image_to_crop).crop(face_rect_new)
                        
                        face_array = np.array(face)
                        print(face_array.shape)
                        plt.imshow(face)
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
                    print(face_rect_new)

                    # Crop the center of the image
                    face = Image.fromarray(image_to_crop).crop(face_rect_new)
                    face_array = np.array(face)
                    print(face_array.shape)
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
                print(face_rect_new)

                # Crop the center of the image
                face = Image.fromarray(image_to_crop).crop(face_rect_new)
                face_array = np.array(face)
                print(face_array.shape)
                plt.imshow(face)
                face_array_list.append(face_array)
                    
    return face_array_list

def get_feature(model_detectron, images, final_shape = (200,200)):
    '''
    function to get closeup feature
        model_detectron: a Detectron model objecct
        images: images in dictionary format for Detectron inference
        final shape: image crop size (H x W)
    '''
    # Create model
    # Weights are automatically downloaded
    model = sixdrepnet.SixDRepNet()

    # define final shape
     # 200, 200 is good for head pose detection task 

    # gete face frop
    face_tensor = crop_face_detectron_batchver(model_detectron, final_shape, images)

    # get x_tensor, if GPU good, try predict_batch method
    out_tensor, feature_tensor = model.predict_batch(face_tensor)

    return feature_tensor

