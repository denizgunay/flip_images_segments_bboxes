import numpy as np
import copy
import json
import cv2 

"""
old_jsonpath : the json location to retrieve the data.
updated_jsonpath : the location for saving updated json file.
image_folder : the location for folder containing images.
json_spec : shows what changed.
"""

#It adds 'FH_' to the image name after flip horizontal.
#If any error raised, first check parameters old_jsonpath, updated_jsonpath and image_folder. 

def flip_horizontal(old_jsonpath = 'sag_sol_train_and_val-19_1.json', updated_jsonpath = 'UPDATED_data.json', image_folder='datasets/sag_sol_train_and_val/',json_spec = False):
    

    def load_json(j):    
        myjson = open(j,'r')
        old_data = myjson.read()
        obj = json.loads(old_data)
        return obj
    
    
    def write_json(updated_jsonpath, new_data):
        with open(updated_jsonpath,'w') as fp:
            json.dump(new_data,fp)
                        
    
    def flip_images(num_images,image_list,image_folder):    
        images = [] 
        img_id_conv = {}
        flip_img_id_conv = {}
        t = 1
        for i in range(num_images):
            image = copy.deepcopy(image_list[i])
            name = image['file_name']
            path_old = image_folder + name
            flip_name = 'FH_' + name
            path_new = image_folder + flip_name
            img = cv2.imread(path_old)
            img_flip_lr = cv2.flip(img, 1)
            cv2.imwrite(path_new, img_flip_lr)
            flip_image = copy.deepcopy(image)
            img_id_conv[image['id']] = t
            image['id'] = t
            image['path'] = path_old
            flip_img_id_conv[flip_image['id']] = t+1
            flip_image['id'] = t+1
            flip_image['file_name'] = flip_name
            flip_image['path'] = path_new    
            images.append(image)
            images.append(flip_image)
            t+=2        
        return images,img_id_conv,flip_img_id_conv
    
    
    def segmentation_update(s,w):    
        new_segment = []
        for segment in s:
            a1 = np.array([w,0])
            repetitions = int(len(segment)/2)
            a2 = np.tile(a1, (repetitions, 1)).reshape(len(segment),1)
            a2 = a2.astype('float64')
            segment = np.array(segment).reshape(np.array(segment).shape[0],1)
            a3 = np.tile(np.array([1,-1]), (repetitions, 1)).reshape(len(segment),1)
            new_seg = np.multiply(a3,(a2 - segment))
            new_seg = np.round(new_seg.astype(np.float64), 1)
            new_seg = new_seg.ravel().tolist()
            new_segment.append(new_seg)
        return new_segment    
    
    
    def annotator(num_annotation,annotation_list,orig_image_list):            
        f = 1
        annotations = []
        keys = {33:2,2:33,35:14,14:35,36:15,15:36,37:16,16:37,38:18,18:38,
                39:19,19:39,40:21,21:40,41:22,22:41,42:23,23:42,43:24,24:43}           
        for i in range(num_annotation):
            old_ann = copy.deepcopy(annotation_list[i])
            new_ann = copy.deepcopy(old_ann)
            img_id_ann = old_ann['image_id']
            width = next((sub['width'] for sub in orig_image_list if sub['id'] == img_id_ann), 0)        
            if width == 0:
                print("A problem with the 'width' variable in the annotator() function!")
                break         
            new_segment = segmentation_update(old_ann['segmentation'],width)
            old_bbox = old_ann['bbox']
            new_bbox = copy.deepcopy(old_bbox)
            x = new_bbox[0]
            w = new_bbox[2]
            new_bbox[0] = width - (x+w)
            old_cat = old_ann['category_id']
            if old_cat in keys:
                new_ann['category_id'] = keys[old_cat]        
            new_ann['image_id'] = flip_img_id_conv[new_ann['image_id']]
            new_ann['segmentation'] = new_segment
            new_ann['bbox'] = new_bbox
            new_ann['id'] = f
            old_ann['id'] = f+1
            old_ann['image_id'] = img_id_conv[old_ann['image_id']]
            annotations.append(new_ann)
            annotations.append(old_ann)
            f+=2            
        return annotations
    
    
    
    obj = load_json(old_jsonpath)
    num_images = len(obj['images'])
    num_annotation = len(obj['annotations'])
    annotation_list = obj['annotations']
    image_list = obj['images']
    orig_image_list = copy.deepcopy(image_list)
    categories = obj['categories']    
    images,img_id_conv,flip_img_id_conv = flip_images(num_images,orig_image_list,image_folder) 
    annotations = annotator(num_annotation,annotation_list,orig_image_list)
 

    
    if json_spec:
        print('---BEFORE AUGMENTATION---')
        print(f'Number of images = {num_images}')
        print(f'Number of annotations = {num_annotation}')
        print()
        print('---AFTER AUGMENTATION---')
        print(f'Number of images = {len(images)}')
        print(f'Number of annotations = {len(annotations)}')
  


    data = {}
    data['images'] = images
    data['categories'] = categories
    data['annotations'] = annotations
    return write_json(updated_jsonpath, data)