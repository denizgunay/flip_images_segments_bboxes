import numpy as np
import copy
import json
import cv2 

# It adds 'FH_' to the image name after flip horizontal.
# If any error raised, first check parameters old_jsonpath, updated_jsonpath and image_folder.

# Parameters;
# old_jsonpath : the json location to retrieve the data.
# updated_jsonpath : the location for saving updated json file.
# image_folder : the location for folder containing images.
# json_spec : this is an optional parameter and shows the number of images and annotations before and after augmentation.



def flip_horizontal(old_jsonpath = 'sag_sol_train_and_val-19_1.json', updated_jsonpath = 'UPDATED_data.json', image_folder='datasets/sag_sol_train_and_val/',json_spec = True):
    

    def load_json(j):    
        myjson = open(j,'r')
        old_data = myjson.read()
        obj = json.loads(old_data)
        return obj
    
    
    def write_json(updated_jsonpath, new_data):
        with open(updated_jsonpath,'w') as fp:
            json.dump(new_data,fp)
                        
    
    def flip_images(image_list,image_folder):
        images = []
        img_id_conv = {}
        flip_img_id_conv = {}
        num_images = len(image_list)
        t = 1
        for i in range(num_images):
            image = copy.deepcopy(image_list[i])
            name = image['file_name']
            path = image_folder + name
            image_already_flipped = next((image['file_name'] for image in image_list if image['file_name'] == 'FH_' + name), False)            
            if image_already_flipped:
                img_id_conv[image['id']] = t
                image['id'] = t
                image['path'] = path
                images.append(image)
                t+=1                
                continue
            elif name.startswith('FH_'):
                img_id_conv[image['id']] = t
                image['id'] = t
                image['path'] = path
                images.append(image)
                t+=1                
                continue
            else:
                flip_image = copy.deepcopy(image)
                flip_name = 'FH_' + flip_image['file_name']
                flip_image['file_name'] = flip_name
                flip_path = image_folder + flip_name
                flip_image['path'] = flip_path
                image['path'] = path
                img = cv2.imread(path)
                img_flip_lr = cv2.flip(img, 1)
                cv2.imwrite(flip_path, img_flip_lr) 
                img_id_conv[image['id']] = t
                image['id'] = t
                flip_img_id_conv[flip_image['id']] = t+1
                flip_image['id'] = t+1
                images.append(image)
                images.append(flip_image)
                t+=2
                continue
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
    
    
    def annotator(annotation_list,orig_image_list):            
        f = 1
        annotations = []
        num_annotation = len(annotation_list)
        keys = {33:2,2:33,35:14,14:35,36:15,15:36,37:16,16:37,38:18,18:38,
                39:19,19:39,40:21,21:40,41:22,22:41,42:23,23:42,43:24,24:43}           
        for i in range(num_annotation):
            old_ann = copy.deepcopy(annotation_list[i])
            new_ann = copy.deepcopy(old_ann)
            img_id_ann = old_ann['image_id']
            if bool(flip_img_id_conv):
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
                continue
            else:
                old_ann['id'] = f
                old_ann['image_id'] = img_id_conv[old_ann['image_id']]
                annotations.append(old_ann)
                f+=1
                continue                
        return annotations
    
    
    obj = load_json(old_jsonpath)
    annotation_list = obj['annotations']    
    image_list = obj['images']
    orig_image_list = copy.deepcopy(image_list)
    categories = obj['categories']    
    images,img_id_conv,flip_img_id_conv = flip_images(orig_image_list,image_folder)   
    annotations = annotator(annotation_list,orig_image_list)
    

    
    if json_spec:
        print('---BEFORE AUGMENTATION---')
        print(f'Number of images = {len(image_list)}')
        print(f'Number of annotations = {len(annotation_list)}')
        print()
        print('---AFTER AUGMENTATION---')
        print(f'Number of images = {len(images)}')
        print(f'Number of annotations = {len(annotations)}')
  

    
    data = {}
    data['images'] = images
    data['categories'] = categories
    data['annotations'] = annotations
    write_json(updated_jsonpath, data)
    return
