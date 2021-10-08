# flip_images_segments_bboxes

This is a *python* code for data augmentation. First, it flips all the images horizontally and saves them to the same folder. In this way, the number of images is doubled. Then, it does the computation of segmentations and bboxes for new flipped images. Lastly, it creates a new COCO Annotation json file which contains both annotations for original images and annotations for flipped images.    

### What you need to run

 You need two things to run this code correctly; 
 
* A COCO annotation JSON file
* A folder that contains images annotated in the COCO JSON file. 






