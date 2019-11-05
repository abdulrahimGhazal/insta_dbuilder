# insta_dbuilder


This is a utility tool to get instagram pictures and metadata, with object detection using yolov3.

Note: Sadly you will need to get the yolov3 weights file from the source of yolov3 at:

https://drive.google.com/drive/folders/1uxgUBemJVw9wZsdpboYbzUN4bcRhsuAI

# Future :

* Make downloading the weights automatic.
* Add Manual for users.
* Add exception handling for cases when the instagram retrieval service has some down time.
* include the part where the data is retrieved.
* now we take the dominant colour of only the first frame if the post is a vedio, because it takes time, but maybe this should be up to the user.
* in get_first_frame we need to select the directory to save in.

