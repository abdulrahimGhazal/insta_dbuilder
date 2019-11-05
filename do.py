from __future__ import print_function
import json
import binascii
import struct
from PIL import Image
import numpy as np
import scipy
import scipy.misc
import scipy.cluster
import datetime
import lzma
import os
import io
import requests
import cv2
import yolo_opencv

def get_immediate_files(directory_path):
    """
    This function gets the immediate files of the input directory.

    Parameters
    ----------
    directory_path : str
        The directory path.

    Returns
    -------
    list
        The immediate files names.

    """
    return [name for name in os.listdir(directory_path)
            if not os.path.isdir(os.path.join(directory_path, name))]

def get_json(file_name):
    """
    This function gets the json format from the file.

    Parameters
    ----------
    file_name : str
        The file path.

    Returns
    -------
    object
        The file objects in Json format.

    """
    return json.loads(lzma.open(file_name).read())
# s = get_json('2013-07-22_01-57-08_UTC.json.xz')
# print(s['node']['owner']['username'])

def get_dominant_color(pic_name):
    """
    This function gets the dominant colour in an image given its path.

    Parameters
    ----------
    pic_name : str
        The picture path.

    Returns
    -------
    list
        The peak colour and the dominant colour.

    """
    NUM_CLUSTERS = 5
    im = Image.open(pic_name)
    im = im.resize((150, 150))      # optional, to reduce time
    ar = np.asarray(im)
    shape = ar.shape
    ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

    codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)     # finding clusters

    vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
    counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences

    index_max = scipy.argmax(counts)                    # find most frequent
    peak = codes[index_max]
    for color in peak:
        color = int(color)
    colour = binascii.hexlify(bytearray(int(c) for c in peak)).decode('ascii')
    return  peak, colour
# print(get_dominant_color('2013-07-22_01-57-08_UTC.jpg')[0])

def cal_hashtags(text):
    """
    This function counts how many hashtags exist in the picture description.

    Parameters
    ----------
    text : str
        The picture description.

    Returns
    -------
    int
        The hashtags count.

    """
    words = text.split(' ')
    count = 0
    for word in words:
        if "#" in word:
            count = count + 1
    return count

def get_day_time(pic_name):
    """
    This function uses the picture name provided by Instgram to specify the datetime of the picture.

    Parameters
    ----------
    pic_name : str
        The picture file name.

    Returns
    -------
    datetime
        The datetime of the picture.

    """
    parts = pic_name.split('_')
    date = parts[0].split('-')
    time = parts[1].split('-')
    year = int(date[0])
    month = int(date[1])
    days = int(date[2])
    hours = int(time[0])
    minutes = int(time[1])
    seconds = int(time[2])
    x = datetime.datetime(year, month, days, hours, minutes, seconds)
    return x

def write_string(text, file_path):
    """
    This function writes a text into file.

    Parameters
    ----------
    text : str
        The text to write.
    file_path : str
        The output file path.

    Returns
    -------

    """
    # FileUtil.check_directory(file_path)
    f = io.open(file_path, 'w', encoding="utf-8")
    f.write(text)


def get_first_frame(file_name):
    """
    This function gets the first frame of a vedio (for object detection phase later) and writes it in an image file.

    Parameters
    ----------
    file_name : str
        The picture file name.

    Returns
    -------
    str
        A string telling the state of the saving process.

    """
    vidcap = cv2.VideoCapture(file_name)
    success,image = vidcap.read()
    cv2.imwrite("frame_"+  file_name[:-3] + ".jpg", image)   # save frame as JPEG file 
    return "frame_"+  file_name[:-3] + ".jpg"

def build_data(files_to_read, num_min):
    """
    This function construct a row of the data for a picture of the first frame of a vedio.

    Parameters
    ----------
    files_to_read : str
        The picture file name.

    Returns
    -------
    str
        A string telling the state of the saving process.

    """
    number_of_hashtags = 0 
    post_dict = get_json(files_to_read[0])
    profile = post_dict['node']['owner']['username']
    likes = post_dict['node']['edge_media_preview_like']['count']
    num_of_comments = post_dict['node']['edge_media_to_comment']['count']
    followers = post_dict['node']['owner']['edge_followed_by']['count']
    following = post_dict['node']['owner']['edge_follow']['count']
    if_text = len(post_dict['node']['edge_media_to_caption']['edges'])
    post_text_len = 0
    if if_text > 0 :
        post_text = post_dict['node']['edge_media_to_caption']['edges'][0]['node']['text']
        post_text_len = len(post_dict['node']['edge_media_to_caption']['edges'][0]['node']['text'])
        has_desc = 1
        number_of_hashtags = cal_hashtags(post_text)
    else:
        number_of_hashtags = 0
        has_desc = 0
        post_text = 0
    # number_of_people_tagged = len(post_dict['node']['edge_media_to_tagged_user']['edges'])
    # number_of_hashtags = cal_hashtags(post_text)
    is_vid = 0
    if files_to_read[1][-3:] == 'mp4':
        pic = get_first_frame(files_to_read[1])
        is_vid = 1
    else:
        pic = files_to_read[1]
        is_vid = 0
        
    dominant_color = get_dominant_color(pic)
    int_dominant_color = [int(i) for i in dominant_color[0]]
    date = get_day_time(files_to_read[0])
    diff = (datetime.datetime.now() - date).days
    num_minutes = num_min
    pred = get_objects(pic)
    has_person = 0
    for ob in pred:
        if 'person' in ob:
            has_person = 1

    result = profile + ',' + str(followers) + ',' + str(following) + ',' + str(num_minutes) + ',' + str(has_desc) + ',' + str(post_text_len) +  ',' + str(number_of_hashtags) + ',' + str(num_of_comments)  + ',' + str(date.year) + ',' + str(date.month) + ',' + str(date.day) + ',' + str(date.hour) + ',' + str(diff) + ',' + str(int_dominant_color[0]) + ',' + str(int_dominant_color[1]) + ',' + str(int_dominant_color[2]) + ',' + str(is_vid) + ',' + str(has_person) + ',' + str(len(pred)) + ','  + str(likes)
    return result


def check_files(this_directory):
    """
    This function checks all files in a directory if they are pictures it includes them in the counting, otherwise, they are ignored.

    Parameters
    ----------
    this_directory : str
        The directory path.

    Returns
    -------
    list
        A list of all available pictures and vedios.

    """
    files = get_immediate_files(this_directory)
    print('got files')
    files_to_read = []
    for file in files:
        if file[-3:] == 'jpg' or file[-3:] == 'mp4':
            for zips in files:
                extension = zips[-3:]
                name = zips[0:-8]
                if zips[-3:] == '.xz' and zips[0:-8] == file[0:-4]:
                    files_to_read.append([zips,file])
                    break

    print('done files' + str(len(files_to_read)))
    return files_to_read

def write_to_file(file_path, lines):
    """
    This function writes a list to a file in the file_path.

    Parameters
    ----------
    file_path : str
        The file path.
    lines : list
        The lines we want to write to the file.

    Returns
    -------

    """
    f = io.open(file_path, 'a', encoding="utf-8")
    for n, line in enumerate(lines):
        if line.startswith(" "):
            lines[n] = "" + line.rstrip()
        else:
            lines[n] = line.rstrip()
        f.write(u''.join(line+'\n'))
    f.close()

def get_minutes(usernames):
    """
    This function calls the itunes API and countes the minutes of the albums for an artist, then divides it by the number of his activity years to get performance.

    Parameters
    ----------
    usernames : str
        The artist username.

    Returns
    -------
    float
        The artist's performance.

    """
    sum = 0
    latest = 0
    earliest = 2019
    for username in usernames:
        a = 'https://itunes.apple.com/search?term=' + username
        b = requests.get(a).json()        
        for track in b['results']:
            release_date = track['releaseDate']
            year = int(release_date[0:4])
            if year < earliest:
                earliest = year
            if year > latest:
                latest = year
            sum = sum + int(track['trackTimeMillis'])
    if latest - earliest > 0:
        duration = latest - earliest
    else:
        duration = 1
    minutes = sum//60000
    performance = minutes//duration
    return performance

def get_objects(file_name):
    """
    This function gets a list of objects detected in a picture.

    Parameters
    ----------
    this_directory : str
        The directory path.

    Returns
    -------
    list
        A list of all available pictures and vedios.

    """
    # instruction = 'python yolo_opencv.py --image ' + file_name + ' --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt'
    # os.system(instruction)
    predictions = yolo_opencv.get_predictions(file_name,'yolov3.cfg', 'yolov3.weights', 'yolov3.txt')
    return predictions

def create_file(files_to_read, file_to_write, user_names=None):
    """
    Main method.

    Parameters
    ----------
    files_to_read : str
        The directory path of files to read.
    file_to_write : str
        The directory path of file to write results on.
    user_names : list
        A list of different usernames of the artist on itunes.

    Returns
    -------

    """
    lines = []
    count = 0
    line=['username,followers, following,  num of minutes,has description, number of words, number of hashtags, number of comments, year, month, day, hour,post age, Red, Green, Blue,is_video,has_person, num_objects, likes']
    if user_names is None:
        num_minutes = 0
    else:
        num_minutes = get_minutes(user_names)
    write_to_file(file_to_write, line)
    for files in files_to_read:
        line= [build_data(files,num_minutes)]
        write_to_file(file_to_write, line)
        print('done ' + str(count) + 'out of :' + str(len(files_to_read)))
        count = count + 1

    
# os.system('python yolo_opencv.py --image 2013-07-22_01-57-08_UTC.jpg --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt')
create_file(check_files(''), '', ['bumble beezy'])
# print(get_objects('2013-07-22_01-57-08_UTC.jpg'))
# print(get_minutes(['баста', 'basta']))