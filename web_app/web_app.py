from __future__ import division
import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Convolution2D, Activation, Dense, Flatten, LeakyReLU
from sklearn.metrics import pairwise
from flask import Flask, request, redirect, url_for,render_template
from werkzeug.utils import secure_filename
# from scipy.misc import imsave
from gevent.pywsgi import WSGIServer
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.backend import set_session
from flask import send_from_directory

UPLOAD_FOLDER = 'uploads'
app = Flask(__name__)

sys.setrecursionlimit(40000)

config_output_filename = "config.pickle"

with open(config_output_filename, 'rb') as f_in:
    C = pickle.load(f_in)

if C.network == 'resnet50':
    import keras_frcnn.resnet as nn
elif C.network == 'vgg':
    import keras_frcnn.vgg as nn

# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

# img_path = options.test_path
# img_path = img_path.replace('\\', '/')


def format_img_size(img, C):
    """ formats the image size based on config """
    img_min_side = float(C.im_size)
    (height, width, _) = img.shape

    if width <= height:
        ratio = img_min_side / width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side / height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio


def format_img_channels(img, C):
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def format_img(img, C):
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio


# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return (real_x1, real_y1, real_x2, real_y2)


if not os.path.exists('./results'):
    os.mkdir('./results')

class_mapping = C.class_mapping

if 'bg' not in class_mapping:
    class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
C.num_rois = int(32)

if C.network == 'resnet50':
    num_features = 1024
elif C.network == 'vgg':
    num_features = 512

if K.common.image_dim_ordering() == 'th':
    input_shape_img = (3, None, None)
    input_shape_features = (num_features, None, None)
else:
    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, num_features)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

global sess1 
sess1 = tf.Session()
set_session(sess1)
global model_rpn

model_rpn = Model(img_input, rpn_layers)
print('Loading weights from {}'.format('./model/model_frcnn_epoch_100.hdf5'))
model_rpn.load_weights('./model/model_frcnn_epoch_100.hdf5', by_name=True)
model_rpn.compile(optimizer='sgd', loss='mse')
global graph1
graph1 = tf.get_default_graph()

global sess2
sess2 = tf.Session()
set_session(sess2)
global model_classifier_only
model_classifier_only = Model([feature_map_input, roi_input], classifier)
model_classifier = Model([feature_map_input, roi_input], classifier)
model_classifier.load_weights('./model/model_frcnn_epoch_100.hdf5', by_name=True)
model_classifier.compile(optimizer='sgd', loss='mse')
global graph2
graph2 = tf.get_default_graph()


global sess3
sess3 = tf.Session()
set_session(sess3)
global classification_model
classification_model = Sequential()
classification_model.add(Convolution2D(##, kernel_size=(#, #), padding='Same', input_shape=(128, 128, 3)))
classification_model.add(LeakyReLU(alpha=#))
classification_model.add(Convolution2D(##, kernel_size=(#, #), padding='Same'))
classification_model.add(LeakyReLU(alpha=##))
classification_model.add(Flatten())
classification_model.add(Dense(##, activation='relu'))
classification_model.add(Dense(##, activation='relu'))
classification_model.add(Dense(#, activation='sigmoid'))
classification_model.load_weights('.model/classification_model.h5')
global graph3
graph3 = tf.get_default_graph()


global sess4
sess4 = tf.Session()
set_session(sess4)
global loaded_model
loaded_model = load_model('./model/model_300_bs-16.h5')
global graph4
graph4 = tf.get_default_graph()

all_imgs = []

classes = {}

bbox_threshold = 0.8

visualise = True

@app.route('/', methods = ['GET','POST'])
def  main_page():
    print('index')
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join('uploads', 'input_image.tif'))
        return redirect(url_for('prediction'))
    return render_template('index.html')

@app.route('/prediction/')
def prediction():

    st = time.time()
    # filepath = os.path.join(img_path, img_name)
    filepath = os.path.join('uploads','input_image.tif')
    image_input = cv2.imread(filepath)
    img = image_input.copy()

    X, ratio = format_img(img, C)
    df_dat = pd.DataFrame(columns=['image_names', 'label', 'xmin', 'ymin', 'xmax', 'ymax'])

    if K.common.image_dim_ordering() == 'tf':
        X = np.transpose(X, (0, 2, 3, 1))

    # get the feature maps and output from the RPN
    with graph1.as_default():
        set_session(sess1)
        [Y1, Y2, F] = model_rpn.predict(X)

    R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.common.image_dim_ordering(), overlap_thresh=0.7)

    # convert from (x1,y1,x2,y2) to (x,y,w,h)
    R[:, 2] -= R[:, 0]
    R[:, 3] -= R[:, 1]

    # apply the spatial pyramid pooling to the proposed regions
    bboxes = {}
    probs = {}

    for jk in range(R.shape[0] // C.num_rois + 1):
        ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
        if ROIs.shape[1] == 0:
            break

        if jk == R.shape[0] // C.num_rois:
            # pad R
            curr_shape = ROIs.shape
            target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
            ROIs_padded[:, :curr_shape[1], :] = ROIs
            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
            ROIs = ROIs_padded

        with graph2.as_default():
            set_session(sess2)
            [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

        for ii in range(P_cls.shape[1]):

            if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                continue

            cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

            if cls_name not in bboxes:
                bboxes[cls_name] = []
                probs[cls_name] = []

            (x, y, w, h) = ROIs[0, ii, :]

            cls_num = np.argmax(P_cls[0, ii, :])
            try:
                (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                tx /= C.classifier_regr_std[0]
                ty /= C.classifier_regr_std[1]
                tw /= C.classifier_regr_std[2]
                th /= C.classifier_regr_std[3]
                x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
            except:
                pass
            bboxes[cls_name].append(
                [C.rpn_stride * x, C.rpn_stride * y, C.rpn_stride * (x + w), C.rpn_stride * (y + h)])
            probs[cls_name].append(np.max(P_cls[0, ii, :]))

    all_dets = []

    for key in bboxes:
        bbox = np.array(bboxes[key])

        new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
        for jk in range(new_boxes.shape[0]):
            (x1, y1, x2, y2) = new_boxes[jk, :]

            (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

            # cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)
            if key == 'valid':
                cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2), (0, 255, 0), 2)
            elif key == 'invalid':
                cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2), (255, 0, 0), 2)

            textLabel = '{}: {}'.format(key, jk)
            all_dets.append((key, jk))

            # print(filepath,textLabel,(real_x1, real_y1, real_x2, real_y2))

            if (key == 'valid'):
                df_dat = df_dat.append(
                    {'filepath': filepath, 'image_names': filepath.split("/")[-1], 'comet_number': jk, 'label': key,
                     'xmin': real_x1, 'ymin': real_y1, 'xmax': real_x2, 'ymax': real_y2}, ignore_index=True)

            (retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
            textOrg = (real_x1, real_y1 - 0)

            # print('retval=',retval)
            # print('baseLine=',baseLine)
            # print('textOrg=',textOrg)
            # print('left',(textOrg[0] - 5, textOrg[1]+baseLine - 5))
            # print('right',(textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5))

            ## drwaing rectangle for text box
            cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                          (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (0, 0, 0), 2)
            cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                          (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (255, 255, 255), -1)
            cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)
            cv2.putText(image_input, 'Input', (0, image_input.shape[0]), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 5)
            cv2.putText(img, 'Prediction', (0, img.shape[0]), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 5)
    out_img = cv2.hconcat([image_input,img])
    # if 'output_image.png' in os.listdir('uploads'):
    #     os.remove('uploads/output_image.png')
    #     os.remove('uploads/predicted_image.png')
    cv2.imwrite('uploads/output_image.png', out_img)
    cv2.imwrite('uploads/predicted_image.png', img)
    # print(df_dat)
    #     bar.next()
    # bar.finish()

    # classification model(damaged or undamaged
    print('-----------------module 2 -------------------------------')


    def histogram_equalize(imgs):
        images = imgs
        for i, img in enumerate(images):
            b, g, r = cv2.split(img)
            red = cv2.equalizeHist(r)
            green = cv2.equalizeHist(g)
            blue = cv2.equalizeHist(b)
            img = cv2.merge((red, green, blue))
            images[i, :, :, :] = img
        return images


    def cropped_image(df):
        x_test = []
        for i in range(df.shape[0]):
            image = cv2.cvtColor(cv2.imread(df.filepath[i]), cv2.COLOR_BGR2RGB)
            cropped = image[df['ymin'][i]:df['ymax'][i], df['xmin'][i]:df['xmax'][i]]
            x_test.append(cropped)
        return np.array(x_test)


    def resize(images, size):
        imgs = []
        for i, img in enumerate(images):
            imgs.append(cv2.resize(img, (size, size)))
        return np.reshape(imgs, (len(imgs), size, size, 3))


    x_test = cropped_image(df_dat)
    x_test = resize(x_test, 128)
    x_test = histogram_equalize(x_test)

    with graph3.as_default():
        set_session(sess3)
        predict = classification_model.predict(x_test, batch_size=64, verbose=1)

    predict = 1 * (predict > 0.5)

    df_dat['damage_analysis'] = predict
    df_dat['damage_analysis'].replace({0: 'undamaged', 1: 'damaged'}, inplace=True)

    print('-----------------module 3 -------------------------------')

    df_damaged = df_dat[df_dat['damage_analysis'] == 'damaged'].reset_index()
    x_test = cropped_image(df_damaged)
    print(x_test.shape)


    def load_imgs(images):
        imgs = []
        shape = []
        for im in range(0, len(images)):
            data = images[im]
            data = data / 255
            height = data.shape[0]
            width = data.shape[1]
            imgs.append(cv2.resize(data, (150, 150)) / 255)
            shape.append(data.shape)
        imgs = np.reshape(np.array(imgs), (len(imgs), 150, 150, 3))
        return imgs, shape


    images, shape = load_imgs(x_test)

    with graph4.as_default():
        set_session(sess4)
        predictions = loaded_model.predict(images, verbose=1)

    for i in range(len(images)):
        img = cv2.resize(images[i], (shape[i][1], shape[i][0]))
        height = shape[i][0]
        width = shape[i][1]
        predictions[i][0:43:2] = (predictions[i][0:43:2]) * (width / 150)
        predictions[i][1:43:2] = (predictions[i][1:43:2]) * (height / 150)

    predictions = pd.DataFrame(predictions)
    keypoints = predictions.astype(np.int)
    print(len(keypoints))

    print('------ done with keypoints prediction ---------------------')


    def otherpoints(i):
        x = keypoints.iloc[i, 0:40:2].values
        y = keypoints.iloc[i, 1:40:2].values
        df = pd.DataFrame()
        df['x'] = x
        df['y'] = y
        return df.values.tolist()


    def dnapercent_intens(image, radius, center, vertices):
        head = image.copy()
        comet = image.copy()
        cv2.circle(head, (int(center[0]), int(center[1])), int(radius), color=(255, 255, 255), thickness=-1)
        head[head != 255] = 0
        head_dna = sum(image[np.where(head == 255)])
        head_area = len(image[np.where(head == 255)])

        cv2.fillPoly(comet, [vertices], color=(255, 255, 255))
        comet[comet != 255] = 0
        total_dna = sum(image[np.where(comet == 255)])
        tail = comet - head
        tail_dna = sum(image[np.where(tail == 255)])
        tail_area = len(image[np.where(tail == 255)])

        return head_dna, total_dna, tail_dna, head, tail, head_area, tail_area

    measurements = []

    for i in range(len(keypoints)):
        image = x_test[i]
        im_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        mask = image.copy()
        distance = pairwise.euclidean_distances([keypoints.iloc[i, -2:]], Y=otherpoints(i))[0]
        tail_length = distance.max() - distance.min()
        head_length = 2 * distance.min()
        vertices = np.array(otherpoints(i), dtype=np.int32)
        vertices = vertices.reshape((-1, 1, 2))
        HDNAP, DNAcontent, TDNAP, headregion, Tail_region, ha, ta = dnapercent_intens(image, distance.min(),
                                                                                      keypoints.iloc[i, -2:].values,
                                                                                      vertices)
        head_int, tot_int, tail_int, h, tail, hea, taa = dnapercent_intens(im_gray, distance.min(),
                                                                           keypoints.iloc[i, -2:].values, vertices)
        Tail_region = cv2.cvtColor(Tail_region, cv2.COLOR_RGB2GRAY)
        images, contours, hierarchy = cv2.findContours(Tail_region, 1, 2)
        cnts = []
        for cnt in contours:
            cnts.append(len(cnt))
        cv2.drawContours(mask, contours, -1, color=(0, 255, 0), thickness=5)
        cnt = contours[np.argmax(cnts)]
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.circle(mask, (cx, cy), 5, color=(0, 0, 255), thickness=-1)
        tail_distance = pairwise.euclidean_distances([keypoints.iloc[i, -2:]], Y=[[cx, cy]])[0]
        Olive_tail_moment = tail_distance[0] * (TDNAP / DNAcontent)
        try:
            head_intensity = (head_int / hea)
        except ZeroDivisionError:
            head_intensity = 0
        try:
            tail_intensity = (tail_int / taa)
        except ZeroDivisionError:
            tail_intensity = 0
        try:
            head_DNA_percentage = (HDNAP / DNAcontent)
        except ZeroDivisionError:
            head_DNA_percentage = 0
        try:
            Tail_DNA_percentage = (TDNAP / DNAcontent)
        except ZeroDivisionError:
            Tail_DNA_percentage = 0

        pred = {'comet_number': i,
                'head_length':round(head_length,3),
                'tail_length':round(tail_length,3),
                'head_intensity':round(head_intensity,3),
                'tail_intensity':round(tail_intensity,3),
                'head_DNA_percentage': round(head_DNA_percentage,3),
                'Tail_DNA_percentage':round(Tail_DNA_percentage,3),
                'Olive_tail_moment':round(Olive_tail_moment,3)}
        measurements.append(pred)

        # measurements.append([i] + 
        #                     [head_length] +
        #                     [tail_length] +
        #                     [head_intensity] +
        #                     [tail_intensity] +
        #                     [head_DNA_percentage * 100] +
        #                     [Tail_DNA_percentage * 100] + [Olive_tail_moment])
    print(measurements)

    # measurements = pd.DataFrame(measurements, columns=['comet_number','head_length', 'tail_length', 'head_intensity', 'tail_intensity',
    #                                                    'head_DNA_percentage', 'Tail_DNA_percentage', 'Olive_Tail_moment'])
    # final_df = df_damaged[['comet_number']]
    # final_df = pd.concat([final_df, measurements], axis=1)
    # final_df = final_df.to_dict('list')

    return render_template('predictions.html',image_file_name ='output_image.png',predicted_image_name='predicted_image.png',predictions = measurements)

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

app.run(host='0.0.0.0',port = 80)
