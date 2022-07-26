import cv2 as cv
import numpy as np
import tensorflow as tf
eps = 0.000001
def recall(y_true, y_pred):
    y_pred = np.round(y_pred)
    true_positives = np.sum(np.round(y_true * y_pred))
    possible_positives = np.sum(np.round(y_true))
    recall_score = true_positives / (possible_positives + eps)
    print("Recall: ", recall_score)
    return recall_score
def precision(y_true, y_pred):
    y_pred = np.round(y_pred)
    true_positives = np.sum(np.round(y_true * y_pred))
    predicted_positives = np.sum(np.round(y_pred))
    precision_score = true_positives / (predicted_positives + eps)
    print("Precision: ", precision_score)
    return precision_score
def f1(y_true, y_pred):
    precision_score = precision(y_true, y_pred)
    recall_score = recall(y_true, y_pred)
    return 2 * ((precision_score * recall_score) / (precision_score + recall_score + eps))
def iou(y_true, y_pred):
    y_pred = np.round(y_pred)
    intersection = y_true * y_pred
    not_true = 1 - y_true
    union = (y_true + (not_true * y_pred))
    return np.clip((np.sum(intersection, axis=-1) + eps) / (np.sum(union, axis=-1) + eps), 0, 1)
def preprocess_skinny(orig_img):
    number_multiple = 32
    nm2 = 32
    tensor_shape = tf.cast(tf.shape(orig_img), tf.float32)
    coefficient = 512**2 / (tensor_shape[0] * tensor_shape[1])
    coefficient = tf.math.sqrt(coefficient)
    orig_img = tf.cond(coefficient >= 1.0, lambda: orig_img,
                        lambda: tf.image.resize(orig_img, [tf.cast(tensor_shape[0] * coefficient, tf.uint16),
                                                    tf.cast(tensor_shape[1] * coefficient, tf.uint16)]))
    orig_img = tf.pad(orig_img, [[0, number_multiple - orig_img.shape[0] % number_multiple], [0,  nm2 - orig_img.shape[1] % nm2], [0, 0]])
    orig_img = np.array(orig_img)
    # print(orig_img.shape)
    orig_img = cv.resize(orig_img, (704,416))
    return orig_img
def calculate_scores():
    file = open('/home/emprise/cv-project/dataset/test.txt', 'r')
    Lines = file.readlines()
    s_labels = []
    s_preds = []
    ws_labels = []
    ws_preds = []
    ds_labels = []
    ds_preds = []
    bd_labels = []
    bd_preds = []
    num = 0
    for line in Lines:
        # print(line.strip())
        # skin_labels_file_name = "skin_labels/" + line.strip() + ".png"
        skin_labels_file_name = "/home/emprise/cv-project/dataset/org/labels/" + line.strip() + ".png"
        wet_labels_file_name = "/home/emprise/cv-project/dataset_wd/org/labels/" + line.strip() + ".png"
        skin_preds_file_name = "/home/emprise/cv-project/results/output_skinny_skin/" + str(num) + ".png"
        wet_preds_file_name = "/home/emprise/cv-project/results/output_deeplab_wet/" + str(num) + ".png"
        num += 1
        # print(input_file_name, output_file_name)
        # skin_labels.append(cv.imread(skin_labels_file_name,cv.IMREAD_GRAYSCALE))
        skin_label = (cv.resize(cv.cvtColor(preprocess_skinny(cv.imread(skin_labels_file_name)), cv.COLOR_BGR2GRAY),(512,512)))/255
        wet_label = (cv.resize(cv.cvtColor(preprocess_skinny(cv.imread(wet_labels_file_name)), cv.COLOR_BGR2GRAY),(512,512)))/255
        dry_label = np.clip(skin_label - wet_label, 0 , 1)
        bd_label = np.clip(1 - skin_label, 0 , 1)
        skin_pred = (cv.resize(cv.imread(skin_preds_file_name,cv.IMREAD_GRAYSCALE),(512,512)))/255
        wet_pred = np.clip(skin_pred * ( (cv.resize(cv.cvtColor(preprocess_skinny(cv.imread(wet_preds_file_name)), cv.COLOR_BGR2GRAY),(512,512)))/255), 0, 1)
        dry_pred = np.clip(skin_pred - wet_pred, 0 , 1)
        bd_pred = np.clip(1 - skin_pred, 0 , 1)
        # cv.imshow('ImageWindow', label_img)
        # cv.waitKey(0)
        s_labels.append(skin_label)
        s_preds.append(skin_pred)
        ws_labels.append(wet_label)
        ws_preds.append(wet_pred)
        ds_labels.append(dry_label)
        ds_preds.append(dry_pred)
        bd_labels.append(bd_label)
        bd_preds.append(bd_pred)
        # cv.imshow('ImageWindow', img)
        # cv.waitKey(0)   
    s_labels = np.asarray(s_labels)
    s_preds = np.asarray(s_preds)
    ws_labels = np.asarray(ws_labels)
    ws_preds = np.asarray(ws_preds)
    ds_labels = np.asarray(ds_labels)
    ds_preds = np.asarray(ds_preds)
    bd_labels = np.asarray(bd_labels)
    bd_preds = np.asarray(bd_preds)

    print(np.sum(ws_labels))

    print("Skin Scores: ")
    print("F1 Score: ", f1(s_labels, s_preds))
    print("IoU Score: ", np.mean(iou(s_labels, s_preds)))
    print("Wet Skin Scores: ")
    print("F1 Score: ", f1(ws_labels, ws_preds))
    print("IoU Score: ", np.mean(iou(ws_labels, ws_preds)))
    print("Dry Skin Scores: ")
    print("F1 score: ", f1(ds_labels, ds_preds))
    print("IoU Score: ", np.mean(iou(ds_labels, ds_preds)))
    print("Background Scores: ")
    print("F1 Score: ", f1(bd_labels, bd_preds))
    print("IoU Score: ", np.mean(iou(bd_labels, bd_preds)))

calculate_scores()
