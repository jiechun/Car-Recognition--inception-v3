import tensorflow as tf
import cv2
import numpy as np
import time
import sys


# change this as you see fit


image_path_in = sys.argv[1]

# image_path_in2 = sys.argv[2]
# image_path_in = image_path_in1+" "+image_path_in2
list = image_path_in.split('/')

file_name = "gai"+list[-1]

print(file_name)
# image_path_in = 'E:/Program Files/myFiles/video/new.mp4'
#image_data = cv2.imread('166.jpg')
# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
               in tf.gfile.GFile("E:\\anaconda\\envs\\tensorflow\\models-master\\tutorials\\image\\imagenet\\tf_files\\retrained_labels.txt")]



# print ('E:/ProgramFiles/myFiles/video/'+list[-1])

# Release everything if job is finished



# Unpersists graph from file
with tf.gfile.FastGFile("E:\\anaconda\\envs\\tensorflow\\models-master\\tutorials\\image\\imagenet\\tf_files\\retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    camera = cv2.VideoCapture(image_path_in)
    es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    kernel = np.ones((5, 5), np.uint8)
    errt, background = camera.read()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('E:\\ProgramFiles\\myFiles\\video\\' + file_name, fourcc, 30, (len(background[0]), len(background)))


    while (True):
        errt, frame = camera.read()
        if errt == True:
            cv2.rectangle(frame, (220, 150), (460, 350), (255, 0, 0), 2)
            background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
            background = cv2.GaussianBlur(background, (21, 21), 0)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
            diff = cv2.absdiff(background, gray_frame)
            diff = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)[1]
            diff = cv2.dilate(diff, es, iterations=2)
            image, cnts, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            background = frame
            for c in cnts:
                if cv2.contourArea(c) < 1500:
                    continue
                (x, y, w, h) = cv2.boundingRect(c)
                img = frame[y:y + h, x:x + w]
                m = x + w
                n = y + h
                cv2.imwrite("E:\\anaconda\\envs\\tensorflow\\models-master\\tutorials\\image\\imagenet\\tf_files\\flower_photos\\che\\1.jpg", img)
                if (x >= 220 and y >= 150 and m <= 460 and n <= 350):
                    image_path = "E:\\anaconda\\envs\\tensorflow\\models-master\\tutorials\\image\\imagenet\\tf_files\\flower_photos\\che\\1.jpg"
                # Read in the image_data
                    image_data = tf.gfile.FastGFile(image_path, 'rb').read()
                    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

                    predictions = sess.run(softmax_tensor, \
                                        {'DecodeJpeg/contents:0': image_data})

                    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
                    node_id = top_k[0]
                    human_string = label_lines[node_id]
                    if(human_string == "car"):
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
            out.write(frame)
            cv2.imshow('im',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    out.release()
    print ('E:/ProgramFiles/myFiles/video/'+file_name)