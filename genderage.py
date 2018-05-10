from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from timeit import default_timer as timer
from datetime import datetime
import math
import time
from data import inputs
import numpy as np
import tensorflow as tf
from model import select_model, get_checkpoint
from utils import *
import os,cv2
import threading

RESIZE_FINAL = 227
GENDER_LIST =['M','F']
AGE_LIST = ['(0, 4)','(4, 10)','(8, 18)','(15, 25)','(23, 35)','(35, 48)','(48, 58)','(60, 100)']
MAX_BATCH_SZ = 128

tf.app.flags.DEFINE_string('device_id', '/gpu:0',
                           'What processing unit to execute inference on')

tf.app.flags.DEFINE_string('target', '',
                           'CSV file containing the filename processed along with best guess and score')

tf.app.flags.DEFINE_string('checkpoint', 'checkpoint',
                          'Checkpoint basename')

tf.app.flags.DEFINE_string('model_type', 'inception',
                           'Type of convnet')

tf.app.flags.DEFINE_string('requested_step', '', 'Within the model directory, a requested step to restore e.g., 9000')

class Track:
	def __init__(self, box, path):
		self.gender = 'not decide'
		self.age = 'not decide'
		self.box = box
		self.path = path
		self.miss = 0
		#self.isNew = True

	def isNeedGender(self):
		if self.gender == 'not decide':
			return True
		else:
			return False

	def isNeedAge(self):
		if self.age == 'not decide':
			return True
		else:
			return False


FLAGS = tf.app.flags.FLAGS
CASCADE="Face_cascade.xml"
FACE_CASCADE=cv2.CascadeClassifier(CASCADE)
coder = ImageCoder()

trackers = []
isNotStop = True
def iou(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2]+boxA[0], boxB[2]+boxB[0])
	yB = min(boxA[3]+boxA[1], boxB[3]+boxB[1])
 
	# return the intersection over union value
	if xB>xA and yB>yA: 
	    interArea = (xB - xA + 1) * (yB - yA + 1)
	    boxAArea = (boxA[2]+ 1) * (boxA[3] + 1)
	    boxBArea = (boxB[2]+ 1) * (boxB[3] + 1)
	    iou = interArea / float(boxAArea+boxBArea-interArea)
	    return iou
	else:
	    return 0

def detect_faces():
	global trackers
	global isNotStop
	font = cv2.FONT_HERSHEY_SIMPLEX
	cap = cv2.VideoCapture(0)
	while isNotStop:
		ret, image= cap.read()
		image_grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		boxes = FACE_CASCADE.detectMultiScale(image_grey,scaleFactor=1.16,minNeighbors=5,minSize=(25,25),flags=0)
		if boxes !=():
			boxes = boxes.tolist()
		for track in trackers:
			maxiou = 0
			for box in boxes:
				iou1 = iou(track.box, box)
				if iou1>maxiou:
					maxiou = iou1
					best_match = box

			if maxiou>0.5:
				track.box = best_match
				x,y,w,h = best_match
				path=image[max(y-int(h/8),0):y+int(1.25*h),max(x-int(w/8),0):x+int(1.25*w)]
				track.path = path
				boxes.remove(best_match)
			else:
				track.miss +=1
				if(track.miss>5):
					trackers.remove(track)

		for box in boxes:
			x,y,w,h = box
			path=image[max(y-int(h/8),0):y+int(1.25*h),max(x-int(w/8),0):x+int(1.25*w)]
			track = Track(box, path)
			trackers.append(track)

		for index in range(len(trackers)):
			cv2.rectangle(image,(trackers[index].box[0],trackers[index].box[1]),(trackers[index].box[2]+trackers[index].box[0],trackers[index].box[3]+trackers[index].box[1]),(255, 255,0),3)
			try :
			    cv2.putText(image,str(trackers[index].age),(trackers[index].box[0],trackers[index].box[1]), font, 0.5,(255,0,255),1,cv2.LINE_AA)
			   # print(str(trackers[index].age))
			except :
			    pass
			try :
			    cv2.putText(image,str(trackers[index].gender),(trackers[index].box[0],trackers[index].box[1]+20), font, 0.5,(255,0,255),1,cv2.LINE_AA)
			    #print(str(trackers[index].gender))
			except :
			    pass
		cv2.imshow('result', image)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			isNotStop = False
	

def age(argv=None):  # pylint: disable=unused-argument
    global isRunRecog
    global trackers
    global isNotStop
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:

        label_list = AGE_LIST # if FLAGS.class_type == 'age' else GENDER_LIST
        nlabels = len(label_list)

        print('Executing on %s' % FLAGS.device_id)
        model_fn = select_model(FLAGS.model_type)

        with tf.device(FLAGS.device_id):
            
            images = tf.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3])
            logits = model_fn(nlabels, images, 1, False)
            init = tf.global_variables_initializer()
            
            requested_step = FLAGS.requested_step if FLAGS.requested_step else None
        
            checkpoint_path = '%s' % ('22801/')

            model_checkpoint_path, global_step = get_checkpoint(checkpoint_path, requested_step, FLAGS.checkpoint)
            
            saver = tf.train.Saver()
            saver.restore(sess, model_checkpoint_path)
                        
            softmax_output = tf.nn.softmax(logits)

            output = None
	    
	    #while not isRunRecog :
		#pass
	    while isNotStop:
		try :
		    for index, track in enumerate(trackers):
			sub_img = track.path.copy()
			image_data = cv2.imencode('.jpg',sub_img)[1].tostring()
			image1 = coder.decode_jpeg(image_data)
			crops = []
			h = image1.shape[0]
			w = image1.shape[1]
			hl = h - RESIZE_FINAL
			wl = w - RESIZE_FINAL

			crop = tf.image.resize_images(sub_img, (RESIZE_FINAL, RESIZE_FINAL))
			crops.append(standardize_image(crop))
			crops.append(tf.image.flip_left_right(crop))

			corners = [ (0, 0), (0, wl), (hl, 0), (hl, wl), (int(hl/2), int(wl/2))]
			for corner in corners:
			    ch, cw = corner
			    cropped = tf.image.crop_to_bounding_box(image1, ch, cw, RESIZE_FINAL, RESIZE_FINAL)
			    crops.append(standardize_image(cropped))
			    flipped = tf.image.flip_left_right(cropped)
			    crops.append(standardize_image(flipped))

			image_batch = tf.stack(crops)

		    	batch_results = sess.run(softmax_output, feed_dict={images:image_batch.eval()})
		    	output = batch_results[0]
		    	batch_sz = batch_results.shape[0]
	    
		    	for i in range(1, batch_sz):
		        	output = output + batch_results[i]
		
		    	output /= batch_sz
		    	best = np.argmax(output)
		    	best_choice = (label_list[best], output[best])
		    	print('Guess @ 1 %s, prob = %.2f' % best_choice)
			track.age = best_choice
		except :
		    images = tf.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3])
		    logits = model_fn(nlabels, images, 1, False)
		    init = tf.global_variables_initializer()
		    
		    requested_step = FLAGS.requested_step if FLAGS.requested_step else None
		
		    checkpoint_path = '%s' % ('22801/')

		    model_checkpoint_path, global_step = get_checkpoint(checkpoint_path, requested_step, FLAGS.checkpoint)
		    
		    saver = tf.train.Saver()
		    saver.restore(sess, model_checkpoint_path)
		                
		    softmax_output = tf.nn.softmax(logits)
		
def gender(argv=None):  # pylint: disable=unused-argument
    global isRunRecog
    global trackers
    global isNotStop
    time.sleep(4)
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:

        label_list = GENDER_LIST
        nlabels = len(label_list)

        print('Executing on %s' % FLAGS.device_id)
        model_fn = select_model(FLAGS.model_type)

        with tf.device(FLAGS.device_id):
            
            images = tf.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3])
            logits = model_fn(nlabels, images, 1, False)
            init = tf.global_variables_initializer()
            
            requested_step = FLAGS.requested_step if FLAGS.requested_step else None
        
            checkpoint_path = '%s' % ('21936/')

            model_checkpoint_path, global_step = get_checkpoint(checkpoint_path, requested_step, FLAGS.checkpoint)
            
            saver = tf.train.Saver()
            saver.restore(sess, model_checkpoint_path)
                        
            softmax_output = tf.nn.softmax(logits)

            output = None
	    
	    ageListLocal =[]
	    #while not isRunRecog :
		#pass
	    while isNotStop:
		try : 
		    for index, track in enumerate(trackers):
			sub_img = track.path.copy()
			image_data = cv2.imencode('.jpg',sub_img)[1].tostring()
			image1 = coder.decode_jpeg(image_data)
			crops = []
			h = image1.shape[0]
			w = image1.shape[1]
			hl = h - RESIZE_FINAL
			wl = w - RESIZE_FINAL

			crop = tf.image.resize_images(sub_img, (RESIZE_FINAL, RESIZE_FINAL))
			crops.append(standardize_image(crop))
			crops.append(tf.image.flip_left_right(crop))

			corners = [ (0, 0), (0, wl), (hl, 0), (hl, wl), (int(hl/2), int(wl/2))]
			for corner in corners:
			    ch, cw = corner
			    cropped = tf.image.crop_to_bounding_box(image1, ch, cw, RESIZE_FINAL, RESIZE_FINAL)
			    crops.append(standardize_image(cropped))
			    flipped = tf.image.flip_left_right(cropped)
			    crops.append(standardize_image(flipped))

			image_batch = tf.stack(crops)

		    	batch_results = sess.run(softmax_output, feed_dict={images:image_batch.eval()})
		    	output = batch_results[0]
		    	batch_sz = batch_results.shape[0]
	    
		    	for i in range(1, batch_sz):
		        	output = output + batch_results[i]
		
		    	output /= batch_sz
		    	best = np.argmax(output)
		    	best_choice = (label_list[best], output[best])
		    	print('Guess @ 1 %s, prob = %.2f' % best_choice)
			track.gender = best_choice
		except : 
		    images = tf.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3])
		    logits = model_fn(nlabels, images, 1, False)
		    init = tf.global_variables_initializer()
		    
		    requested_step = FLAGS.requested_step if FLAGS.requested_step else None
		
		    checkpoint_path = '%s' % ('21936/')

		    model_checkpoint_path, global_step = get_checkpoint(checkpoint_path, requested_step, FLAGS.checkpoint)
		    
		    saver = tf.train.Saver()
		    saver.restore(sess, model_checkpoint_path)
		                
		    softmax_output = tf.nn.softmax(logits)



faces = threading.Thread(target=detect_faces)
faces.start()

agethread = threading.Thread(target=age)
genderthread = threading.Thread(target=gender)
agethread.start()
genderthread.start()
genderthread.join()
agethread.join()
faces.join()



