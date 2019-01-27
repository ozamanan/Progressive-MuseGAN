"""Main function
"""
import numpy as np
import tensorflow as tf
import errno
import os
import SharedArray as sa
from musegan.bmusegan.models import GAN, RefineGAN
from config import EXP_CONFIG, DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, TF_CONFIG

def mkdir_p(path):
	try:
		os.makedirs(path)
	except OSError as exc:  # Python >2.5
		if exc.errno == errno.EEXIST and os.path.isdir(path):
			pass
		else:
			raise

def load_data():
	"""Load and return the training data."""
	print('[*] Loading data...')
	if DATA_CONFIG['training_data_location'] == 'sa':
		x_train = sa.attach(DATA_CONFIG['training_data'])
	elif DATA_CONFIG['training_data_location'] == 'hd':
		x_train = np.load(DATA_CONFIG['training_data'])
	x_train = x_train.reshape(
		-1, MODEL_CONFIG['num_bar'], MODEL_CONFIG['num_timestep'],
		MODEL_CONFIG['num_pitch'], MODEL_CONFIG['num_track']
	)
	print('Training set size:', len(x_train))
	return x_train

def pretrain():
	"""Create and pretrain a two-stage model"""
	x_train = load_data()
	r_g = [1,2,3,4,5,6,7,8,9,10,11]
	with tf.Session(config=TF_CONFIG) as sess:
		for i in range(len(r_g)):
			pggan_checkpoint_dir_write = "./output/{}/{}/".format(EXP_CONFIG['exp_name'], r_g[i])
			# sample_path = "./output/{}/{}/sample_{}_{}".format(FLAGS.OPER_NAME, FLAGS.OPER_FLAG, fl[i], t)
			mkdir_p(pggan_checkpoint_dir_write)
			#mkdir_p(sample_path)
			pggan_checkpoint_dir_read = "./output/{}/{}/".format(EXP_CONFIG['exp_name'], r_g[i])
			gan = GAN(sess, MODEL_CONFIG, resolution=r_g[i], model_path=pggan_checkpoint_dir_write)
			gan.init_all()
			if r_g[i] > 0:
				gan.load_latest(pggan_checkpoint_dir_read)
			gan.train(x_train, TRAIN_CONFIG)

def train_end2end():
	"""Create and train an end-to-end model"""
	x_train = load_data()
	with tf.Session(config=TF_CONFIG) as sess:
		end2end_gan = End2EndGAN(sess, MODEL_CONFIG)
		end2end_gan.init_all()
		if EXP_CONFIG['pretrained_dir'] is not None:
			end2end_gan.load_latest(EXP_CONFIG['pretrained_dir'])
		end2end_gan.train(x_train, TRAIN_CONFIG)

if __name__ == '__main__':
	print("Start experiment: {}".format(EXP_CONFIG['exp_name']))
	pretrain()
