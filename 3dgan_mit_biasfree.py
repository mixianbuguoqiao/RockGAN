#!/usr/bin/env python
import os

import numpy as np
import visdom

import dataIO as d

from utils import *

list= []
'''
Global Parameters
'''
n_epochs = 10000
batch_size = 1

g_lr = 0.00025
d_lr = 0.00001


beta = 0.5
d_thresh = 0.8
z_size = 200
y_dim = 6
leak_value = 0.2
cube_len = 64
obj_ratio = 0.7
obj = 'chair'

train_sample_directory = './train_sample/'
model_directory = './models/'

is_local = False
config = tf.ConfigProto(
    device_count={'GPU': 0}
)
weights = {}


def generator(z, y, batch_size=batch_size, phase_train=True, reuse=False):
    strides = [1, 2, 2, 2, 1]

    with tf.variable_scope("gen", reuse=reuse):
        z = tf.reshape(z, (batch_size, 1, 1, 1, z_size))
        yb = tf.reshape(y, shape=[batch_size, 1, 1, 1, y_dim])
        z = tf.concat([z, yb], 4)
        # z = tf.reshape(z, (batch_size, 1, 1, 1, z_size)) batch, long, width ,height, in_channels

        g_1 = tf.nn.conv3d_transpose(z, weights['wg1'], (batch_size, 4, 4, 4, 512), strides=[1, 1, 1, 1, 1],
                                     padding="VALID")

        g_1 = tf.contrib.layers.batch_norm(g_1, is_training=phase_train)

        g_1 = tf.nn.relu(g_1)

        g_2 = tf.nn.conv3d_transpose(g_1, weights['wg2'], (batch_size, 8, 8, 8, 256), strides=strides, padding="SAME")
        g_2 = tf.contrib.layers.batch_norm(g_2, is_training=phase_train)
        g_2 = tf.nn.relu(g_2)

        g_3 = tf.nn.conv3d_transpose(g_2, weights['wg3'], (batch_size, 16, 16, 16, 128), strides=strides,
                                     padding="SAME")
        g_3 = tf.contrib.layers.batch_norm(g_3, is_training=phase_train)
        g_3 = tf.nn.relu(g_3)

        g_4 = tf.nn.conv3d_transpose(g_3, weights['wg4'], (batch_size, 32, 32, 32, 64), strides=strides, padding="SAME")
        g_4 = tf.contrib.layers.batch_norm(g_4, is_training=phase_train)
        g_4 = tf.nn.relu(g_4)

        g_5 = tf.nn.conv3d_transpose(g_4, weights['wg5'], (batch_size, 64, 64, 64, 1), strides=strides, padding="SAME")

        g_5 = tf.nn.tanh(g_5)

    print(g_1, 'g1')
    print(g_2, 'g2')
    print(g_3, 'g3')
    print(g_4, 'g4')
    print(g_5, 'g5')

    return g_5


def discriminator(inputs, y, phase_train=True, reuse=False):
    strides = [1, 2, 2, 2, 1]
    with tf.variable_scope("dis", reuse=reuse):
        print("input shape:")
        print(inputs.shape)
        print("mask shape：")
        print(y.shape)
        inputs = tf.concat([inputs, y], 4)
        d_1 = tf.nn.conv3d(inputs, weights['wd1'], strides=strides, padding="SAME")
        d_1 = tf.contrib.layers.batch_norm(d_1, is_training=phase_train)
        d_1 = lrelu(d_1, leak_value)

        d_2 = tf.nn.conv3d(d_1, weights['wd2'], strides=strides, padding="SAME")
        d_2 = tf.contrib.layers.batch_norm(d_2, is_training=phase_train)
        d_2 = lrelu(d_2, leak_value)

        d_3 = tf.nn.conv3d(d_2, weights['wd3'], strides=strides, padding="SAME")
        d_3 = tf.contrib.layers.batch_norm(d_3, is_training=phase_train)
        d_3 = lrelu(d_3, leak_value)

        d_4 = tf.nn.conv3d(d_3, weights['wd4'], strides=strides, padding="SAME")
        d_4 = tf.contrib.layers.batch_norm(d_4, is_training=phase_train)
        d_4 = lrelu(d_4)

        d_5 = tf.nn.conv3d(d_4, weights['wd5'], strides=[1, 1, 1, 1, 1], padding="VALID")

        d_5_no_sigmoid = d_5
        d_5 = tf.nn.sigmoid(d_5)

    print(d_1, 'd1')
    print(d_2, 'd2')
    print(d_3, 'd3')
    print(d_4, 'd4')
    print(d_5, 'd5')

    return d_5, d_5_no_sigmoid


def initialiseWeights():
    global weights
    xavier_init = tf.contrib.layers.xavier_initializer()

    weights['wg1'] = tf.get_variable("wg1", shape=[4, 4, 4, 512, z_size + y_dim], initializer=xavier_init)
    weights['wg2'] = tf.get_variable("wg2", shape=[4, 4, 4, 256, 512], initializer=xavier_init)
    weights['wg3'] = tf.get_variable("wg3", shape=[4, 4, 4, 128, 256], initializer=xavier_init)
    weights['wg4'] = tf.get_variable("wg4", shape=[4, 4, 4, 64, 128], initializer=xavier_init)
    weights['wg5'] = tf.get_variable("wg5", shape=[4, 4, 4, 1, 64], initializer=xavier_init)

    weights['wd1'] = tf.get_variable("wd1", shape=[4, 4, 4, 2, 64], initializer=xavier_init)
    weights['wd2'] = tf.get_variable("wd2", shape=[4, 4, 4, 64, 128], initializer=xavier_init)
    weights['wd3'] = tf.get_variable("wd3", shape=[4, 4, 4, 128, 256], initializer=xavier_init)
    weights['wd4'] = tf.get_variable("wd4", shape=[4, 4, 4, 256, 512], initializer=xavier_init)
    weights['wd5'] = tf.get_variable("wd5", shape=[4, 4, 4, 512, 1], initializer=xavier_init)

    return weights


def trainGAN(is_dummy=False, checkpoint=None):
    weights = initialiseWeights()

    z_vector = tf.placeholder(shape=[batch_size, z_size], dtype=tf.float32)
    y_vector = tf.placeholder(tf.float32, [batch_size, y_dim])
    x_vector = tf.placeholder(shape=[batch_size, cube_len, cube_len, cube_len, 1], dtype=tf.float32)

    net_g_train = generator(z_vector, y_vector, phase_train=True, reuse=False)

    y_mask = tf.placeholder(tf.float32, shape=[batch_size, 64, 64, 64, 1])

    d_output_x, d_no_sigmoid_output_x = discriminator(x_vector, y_mask, phase_train=True, reuse=False)
    d_output_x = tf.maximum(tf.minimum(d_output_x, 0.99), 0.01)
    summary_d_x_hist = tf.summary.histogram("d_prob_x", d_output_x)

    d_output_z, d_no_sigmoid_output_z = discriminator(net_g_train, y_mask, phase_train=True, reuse=True)
    d_output_z = tf.maximum(tf.minimum(d_output_z, 0.99), 0.01)
    summary_d_z_hist = tf.summary.histogram("d_prob_z", d_output_z)

    # Compute the discriminator accuracy
    n_p_x = tf.reduce_sum(tf.cast(d_output_x > 0.5, tf.int32))
    n_p_z = tf.reduce_sum(tf.cast(d_output_z < 0.5, tf.int32))
    d_acc = tf.divide(n_p_x + n_p_z, 2 * batch_size)

    # Compute the discriminator and generator loss
    # d_loss = -tf.reduce_mean(tf.log(d_output_x) + tf.log(1-d_output_z))
    # g_loss = -tf.reduce_mean(tf.log(d_output_z))

    d_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_no_sigmoid_output_x, labels=tf.ones_like(
        d_output_x)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=d_no_sigmoid_output_z,
                                                               labels=tf.zeros_like(d_output_z))
    g_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_no_sigmoid_output_z, labels=tf.ones_like(d_output_z))


    d_loss = tf.reduce_mean(d_loss)
    g_loss = tf.reduce_mean(g_loss)

    summary_d_loss = tf.summary.scalar("d_loss", d_loss)
    summary_g_loss = tf.summary.scalar("g_loss", g_loss)
    summary_n_p_z = tf.summary.scalar("n_p_z", n_p_z)
    summary_n_p_x = tf.summary.scalar("n_p_x", n_p_x)
    summary_d_acc = tf.summary.scalar("d_acc", d_acc)

    net_g_test = generator(z_vector, y_vector, phase_train=False, reuse=True)

    para_g = [var for var in tf.trainable_variables() if any(x in var.name for x in ['wg', 'bg', 'gen'])]
    para_d = [var for var in tf.trainable_variables() if any(x in var.name for x in ['wd', 'bd', 'dis'])]

    # only update the weights for the discriminator network
    optimizer_op_d = tf.train.AdamOptimizer(learning_rate=d_lr, beta1=beta).minimize(d_loss, var_list=para_d)
    # only update the weights for the generator network
    optimizer_op_g = tf.train.AdamOptimizer(learning_rate=g_lr, beta1=beta).minimize(g_loss, var_list=para_g)

    saver = tf.train.Saver()
    # vis = visdom.Visdom()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        if checkpoint is not None:
            saver.restore(sess, checkpoint)

        if is_dummy:
            volumes = np.random.randint(0, 2, (batch_size, cube_len, cube_len, cube_len))
            print('Using Dummy Data')
        else:
            volumes = d.getAllchair(obj=obj, train=True, is_local=is_local, obj_ratio=obj_ratio)
            print('Using ' + obj + ' Data')
        volumes = volumes[..., np.newaxis].astype(np.float)
        # volumes *= 2.0
        # volumes -= 1.0
        bound_list, mask_list = d.getBounds()
        #test_bound = np.array([22, 22, 0, 42, 42, 62], dtype=float)
        test_bound = np.array([22, 22, 0, 42, 42, 62], dtype=int)
        test_bound_list = np.tile(test_bound, bound_list.shape[0])
        test_bound_list = test_bound_list.reshape((bound_list.shape[0], bound_list.shape[1]))
        test_mask_list=d.get_mask_list_by_bound(test_bound_list)
        for epoch in range(n_epochs):

            idx = np.random.randint(len(volumes), size=batch_size)
            x = volumes[idx]
            y = bound_list[idx]/64
            y_mask_array = mask_list[idx]

            z_sample = np.random.normal(0, 0.33, size=[batch_size, z_size]).astype(np.float32)
            z = np.random.normal(0, 0.33, size=[batch_size, z_size]).astype(np.float32)
            # z = np.random.uniform(0, 1, size=[batch_size, z_size]).astype(np.float32)

            # Update the discriminator and generator
            d_summary_merge = tf.summary.merge([summary_d_loss,
                                                summary_d_x_hist,
                                                summary_d_z_hist,
                                                summary_n_p_x,
                                                summary_n_p_z,
                                                summary_d_acc])

            summary_d, discriminator_loss = sess.run([d_summary_merge, d_loss],
                                                     feed_dict={z_vector: z, x_vector: x, y_vector: y, y_mask: y_mask_array})
            summary_g, generator_loss = sess.run([summary_g_loss, g_loss], feed_dict={z_vector: z, y_vector: y, y_mask: y_mask_array})
            d_accuracy, n_x, n_z = sess.run([d_acc, n_p_x, n_p_z],
                                            feed_dict={z_vector: z, x_vector: x,y_vector: y, y_mask: y_mask_array})
            print(n_x, n_z)

            if d_accuracy < d_thresh:
                sess.run([optimizer_op_d], feed_dict={z_vector: z,x_vector: x,y_vector:y,y_mask: y_mask_array})

                print('Discriminator Training ', "epoch: ", epoch, ', d_loss:', discriminator_loss, 'g_loss:',
                      generator_loss, "d_acc: ", d_accuracy)

                list.append(dict(netType='D', epoch=epoch, d_loss=discriminator_loss.item(), g_loss=generator_loss.item(), d_acc=d_accuracy))
            sess.run([optimizer_op_g], feed_dict={z_vector: z,y_vector:y,y_mask: y_mask_array})




            list.append(dict(netType='G', epoch=epoch, d_loss=discriminator_loss.item(), g_loss=generator_loss.item(), d_acc=d_accuracy))

            print('Generator Training ', "epoch: ", epoch, ', d_loss:', discriminator_loss, 'g_loss:', generator_loss,
                  "d_acc: ", d_accuracy)

            # output generated chairs
            if epoch % 200 == 0:
                #use fixed bound and mask
                y = test_bound_list[idx] / 64
                y_mask_array = test_mask_list[idx]

                g_objects = sess.run(net_g_test, feed_dict={z_vector: z_sample, y_vector: y,y_mask: y_mask_array})
                if not os.path.exists(train_sample_directory):
                    os.makedirs(train_sample_directory)
                g_objects.dump(train_sample_directory + '/biasfree_' + str(epoch))
                id_ch = np.random.randint(0, batch_size, 4)
                for i in range(4):
                    if g_objects[id_ch[i]].max() > 0.5:
                        d.plotVoxelVisdom(np.squeeze(g_objects[id_ch[i]] > 0.5), '_'.join(map(str, [epoch, i])))

            if epoch % 400 == 10:
                if not os.path.exists(model_directory):
                    os.makedirs(model_directory)
                saver.save(sess, save_path=model_directory + '/biasfree_' + str(epoch) + '.cptk')
            # if epoch == 10:
            #     mongo.save_file("stone",g_lr, d_lr,z_size, list)

IsInit = False
def testGAN(trained_model_path=None, n_batches=40):
    weights = initialiseWeights()

    z_vector = tf.placeholder(shape=[batch_size, z_size], dtype=tf.float32)
    y_vector = tf.placeholder(tf.float32, [None, y_dim])
    y_mask = tf.placeholder(tf.float32, shape=[batch_size, 64, 64, 64, 1])

    net_g_test = generator(z_vector, y_vector,y_mask, phase_train=True)

    vis = visdom.Visdom()

    sess = tf.Session()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver.restore(sess, trained_model_path)

        # output generated chairs
        for i in range(n_batches):

            next_sigma = float(input())

            z_sample = np.random.normal(0, next_sigma, size=[batch_size, z_size]).astype(np.float32)
            g_objects = sess.run(net_g_test, feed_dict={z_vector: z_sample})
            id_ch = np.random.randint(0, batch_size, 4)
            for i in range(4):
                print(g_objects[id_ch[i]].max(), g_objects[id_ch[i]].min(), g_objects[id_ch[i]].shape)
                if g_objects[id_ch[i]].max() > 0.5:
                    d.plotVoxelVisdom(np.squeeze(g_objects[id_ch[i]] > 0.5), vis, '_'.join(map(str, [i])))

min_x=min_y=min_z=max_x=max_y=max_z=0
def SetBoundState(ix,iy,iz,ax,ay,az):
    global min_x,min_y,min_z,max_x,max_y,max_z
    min_x=ix
    min_y=iy
    min_z=iz
    max_x=ax
    max_y=ay
    max_z=az

def ganGetOneResult(trained_model_path):
    global IsInit
    if IsInit == False:
        weights = initialiseWeights()
        IsInit= True
    z_vector = tf.placeholder(shape=[batch_size, z_size], dtype=tf.float32)
    y_vector = tf.placeholder(tf.float32, [batch_size, y_dim])
    net_g_test = generator(z_vector, y_vector, phase_train=True)
    saver = tf.train.Saver()
    with tf.variable_scope(tf.get_variable_scope(), reuse=None):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, trained_model_path)
            while True:
                test_bound = np.array([min_x, min_y, min_z, max_x, max_y, max_z], dtype=float)
                test_bound_list = np.tile(test_bound, batch_size)
                test_bound_list = test_bound_list.reshape((batch_size, 6))
                test_mask_list = d.get_mask_list_by_bound(test_bound_list)
                y_mask = tf.placeholder(tf.float32, shape=[batch_size, 64, 64, 64, 1])
                z_sample = np.random.normal(0, 0.33, size=[batch_size, z_size]).astype(np.float32)
                y = test_bound_list/ 64
                y_mask_array = test_mask_list
                g_objects = sess.run(net_g_test, feed_dict={z_vector: z_sample, y_vector: y, y_mask: y_mask_array})
                id_ch = np.random.randint(0, batch_size, 4)
                i=0
                print(g_objects[id_ch[i]].max(), g_objects[id_ch[i]].min(), g_objects[id_ch[i]].shape)
                if g_objects[id_ch[i]].max() > 0.5:
                    result= np.squeeze(g_objects[id_ch[i]] > 0.5)
                    yield result
    yield None

if __name__ == '__main__':
    trainGAN(is_dummy=False, checkpoint=None)