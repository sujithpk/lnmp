import tensorflow as tf


def hnet_loss(gt_pts, transformation_coeffcient, name):
    """
    
    :param name:
    :return: 
    """
    with tf.variable_scope(name):

        transformation_coeffcient = tf.concat([transformation_coeffcient, [1.0]], axis=-1)
        H_indices = tf.constant([[0], [1], [2], [4], [5], [7], [8]])
        H_shape = tf.constant([9])
        H = tf.scatter_nd(H_indices, transformation_coeffcient, H_shape)
        H = tf.reshape(H, shape=[3, 3])

        gt_pts = tf.transpose(gt_pts)
        pts_projects = tf.matmul(H, gt_pts)

        Y = tf.transpose(pts_projects[1, :])
        X = tf.transpose(pts_projects[0, :])
        Y_One = tf.add(tf.subtract(Y, Y), tf.constant(1.0, tf.float32))
        Y_stack = tf.stack([tf.pow(Y, 3), tf.pow(Y, 2), Y, Y_One], axis=1)
        w = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(Y_stack), Y_stack)),
                                tf.transpose(Y_stack)),
                      tf.expand_dims(X, -1))

        x_preds = tf.matmul(Y_stack, w)
        preds = tf.transpose(tf.stack([tf.squeeze(x_preds, -1), Y, Y_One], axis=1))
        x_transformation_back = tf.matmul(tf.matrix_inverse(H), preds)

        loss = tf.reduce_mean(tf.pow(gt_pts[0, :] - x_transformation_back[0, :], 2))

    return loss


def hnet_transformation(gt_pts, transformation_coeffcient, name):
    """

    :param gt_pts:
    :param transformation_coeffcient:
    :param name:
    :return:
    """
    with tf.variable_scope(name):
        transformation_coeffcient = tf.concat([transformation_coeffcient, [1.0]], axis=-1)
        H_indices = tf.constant([[0], [1], [2], [4], [5], [7], [8]])
        H_shape = tf.constant([9])
        H = tf.scatter_nd(H_indices, transformation_coeffcient, H_shape)
        H = tf.reshape(H, shape=[3, 3])

        gt_pts = tf.transpose(gt_pts)
        pts_projects = tf.matmul(H, gt_pts)

        Y = tf.transpose(pts_projects[1, :])
        X = tf.transpose(pts_projects[0, :])
        Y_One = tf.add(tf.subtract(Y, Y), tf.constant(1.0, tf.float32))
        Y_stack = tf.stack([tf.pow(Y, 3), tf.pow(Y, 2), Y, Y_One], axis=1)
        w = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(Y_stack), Y_stack)),
                                tf.transpose(Y_stack)),
                      tf.expand_dims(X, -1))

        x_preds = tf.matmul(Y_stack, w)
        preds = tf.transpose(tf.stack([tf.squeeze(x_preds, -1), Y, Y_One], axis=1))
        preds_fit = tf.stack([tf.squeeze(x_preds, -1), Y], axis=1)
        x_transformation_back = tf.matmul(tf.matrix_inverse(H), preds)

    return x_transformation_back


if __name__ == '__main__':
    gt_labels = tf.constant([[[1.0, 1.0, 1.0], [2.0, 2.0, 1.0], [3.0, 3.0, 1.0]],
                             [[1.0, 1.0, 1.0], [2.0, 2.0, 1.0], [3.0, 3.0, 1.0]]],
                            dtype=tf.float32, shape=[6, 3])
    transformation_coffecient = tf.constant([[0.58348501, -0.79861236, 2.30343866,
                                              -0.09976104, -1.22268307, 2.43086767]],
                                            dtype=tf.float32, shape=[6])

    # import numpy as np
    # c_val = [0.58348501, -0.79861236, 2.30343866,
    #          -0.09976104, -1.22268307, 2.43086767]
    # R = np.zeros([3, 3], np.float32)
    # R[0, 0] = c_val[0]
    # R[0, 1] = c_val[1]
    # R[0, 2] = c_val[2]
    # R[1, 1] = c_val[3]
    # R[1, 2] = c_val[4]
    # R[2, 1] = c_val[5]
    # R[2, 2] = 1
    #
    # print(np.mat(R).I)

    _loss = hnet_loss(gt_labels, transformation_coffecient, 'loss')

    _pred = hnet_transformation(gt_labels, transformation_coffecient, 'inference')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        loss_val = sess.run(_loss)
        pred = sess.run(_pred)
        print(loss_val)
        print(pred)
