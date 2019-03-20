import tensorflow as tf

from encoder_decoder_model import vgg_encoder
from encoder_decoder_model import fcn_decoder
from encoder_decoder_model import dense_encoder
from encoder_decoder_model import cnn_basenet
from lanenet_model import lanenet_discriminative_loss


class LaneNet(cnn_basenet.CNNBaseModel):

    def __init__(self, phase, net_flag='vgg'):
        """

        """
        super(LaneNet, self).__init__()
        self._net_flag = net_flag
        self._phase = phase
        if self._net_flag == 'vgg':
            self._encoder = vgg_encoder.VGG16Encoder(phase=phase)
        elif self._net_flag == 'dense':
            self._encoder = dense_encoder.DenseEncoder(l=20, growthrate=8,
                                                       with_bc=True,
                                                       phase=phase,
                                                       n=5)
        self._decoder = fcn_decoder.FCNDecoder(phase=phase)
        return

    def __str__(self):
        """

        :return:
        """
        info = 'Semantic Segmentation use {:s} as basenet to encode'.format(self._net_flag)
        return info






















    def _build_model(self, input_tensor, name):
        """
        :param input_tensor:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            # first encode
            encode_ret = self._encoder.encode(input_tensor=input_tensor,
                                              name='encode')

            # second decode
            if self._net_flag.lower() == 'vgg':
                decode_ret = self._decoder.decode(input_tensor_dict=encode_ret,
                                                  name='decode',
                                                  decode_layer_list=['pool5',
                                                                     'pool4',
                                                                     'pool3'])
                return decode_ret
                
                
            elif self._net_flag.lower() == 'dense':
                decode_ret = self._decoder.decode(input_tensor_dict=encode_ret,
                                                  name='decode',
                                                  decode_layer_list=['Dense_Block_5',
                                                                     'Dense_Block_4',
                                                                     'Dense_Block_3'])
                return decode_ret





























    def compute_loss(self, input_tensor, binary_label, instance_label, name):
        """
        :param input_tensor:
        :param binary_label:
        :param instance_label:
        :param name:
        :return:
        """
        with tf.variable_scope(name):

            inference_ret = self._build_model(input_tensor=input_tensor, name='inference')

            decode_logits = inference_ret['logits']
            binary_label_plain = tf.reshape(
                binary_label,
                shape=[binary_label.get_shape().as_list()[0] *
                       binary_label.get_shape().as_list()[1] *
                       binary_label.get_shape().as_list()[2]])

            unique_labels, unique_id, counts = tf.unique_with_counts(binary_label_plain)
            counts = tf.cast(counts, tf.float32)
            inverse_weights = tf.divide(1.0,
                                        tf.log(tf.add(tf.divide(tf.constant(1.0), counts),
                                                      tf.constant(1.02))))
            inverse_weights = tf.gather(inverse_weights, binary_label)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            binary_segmenatation_loss = tf.losses.sparse_softmax_cross_entropy(
                labels=binary_label, logits=decode_logits, weights=inverse_weights)
                
                
            binary_segmenatation_loss = tf.reduce_mean(binary_segmenatation_loss)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            


            decode_deconv = inference_ret['deconv']

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            pix_embedding = self.conv2d(inputdata=decode_deconv, out_channel=4, kernel_size=1,
                                        use_bias=False, name='pix_embedding_conv')
            pix_embedding = self.relu(inputdata=pix_embedding, name='pix_embedding_relu')
            
            
            
            
            
            
            
            
            
            
            

            image_shape = (pix_embedding.get_shape().as_list()[1], pix_embedding.get_shape().as_list()[2])
            disc_loss, l_var, l_dist, l_reg = \
                lanenet_discriminative_loss.discriminative_loss(
                    pix_embedding, instance_label, 4, image_shape, 0.5, 3.0, 1.0, 1.0, 0.001)













            l2_reg_loss = tf.constant(0.0, tf.float32)
            for vv in tf.trainable_variables():
                if 'bn' in vv.name:
                    continue
                else:
                    l2_reg_loss = tf.add(l2_reg_loss, tf.nn.l2_loss(vv))
            l2_reg_loss *= 0.001
            total_loss = 0.5 * binary_segmenatation_loss + 0.5 * disc_loss + l2_reg_loss

            ret = {
                'total_loss': total_loss,
                'binary_seg_logits': decode_logits,
                'instance_seg_logits': pix_embedding,
                'binary_seg_loss': binary_segmenatation_loss,
                'discriminative_loss': disc_loss
            }

            return ret































    def inference(self, input_tensor, name):
        """

        :param input_tensor:
        :param name:
        :return:
        """
        with tf.variable_scope(name):

            inference_ret = self._build_model(input_tensor=input_tensor, name='inference')

            decode_logits = inference_ret['logits']
            binary_seg_ret = tf.nn.softmax(logits=decode_logits)
            binary_seg_ret = tf.argmax(binary_seg_ret, axis=-1)

            decode_deconv = inference_ret['deconv']

            pix_embedding = self.conv2d(inputdata=decode_deconv, out_channel=4, kernel_size=1,
                                        use_bias=False, name='pix_embedding_conv')
            pix_embedding = self.relu(inputdata=pix_embedding, name='pix_embedding_relu')

            return binary_seg_ret, pix_embedding


if __name__ == '__main__':
    model = LaneNet(tf.constant('train', dtype=tf.string))
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input')
    binary_label = tf.placeholder(dtype=tf.int64, shape=[1, 256, 512, 1], name='label')
    instance_label = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 1], name='label')
    ret = model.compute_loss(input_tensor=input_tensor, binary_label=binary_label,
                             instance_label=instance_label, name='loss')
    for vv in tf.trainable_variables():
        if 'bn' in vv.name:
            continue
        print(vv.name)