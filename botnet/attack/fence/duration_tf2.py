import math
import random

import numpy as np
import tensorflow as tf

from  fence.neris_model_data_utilities import  get_raw_delta

def update_duration(self, pbd_raw_adversary, pbd_scaled_adversary,input_raw_vector, input_vector, shape, inds,
                    iter_conns_tcp, iter_conns_udp, iter_packets_udp, iter_packets_tcp,
                    total_packets_udp, total_packets_tcp, d_max):
    pbd_scaled_adversary1 = tf.convert_to_tensor(pbd_scaled_adversary, dtype=tf.float32, name="pbd_scaled_adversary1")
    with tf.GradientTape() as tape1:
    # explicitly indicate that our image should be tacked for
    # gradient updates
        tape1.watch(pbd_scaled_adversary1)
        # use our model to make predictions on the input image and
        # then compute the loss
        pred = self.model.model(pbd_scaled_adversary1, training=False)

        loss_logit =  self.model.model(pbd_scaled_adversary1)
        #loss_logit = self.bce(tf.convert_to_tensor([self.label]), pred)

    res_grad = tape1.gradient(loss_logit, pbd_scaled_adversary1)
    
    
    #res_grad = sess.run(gradient, feed_dict={attack: pbd_scaled_adversary})
    f_id = self.DURATION_ID
    abs_grad = abs(res_grad)
    delta_duration = abs_grad[0, inds[f_id]]

    if(res_grad[0, inds[f_id]] < 0):
        delta_sign = -1 
    else:
        delta_sign = 1
    raw_delta = np.nan_to_num(get_raw_delta(pbd_scaled_adversary, delta_duration, delta_sign, self.adv_scaler, inds[f_id], shape))

    if inds[f_id] in self.integers: 
        if delta_sign < 0:
            raw_delta = math.ceil(raw_delta)
        else:
            raw_delta = math.floor(raw_delta)

    delta_duration = abs(raw_delta)

    new_duration = pbd_raw_adversary[0, inds[f_id]] - raw_delta

    if new_duration < input_raw_vector[0, inds[f_id]] :
        new_duration = input_raw_vector[0, inds[f_id]] 
        delta_duration = pbd_raw_adversary[0, inds[f_id]] - new_duration
    
    total_conns_udp = pbd_raw_adversary[0, inds[self.UDP_ID]] - input_raw_vector[0, inds[self.UDP_ID]]
    total_conns_tcp = pbd_raw_adversary[0, inds[self.TCP_ID]] - input_raw_vector[0, inds[self.TCP_ID]]

    total_duration = new_duration - input_raw_vector[0, inds[f_id]] 
    total_packets = total_packets_udp + total_packets_tcp

    #lower boundary
    if total_duration < total_packets* self.MIN_DURATION_PACKET_TCP:

        delta_duration = total_packets * self.MIN_DURATION_PACKET_TCP * -1

        #udpate dependencies        
        pbd_raw_adversary[0, inds[f_id]], pbd_raw_adversary[0, inds[f_id + 1]], pbd_raw_adversary[0, inds[f_id+ 2]] = self.update_from_total_up(input_raw_vector,
                                                                                input_raw_vector, inds, delta_duration, f_id, total_conns_udp,  total_conns_tcp,
                                                                                total_packets_udp, total_packets_tcp)
    
    
    #upper boundary
    elif total_duration > total_packets_tcp * self.MAX_DURATION_PACKET_TCP + total_packets_udp * self.MAX_DURATION_PACKET_UDP:
        delta_duration = (total_packets_tcp * self.MAX_DURATION_PACKET_TCP + total_packets_udp * self.MAX_DURATION_PACKET_UDP) * -1

        #udpate dependencies
        pbd_raw_adversary[0, inds[f_id]], pbd_raw_adversary[0, inds[f_id + 1]], pbd_raw_adversary[0, inds[f_id+ 2]]  = self.update_from_total_up(input_raw_vector, 
                                                                        input_raw_vector, inds, delta_duration, f_id,total_conns_udp,  total_conns_tcp,
                                                                            total_packets_udp, total_packets_tcp)
    #feasible update
    else:
        if delta_sign < 0:
            pbd_raw_adversary[0, inds[f_id]], pbd_raw_adversary[0, inds[f_id + 1]], pbd_raw_adversary[0, inds[f_id+ 2]]  = self.update_from_total_up(pbd_raw_adversary, input_raw_vector, inds, delta_duration * delta_sign,
                                                                                f_id,total_conns_udp,  total_conns_tcp, total_packets_udp, total_packets_tcp)
            
    
        elif  pbd_raw_adversary[0, inds[f_id]] != input_raw_vector[0, inds[f_id]] and delta_sign > 0:
            pbd_raw_adversary[0, inds[f_id]], pbd_raw_adversary[0, inds[f_id + 1]], pbd_raw_adversary[0, inds[f_id+ 2]] = self.update_from_total_down(pbd_raw_adversary, input_raw_vector, inds, delta_duration * delta_sign,
                                                                                    f_id, total_conns_udp,  total_conns_tcp, total_packets_udp, total_packets_tcp)

    return pbd_raw_adversary



def update_max_min_duration(self, total_added, total_conns_tcp, total_conns_udp, total_packets_tcp, total_packets_udp,new_max_f, new_min_f, raw_delta):
    #both types of connections were added
    if total_conns_tcp != 0 and total_conns_udp !=0:

        total_udp_added_packets = total_packets_udp
        total_tcp_added_packets = total_packets_tcp
        #lower bound
        if abs(raw_delta)  == total_udp_added_packets *  self.MIN_DURATION_PACKET_UDP  + total_tcp_added_packets * self.MIN_DURATION_PACKET_TCP  :
        
            if total_conns_tcp != 1:
                min_tcp_per_connection = math.floor(total_tcp_added_packets * self.MIN_DURATION_PACKET_TCP/ total_conns_tcp)
                max_tcp_per_connection = total_tcp_added_packets * self.MIN_DURATION_PACKET_TCP - (total_conns_tcp - 1) * min_tcp_per_connection

            else:
                min_tcp_per_connection = total_tcp_added_packets * self.MIN_DURATION_PACKET_TCP
                max_tcp_per_connection = total_tcp_added_packets * self.MIN_DURATION_PACKET_TCP 


            if total_conns_udp != 1:
                min_udp_per_connection = math.floor(total_udp_added_packets * self.MIN_DURATION_PACKET_UDP/ total_conns_udp)
                max_udp_per_connection = total_udp_added_packets * self.MIN_DURATION_PACKET_UDP - (total_conns_udp - 1) * min_udp_per_connection

            else:
                min_udp_per_connection = total_udp_added_packets * self.MIN_DURATION_PACKET_UDP
                max_udp_per_connection = total_udp_added_packets * self.MIN_DURATION_PACKET_UDP
        #upper bound
        elif abs(raw_delta) == total_udp_added_packets *  self.MAX_DURATION_PACKET_UDP + total_tcp_added_packets * self.MAX_DURATION_PACKET_TCP :

            if total_conns_tcp != 1:
                min_tcp_per_connection = math.floor(total_tcp_added_packets * self.MAX_DURATION_PACKET_TCP/ total_conns_tcp)
                max_tcp_per_connection = total_tcp_added_packets * self.MAX_DURATION_PACKET_TCP - (total_conns_tcp - 1) * min_tcp_per_connection
            else:
                min_tcp_per_connection = total_tcp_added_packets * self.MAX_DURATION_PACKET_TCP
                max_tcp_per_connection = total_tcp_added_packets * self.MAX_DURATION_PACKET_TCP 
            
            if total_conns_udp != 1:
                min_udp_per_connection = math.floor(total_udp_added_packets * self.MAX_DURATION_PACKET_UDP/ total_conns_udp)
                max_udp_per_connection = total_udp_added_packets * self.MAX_DURATION_PACKET_UDP - (total_conns_udp - 1) * min_udp_per_connection
            else:
                min_udp_per_connection = total_udp_added_packets * self.MAX_DURATION_PACKET_UDP
                max_udp_per_connection = total_udp_added_packets * self.MAX_DURATION_PACKET_UDP 

        else:

            udp_min_duration = self.MIN_DURATION_PACKET_UDP * total_udp_added_packets
            udp_max_duration = self.MAX_DURATION_PACKET_UDP * total_udp_added_packets
            tcp_min_duration = self.MIN_DURATION_PACKET_TCP * total_tcp_added_packets
            tcp_max_duration = self.MAX_DURATION_PACKET_TCP * total_tcp_added_packets

            lower_rand_udp = max(udp_min_duration, total_added - tcp_max_duration)
            upper_rand_udp = min(udp_max_duration, total_added - tcp_min_duration)

            duration_udp = random.uniform(lower_rand_udp, upper_rand_udp)
            duration_tcp = total_added - duration_udp

            if total_conns_udp != 1:
                min_udp_per_connection = math.floor(duration_udp/ total_conns_udp)
                max_udp_per_connection = duration_udp - (total_conns_udp - 1) * min_udp_per_connection                    

            else:
                min_udp_per_connection = duration_udp
                max_udp_per_connection = duration_udp

            if total_conns_tcp != 1:

                min_tcp_per_connection = math.floor(duration_tcp/ total_conns_tcp)
                max_tcp_per_connection = duration_tcp - (total_conns_tcp - 1) * min_tcp_per_connection

            else:
                min_tcp_per_connection = duration_tcp
                max_tcp_per_connection = duration_tcp


        minimum = min_udp_per_connection
        if min_tcp_per_connection < min_udp_per_connection:
            minimum = min_tcp_per_connection

        if minimum < new_min_f :
            new_min_f = minimum
        
        maximum = max_udp_per_connection
        if max_tcp_per_connection > max_udp_per_connection:
            maximum = max_tcp_per_connection

        if maximum > new_max_f :
            new_max_f = maximum
    #only tcp were added
    elif total_conns_tcp != 0:

        if total_conns_tcp != 1:
            min_tcp_per_connection = math.floor(total_added/ total_conns_tcp)
            max_tcp_per_connection = total_added - (total_conns_tcp - 1) * min_tcp_per_connection

        else:
            min_tcp_per_connection = total_added
            max_tcp_per_connection = total_added 

        if min_tcp_per_connection < new_min_f:
            new_min_f = min_tcp_per_connection

        if max_tcp_per_connection > new_max_f:
            new_max_f = max_tcp_per_connection

    #only udp were added
    elif total_conns_udp != 0:

        if total_conns_udp != 1:
            min_udp_per_connection = math.floor(total_added/ total_conns_udp)
            max_udp_per_connection = total_added - (total_conns_udp - 1) * min_udp_per_connection

        else:
            min_udp_per_connection = total_added
            max_udp_per_connection = total_added                 

        if min_udp_per_connection < new_min_f:
            new_min_f = min_udp_per_connection

        if max_udp_per_connection > new_max_f:
            new_max_f = max_udp_per_connection

    return new_max_f, new_min_f