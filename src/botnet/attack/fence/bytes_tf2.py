import math
import random

import numpy as np
import tensorflow as tf

from fence.duration_tf2 import  update_duration
from fence.neris_model_data_utilities import  get_raw_delta



def update_bytes(self, pbd_scaled_adversary, pbd_raw_adversary, input_vector_raw, input_vector, inds, shape,
                iter_conns_tcp,  iter_conns_udp, total_conns_tcp, total_conns_udp, iter_packets_udp, iter_packets_tcp, total_packets_udp, total_packets_tcp, d_max):
    pbd_raw_adversary1 = np.copy(pbd_raw_adversary)
    f_id = self.BYTES_ID
    pbd_scaled_adversary1 = tf.convert_to_tensor(pbd_scaled_adversary, dtype=tf.float32, name="pbd_scaled_adversary1")
    with tf.GradientTape() as tape1:
    # explicitly indicate that our image should be tacked for
    # gradient updates
        tape1.watch(pbd_scaled_adversary1)
        # use our model to make predictions on the input image and
        # then compute the loss
        #pred = self.model.model(pbd_scaled_adversary1,training=False)

        loss_logit =  self.model.model(pbd_scaled_adversary1)
        #loss_logit = self.bce(tf.convert_to_tensor([self.label]), pred)

    res_grad = tape1.gradient(loss_logit, pbd_scaled_adversary1)
    #res_grad = sess.run(gradient, feed_dict={attack: pbd_scaled_adversary})
    abs_grad = abs(res_grad)
    delta_bytes = abs_grad[0, inds[f_id]]
    
    if(res_grad[0, inds[f_id]] < 0):
        delta_sign = -1 
    else:
        delta_sign = 1

    raw_delta = np.nan_to_num(get_raw_delta(pbd_scaled_adversary, delta_bytes, delta_sign, self.adv_scaler, inds[f_id], shape))

    if inds[f_id] in self.integers: 
        if delta_sign < 0:
            raw_delta = math.ceil(raw_delta)
        else:
            raw_delta = math.floor(raw_delta)

    delta_bytes = abs(raw_delta)

    new_bytes = pbd_raw_adversary[0, inds[f_id]] - raw_delta

    if new_bytes < input_vector_raw[0, inds[f_id]] :
        new_bytes = input_vector_raw[0, inds[f_id]] 
        delta_bytes = pbd_raw_adversary[0, inds[f_id]] - new_bytes 
    
    total_bytes = new_bytes - input_vector_raw[0, inds[f_id]] 
    total_packets = total_packets_udp + total_packets_tcp


    #upper boundary
    if total_bytes >= total_packets_tcp * self.MAX_BYTES_PACKET_TCP + total_packets_udp * self.MAX_BYTES_PACKET_UDP:
        delta_bytes  = (total_packets_tcp * self.MAX_BYTES_PACKET_TCP + total_packets_udp * self.MAX_BYTES_PACKET_UDP ) * -1


        pbd_raw_adversary[0, inds[f_id]], pbd_raw_adversary[0, inds[f_id + 1]], pbd_raw_adversary[0, inds[f_id+ 2]] = self.update_from_total_up(pbd_raw_adversary,
                                                                            input_vector_raw, inds, delta_bytes, f_id, total_conns_udp, total_conns_tcp,
                                                                            total_packets_udp, total_packets_tcp)

        pbd_scaled_adversary = self.adv_scaler.transform(pbd_raw_adversary)

        pbd_raw_adversary = update_duration(self, pbd_raw_adversary, pbd_scaled_adversary, input_vector_raw, input_vector, shape, inds, iter_conns_tcp,
                                                    iter_conns_udp,  iter_packets_udp, iter_packets_tcp,
                                                total_packets_udp, total_packets_tcp,  d_max)
    #lower boundary
    elif total_bytes <= total_packets * self.MIN_BYTES_PACKET_TCP:

        delta_bytes = total_packets * self.MIN_BYTES_PACKET_TCP * -1

        pbd_raw_adversary[0, inds[f_id]], pbd_raw_adversary[0, inds[f_id + 1]], pbd_raw_adversary[0, inds[f_id+ 2]] = self.update_from_total_up(pbd_raw_adversary, 
                                                                            input_vector_raw, inds, delta_bytes,
                                                                                f_id, total_conns_udp,  total_conns_tcp, total_packets_udp, total_packets_tcp)
    
        pbd_scaled_adversary = self.adv_scaler.transform(pbd_raw_adversary)

        pbd_raw_adversary = update_duration(self,pbd_raw_adversary, pbd_scaled_adversary,input_vector_raw, input_vector, 
                                                shape, inds, iter_conns_tcp, iter_conns_udp,
                                                iter_packets_tcp, iter_packets_tcp,total_packets_udp, total_packets_tcp, d_max)
    #feasible update
    else:

        if delta_sign < 0:
            pbd_raw_adversary[0, inds[f_id]], pbd_raw_adversary[0, inds[f_id + 1]], pbd_raw_adversary[0, inds[f_id+ 2]]= self.update_from_total_up(pbd_raw_adversary,
                                                                                    input_vector_raw, inds, delta_bytes * delta_sign,
                                                                                f_id, total_conns_udp, total_conns_tcp, total_packets_udp, total_packets_tcp)
        
        elif  pbd_raw_adversary[0, inds[f_id]] != input_vector_raw[0, inds[f_id]] and delta_sign > 0:

            pbd_raw_adversary[0, inds[f_id]], pbd_raw_adversary[0, inds[f_id + 1]], pbd_raw_adversary[0, inds[f_id+ 2]] = self.update_from_total_down(pbd_raw_adversary,
                                                                                        input_vector_raw, inds, delta_bytes * delta_sign,
                                                                                    f_id, total_conns_udp, total_conns_tcp, total_packets_udp, total_packets_tcp)

        pbd_scaled_adversary = self.adv_scaler.transform(pbd_raw_adversary)

        pbd_raw_adversary = update_duration( self,pbd_raw_adversary, pbd_scaled_adversary, input_vector_raw, input_vector, shape, inds, iter_conns_tcp, iter_conns_udp,
                                                iter_packets_udp, iter_packets_tcp, total_packets_udp, iter_packets_tcp, d_max)

    diff = pbd_raw_adversary[0, inds[f_id]] - pbd_raw_adversary1[0, inds[f_id]]


    if iter_conns_udp>0 and iter_conns_tcp>0:
        bytes_avg = int(diff/2)
        pbd_raw_adversary[0, inds[f_id+self.TCP_ID]] += bytes_avg
        pbd_raw_adversary[0, inds[f_id+self.UDP_ID]] += (diff-bytes_avg)

    elif iter_conns_tcp>0:
        pbd_raw_adversary[0, inds[f_id+self.TCP_ID]] += diff
    else :
        pbd_raw_adversary[0, inds[f_id+self.UDP_ID]] += diff

    return pbd_raw_adversary


def update_max_min_bytes(self, total_added, total_conns_tcp, total_conns_udp, total_packets_tcp, total_packets_udp, new_max_f, new_min_f, raw_delta):

    total_added = round(total_added)
    try:                    

        #both types of connection were added
        if total_conns_tcp !=0 and total_conns_udp !=0 :
            #get udp and tcp packets
            total_udp_added_packets = total_packets_udp
            total_tcp_added_packets = total_packets_tcp

            #lower bound
            if abs(raw_delta)  <= total_udp_added_packets *  self.MIN_BYTES_PACKET_UDP  + total_tcp_added_packets * self.MIN_BYTES_PACKET_TCP  :

                if total_conns_tcp != 1:
                    min_tcp_per_connection = math.floor(total_tcp_added_packets * self.MIN_BYTES_PACKET_TCP/ total_conns_tcp)
                    max_tcp_per_connection = total_tcp_added_packets * self.MIN_BYTES_PACKET_TCP - (total_conns_tcp - 1) * min_tcp_per_connection

                else:
                    min_tcp_per_connection = total_tcp_added_packets * self.MIN_BYTES_PACKET_TCP
                    max_tcp_per_connection = total_tcp_added_packets * self.MIN_BYTES_PACKET_TCP

                if total_conns_udp != 1:
                    min_udp_per_connection = math.floor(total_udp_added_packets * self.MIN_BYTES_PACKET_UDP/ total_conns_udp)
                    max_udp_per_connection = total_udp_added_packets * self.MIN_BYTES_PACKET_UDP - (total_conns_udp - 1) * min_udp_per_connection

                else:
                    min_udp_per_connection = total_udp_added_packets * self.MIN_BYTES_PACKET_UDP
                    max_udp_per_connection = total_udp_added_packets * self.MIN_BYTES_PACKET_UDP

            #upper bound
            elif abs(raw_delta) >= total_udp_added_packets *  self.MAX_BYTES_PACKET_UDP + total_tcp_added_packets * self.MAX_BYTES_PACKET_TCP :
        
                if total_conns_tcp != 1:
                    min_tcp_per_connection = math.floor(total_tcp_added_packets * self.MAX_BYTES_PACKET_TCP/ total_conns_tcp)
                    max_tcp_per_connection = total_tcp_added_packets * self.MAX_BYTES_PACKET_TCP - (total_conns_tcp - 1) * min_tcp_per_connection

                else:
                    min_tcp_per_connection = total_tcp_added_packets * self.MAX_BYTES_PACKET_TCP
                    max_tcp_per_connection = total_tcp_added_packets * self.MAX_BYTES_PACKET_TCP


                if total_conns_udp != 1:
                    min_udp_per_connection = math.floor(total_udp_added_packets * self.MAX_BYTES_PACKET_UDP/ total_conns_udp)
                    max_udp_per_connection = total_udp_added_packets * self.MAX_BYTES_PACKET_UDP - (total_conns_udp - 1) * min_udp_per_connection

                else:
                    min_udp_per_connection = total_udp_added_packets * self.MAX_BYTES_PACKET_UDP
                    max_udp_per_connection = total_udp_added_packets * self.MAX_BYTES_PACKET_UDP

            else:      
                    udp_min_bytes = self.MIN_BYTES_PACKET_UDP * total_udp_added_packets
                    udp_max_bytes = self.MAX_BYTES_PACKET_UDP * total_udp_added_packets
                    tcp_min_bytes = self.MIN_BYTES_PACKET_TCP * total_tcp_added_packets
                    tcp_max_bytes = self.MAX_BYTES_PACKET_TCP * total_tcp_added_packets

                    lower_rand_udp = round(max(udp_min_bytes, total_added - tcp_max_bytes))
                    upper_rand_udp = round(min(udp_max_bytes, total_added - tcp_min_bytes))

                    bytes_udp = random.randint(lower_rand_udp, upper_rand_udp + 1)
                    bytes_tcp = total_added - bytes_udp

                    if total_conns_udp != 1:
                        min_udp_per_connection = math.floor(bytes_udp/ total_conns_udp)
                        max_udp_per_connection = bytes_udp - (total_conns_udp - 1) * min_udp_per_connection                    

                    else:
                        min_udp_per_connection = bytes_udp
                        max_udp_per_connection = bytes_udp

                    if total_conns_tcp != 1:

                        min_tcp_per_connection = math.floor(bytes_tcp/ total_conns_tcp)
                        max_tcp_per_connection = bytes_tcp - (total_conns_tcp - 1) * min_tcp_per_connection

                    else:
                        min_tcp_per_connection = bytes_tcp
                        max_tcp_per_connection = bytes_tcp


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
    except Exception as e:
        print("total_addded",total_added)
        print("total_udp_added_packets", total_udp_added_packets)
        print("total_tcp_added_packets", total_tcp_added_packets)

    return new_max_f, new_min_f