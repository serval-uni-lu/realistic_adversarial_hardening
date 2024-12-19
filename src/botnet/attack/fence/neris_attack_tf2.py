import math
import random

import numpy as np

import tensorflow as tf
from tensorflow.random import set_seed
from tensorflow.keras.losses import BinaryCrossentropy


from fence.duration_tf2 import update_max_min_duration
from fence.bytes_tf2 import update_max_min_bytes, update_bytes
from fence.neris_model_data_utilities import MDModel,  sigmoid,  get_raw_delta

np.set_printoptions(precision = 5)
np.set_printoptions(suppress = True) 

class Neris_attack():

    def __init__(self, model_path,  iterations, distance, scaler, mins, maxs):

        #features that can be updated
        self.UPDATE = np.load("../data/mod_features.npy")

        #Families: Total bytes - Min bytes - Max bytes - Total duration- Min duration - Max duration - Total packets - Min packets - Max packets
        self.FAMILIES = np.load("../data/families.npy")
        #Integer features
        self.integers = np.load("../data/int_features.npy")
        #Number of attack iterations
        self.NUM_ITERATIONS = iterations
        self.distance = distance

        #Ports that can not have TCP or UDP connections
        self.PORTS_NO_UDP = [0, 4, 5, 6, 8, 9, 14, 16]
        self.PORTS_NO_TCP = [10, 12, 13]

        self.MODEL_PATH = model_path

        self.PACKETS_ID = 6
        self.BYTES_ID = 0
        self.DURATION_ID = 3
        self.TCP_ID = 9
        self.UDP_ID = 10
        self.ICMP_ID = 11

        self.MAX_PACKET_TCP_CONN = 87
        self.MAX_PACKET_UDP_CONN = 49

        self.MIN_DURATION_PACKET_TCP = 0.0001
        self.MIN_DURATION_PACKET_UDP = 0.0001
        self.MAX_DURATION_PACKET_TCP = 20.83
        self.MAX_DURATION_PACKET_UDP = 24.82

        self.MAX_BYTES_PACKET_TCP = 1292
        self.MAX_BYTES_PACKET_UDP = 1036
        self.MIN_BYTES_PACKET_TCP = 20
        self.MIN_BYTES_PACKET_UDP = 20

        self.num_features = 756
        self.num_port_families = 36


        self.label = 1
        self.model =  MDModel(self.MODEL_PATH) 

        self.adv_scaler = scaler
        self.min_features = mins
        self.max_features = maxs

    def attack_round(self, input_vector, label):

        self.target = 1 - label
        adversarial_samples = []

        #size of adversary
        shape = (1, self.num_features)
        #value of border features(there is no need to update them after they become 'border')
        #so we set them to some very small value, so the attack ignores them aftewards
        MAX_BORDERS = -1000000
        #maximum distance between adversary and input vector
        d_max = self.distance
        #number of successful attacks
        success = 0     
        #target = 1 - label
        
        self.bce = BinaryCrossentropy()

        #hyperparameters : number of connections that we add
        connections_ports = np.ones((2, self.num_port_families))
        #features, that already were argmaxes
        updated = []
        borders = []
        adversary = np.copy(input_vector)
        adversary1 = np.copy(adversary)
        adversary1 = tf.convert_to_tensor(adversary1, dtype=tf.float32,  name="adversary1")
        #number of added packets for both types of connections
        total_packets_udp = np.zeros(len(self.FAMILIES))
        total_packets_tcp  = np.zeros(len(self.FAMILIES))
        conns_udp = np.ones(len(self.FAMILIES))
        conns_tcp = np.ones(len(self.FAMILIES))
        for j in range(self.NUM_ITERATIONS):
            raw_adversary = self.adv_scaler.inverse_transform(adversary)
            with tf.GradientTape() as tape:
            # explicitly indicate that our image should be tacked for
            # gradient updates
                tape.watch(adversary1)
                # use our model to make predictions on the input image and
                # then compute the loss
                pred = self.model.model(adversary1, training=False)
                loss_logit =  self.model.model(adversary1)
            res_grad = tape.gradient(loss_logit, adversary1)
            abs_grad = abs(res_grad).numpy()
            abs_grad_update = np.copy(abs_grad)
            abs_grad[:,borders] = MAX_BORDERS
            arg_max = np.argmax(abs_grad)
            borders.append(arg_max)

            if arg_max in self.UPDATE:
                if arg_max not in updated:
                    updated.append(arg_max)

                    #get indicies of the family we want to update
                    for k in range(len(self.FAMILIES)):
                        if arg_max in self.FAMILIES[k]:

                            inds = self.FAMILIES[k]
                            deltas_update = abs_grad_update[0, inds]
                            delta_signs = np.zeros((deltas_update.shape))
                            for l in range(len(inds)):
                                if(res_grad[0, inds[l]] < 0):
                                    delta_signs[l] = -1 
                                else:
                                    delta_signs[l] = 1

                    port_id = int(np.floor((inds[self.BYTES_ID] - 3)/21))
                    input_vector_raw = self.adv_scaler.inverse_transform(input_vector)


                    current_scaled_adversary = np.copy(adversary)
                    current_raw_adversary = self.adv_scaler.inverse_transform(current_scaled_adversary)
                    #update connections by the value from connection_ports(hyperparameter)
                    iter_conns_tcp = 0
                    iter_conns_udp = 0
                    for k in [self.TCP_ID, self.UDP_ID]:   
                        if k == self.TCP_ID and port_id not in self.PORTS_NO_TCP:                           
                            connections_ports[k - 9, int(port_id)] += 10
                            iter_conns_tcp = connections_ports[k - 9, int(port_id)]
                            conns_tcp[port_id] = conns_tcp[port_id] + iter_conns_tcp

                        elif k == self.UDP_ID and port_id not in self.PORTS_NO_UDP:
                            connections_ports[k - 9, int(port_id)] += 10
                            iter_conns_udp = connections_ports[k - 9, int(port_id)]
                            conns_udp[port_id] = conns_udp[port_id] + iter_conns_udp

                    total_conns_tcp = conns_tcp[port_id]
                    total_conns_udp = conns_udp[port_id]

                    #update scaled feature vector after connections update
                    current_scaled_adversary = self.adv_scaler.transform(current_raw_adversary)
                    distance_tmp = np.linalg.norm(current_scaled_adversary - input_vector)

                    if iter_conns_tcp + iter_conns_udp  > 0 and distance_tmp <= d_max:

                        pbd_scaled_adversary = np.copy(current_scaled_adversary)
                        pbd_scaled_adversary1 = tf.convert_to_tensor(pbd_scaled_adversary, dtype=tf.float32, name="pbd_scaled_adversary1")

                        with tf.GradientTape() as tape1:
                            tape1.watch(pbd_scaled_adversary1)
                            pred = self.model.model(pbd_scaled_adversary1, training=False)
                            loss_logit =  self.model.model(pbd_scaled_adversary1)

                        res_grad = tape1.gradient(loss_logit, pbd_scaled_adversary1)
                        pbd_raw_adversary = np.copy(current_raw_adversary)
                        f_id = self.PACKETS_ID
                        abs_grad = abs(res_grad)
                        delta_packets = abs_grad[0, inds[f_id]]
                        if(res_grad[0, inds[f_id]] < 0):
                            delta_sign = -1 
                        else:
                            delta_sign = 1

                        raw_delta = np.nan_to_num(get_raw_delta(pbd_scaled_adversary, delta_packets, delta_sign, self.adv_scaler, inds[f_id], shape))
                        
                        
                        if inds[f_id] in self.integers: 
                            if delta_sign < 0:
                                raw_delta = math.ceil(raw_delta)
                            else:
                                raw_delta = math.floor(raw_delta)
                        raw_delta_packets = abs(raw_delta)

                        current_raw = np.copy(pbd_raw_adversary)
                                    
                        #corner case: delta_packets is 0, then we add minimum possible number of packets per UDP.TCP connection, which in our case is 2
                        if raw_delta_packets == 0:

                            raw_delta =(iter_conns_tcp + iter_conns_udp) * -2

                            iter_packets_udp = iter_conns_tcp * 2                            
                            iter_packets_tcp = iter_conns_udp * 2

                            total_packets_udp[port_id] +=iter_packets_udp
                            total_packets_tcp[port_id] +=iter_packets_tcp

                            #increase the total value of packets plus project mathematical dependencies(min/ max packets per UDP/TCP connection) if needed
                            pbd_raw_adversary[0, inds[f_id]], pbd_raw_adversary[0, inds[f_id + 1]], pbd_raw_adversary[0, inds[f_id+ 2]]  = self.update_from_total_up(pbd_raw_adversary, input_vector_raw, inds,
                                                                                                raw_delta,  f_id,total_conns_udp, total_conns_tcp, total_packets_udp[port_id], total_packets_tcp[port_id])
                            pbd_scaled_adversary = self.adv_scaler.transform(pbd_raw_adversary)
                            distance = np.linalg.norm(pbd_scaled_adversary - input_vector)

                            #check resulting distance, and if we are inside L2 norm ball proceed to updating bytes and duration
                            if distance < d_max:

                                pbd_raw_adversary = update_bytes(self, pbd_scaled_adversary, pbd_raw_adversary, input_vector_raw, input_vector, inds, shape,                                                                
                                                                iter_conns_tcp, iter_conns_udp, total_conns_tcp, total_conns_udp, iter_packets_udp, iter_packets_tcp,  total_packets_udp[port_id], total_packets_tcp[port_id], d_max)
                                                                


                                pbd_scaled_adversary = self.adv_scaler.transform(pbd_raw_adversary)
                                distance = np.linalg.norm(pbd_scaled_adversary - input_vector)
                                        
                                #check resulting distance, and if we are outside L2 norm ball reverse changes
                                if distance > d_max:
                                    pbd_scaled_adversary = adversary
                                    pbd_raw_adversary = self.adv_scaler.inverse_transform(pbd_scaled_adversary)
                                    total_packets_udp[port_id] -=iter_packets_udp
                                    total_packets_tcp[port_id] -=iter_packets_tcp

                            #check resulting distance, and if we are outside L2 norm ball reverse changes
                            else:
                                pbd_scaled_adversary = adversary
                                pbd_raw_adversary = self.adv_scaler.inverse_transform(pbd_scaled_adversary)
                                total_packets_udp[port_id] -=iter_packets_udp
                                total_packets_tcp[port_id] -=iter_packets_tcp

                        #normal case, conducting binary search on representative features, which is 'total_packets_sent' in our case
                        else:
                            while raw_delta_packets != 0:
                                        
                                new_packets = current_raw[0, inds[f_id]] - raw_delta_packets * delta_sign
                                if new_packets < input_vector_raw[0, inds[f_id]] :
                                    new_packets = input_vector_raw[0, inds[f_id]] 
                                    raw_delta_packets =  current_raw[0, inds[f_id]] - new_packets
                                #resulting number of added packets after possible update
                                total_packets = new_packets - input_vector_raw[0, inds[f_id]] 
                                    
                                #overall number of added tcp/udp connections
                                total_conns = total_conns_tcp + total_conns_udp
                                    
                                #Projection on lower bound of packets per connection(physical constraint)
                                if total_packets  < total_conns * 2:
                                            
                                    raw_delta = total_conns * -2
                                    iter_packets_udp = total_conns_udp * 2
                                    iter_packets_tcp = total_conns_tcp * 2
                                    total_packets_udp[port_id] +=iter_packets_udp
                                    total_packets_tcp[port_id] +=iter_packets_tcp
                                    #udpate total value of sent packets and adjust mathematical dependencies (min/max sent packets per connections)if needed
                                    pbd_raw_adversary[0, inds[f_id]], pbd_raw_adversary[0, inds[f_id + 1]], pbd_raw_adversary[0, inds[f_id+ 2]]  = self.update_from_total_up(pbd_raw_adversary, input_vector_raw,
                                                                                                        inds, raw_delta, f_id, total_conns_udp, total_conns_tcp, total_packets_udp[port_id], total_packets_tcp[port_id])

                                    pbd_scaled_adversary = self.adv_scaler.transform(pbd_raw_adversary)
                                    distance = np.linalg.norm(pbd_scaled_adversary - input_vector)
                                             
                                    #check resulting distance, if feature vector is inside L2 norm ball, proceed to update bytes and packets
                                    if distance < d_max:

                                        pbd_raw_adversary = update_bytes(self, pbd_scaled_adversary, pbd_raw_adversary, input_vector_raw, input_vector, inds, shape,                                                                
                                                                iter_conns_tcp, iter_conns_udp, total_conns_tcp, total_conns_udp, iter_packets_udp, iter_packets_tcp,  total_packets_udp[port_id], total_packets_tcp[port_id], d_max)          

                                        pbd_scaled_adversary = self.adv_scaler.transform(pbd_raw_adversary)
                                        distance = np.linalg.norm(pbd_scaled_adversary - input_vector)
                                        #check resulting distance, if outside L2 norm ball, reverse changes
                                        if distance > d_max:

                                            pbd_scaled_adversary = adversary
                                            pbd_raw_adversary = self.adv_scaler.inverse_transform(pbd_scaled_adversary)
                                        else:
                                            ############################################# -= => =

                                            total_packets_udp[port_id] = iter_packets_udp
                                            total_packets_tcp[port_id] = iter_packets_tcp

                                    #check resulting distance, if outside L2 norm ball, reverse changes
                                    else:

                                        pbd_scaled_adversary = adversary
                                        pbd_raw_adversary = self.adv_scaler.inverse_transform(pbd_scaled_adversary)
                                        
                                    raw_delta_packets = 0
                                        
                                #Projection on upper bound with binary search
                                elif total_packets >  self.MAX_PACKET_TCP_CONN * total_conns_tcp +  self.MAX_PACKET_UDP_CONN * total_conns_udp:

                                    raw_delta = (self.MAX_PACKET_TCP_CONN * total_conns_tcp +  self.MAX_PACKET_UDP_CONN * total_conns_udp) * -1

                                    iter_packets_udp = self.MAX_PACKET_UDP_CONN * total_conns_udp
                                    iter_packets_tcp = self.MAX_PACKET_TCP_CONN * total_conns_tcp 
                                    total_packets_udp[port_id] +=iter_packets_udp
                                    total_packets_tcp[port_id] +=iter_packets_tcp
                                    #udpate total number of sent packets and adjust mathematical dependencies
                                    pbd_raw_adversary[0, inds[f_id]], pbd_raw_adversary[0, inds[f_id + 1]], pbd_raw_adversary[0, inds[f_id+ 2]]  = self.update_from_total_up(pbd_raw_adversary, input_vector_raw,
                                                                                                    inds, raw_delta,  f_id, total_conns_udp, total_conns_tcp, total_packets_udp[port_id], total_packets_tcp[port_id])

                                    pbd_scaled_adversary = self.adv_scaler.transform(pbd_raw_adversary)

                                    #update bytes and duration
                                    pbd_raw_adversary = update_bytes(self, pbd_scaled_adversary, pbd_raw_adversary, input_vector_raw, input_vector, inds, shape,                                                                
                                                                iter_conns_tcp, iter_conns_udp, total_conns_tcp, total_conns_udp, iter_packets_udp, iter_packets_tcp,  total_packets_udp[port_id], total_packets_tcp[port_id], d_max)
                                    pbd_scaled_adversary = self.adv_scaler.transform(pbd_raw_adversary)
                                    distance_tmp = np.linalg.norm(pbd_scaled_adversary - input_vector)

                                    current_prob = sigmoid(self.model.model(pbd_scaled_adversary,training=False))
                                    #perform binary search if we are outside L2 norm ball or probability is > 0.5
                                    if distance_tmp > d_max or current_prob > 0.5:

                                        raw_delta_packets  = math.floor(raw_delta_packets/2)

                                        if raw_delta_packets == 0 :                                                
                                            pbd_raw_adversary = raw_adversary
                                            pbd_scaled_adversary = self.adv_scaler.transform(pbd_raw_adversary)

                                    else:
                                        raw_delta_packets = 0
                                        total_packets_udp[port_id] -= iter_packets_udp
                                        total_packets_tcp[port_id] -= iter_packets_tcp

                                #Feasible update with binary search over delta for 'total packets sent' which is representative feauture
                                else:    
                                    zeroing = False                       
                                
                                    if delta_sign < 0:
                                        #if we have TCP and UDP types of connections then we need to spread tha packets between connections                                                
                                        if iter_conns_udp != 0 and iter_conns_tcp != 0:
                                            if raw_delta_packets < (iter_conns_udp+ iter_conns_tcp)*2:
                                                raw_delta_packets = iter_conns_udp*2 + iter_conns_tcp*2
                                                zeroing=True
                                            elif raw_delta_packets > iter_conns_udp*self.MAX_PACKET_UDP_CONN + iter_conns_tcp*self.MAX_PACKET_TCP_CONN:
                                                raw_delta_packets = iter_conns_udp*self.MAX_PACKET_UDP_CONN + iter_conns_tcp*self.MAX_PACKET_TCP_CONN

                                            min_tcp = iter_conns_tcp*2
                                            iter_packets_udp = random.randint(iter_conns_udp*2, iter_conns_udp*self.MAX_PACKET_UDP_CONN)
                                            if iter_packets_udp > raw_delta_packets-min_tcp:
                                                iter_packets_udp = raw_delta_packets-min_tcp
                                            iter_packets_tcp = raw_delta_packets - iter_packets_udp

                                        #if we only have tcp connections
                                        elif iter_conns_tcp !=0:

                                            iter_packets_udp = 0
                                            iter_packets_tcp = raw_delta_packets
                                        #if we only have udp connections
                                        elif iter_conns_udp != 0:

                                            iter_packets_tcp = 0
                                            iter_packets_udp = raw_delta_packets

                                        total_packets_udp[port_id] += iter_packets_udp
                                        total_packets_tcp[port_id] += iter_packets_tcp                                       

                                            #udpate total number of sent packets and adjust mathematical depedencies(min/msx packets sent per connection)
                                        pbd_raw_adversary[0, inds[f_id]], pbd_raw_adversary[0, inds[f_id + 1]], pbd_raw_adversary[0, inds[f_id+ 2]]  = self.update_from_total_up(pbd_raw_adversary, input_vector_raw,
                                                                                                    inds, raw_delta_packets * delta_sign,  f_id, total_conns_udp, total_conns_tcp, 
                                                                                                    total_packets_udp[port_id], total_packets_tcp[port_id])
                        
                                        pbd_scaled_adversary = self.adv_scaler.transform(pbd_raw_adversary)
                                        #update bytes and duration
                                        pbd_raw_adversary = update_bytes(self, pbd_scaled_adversary, pbd_raw_adversary, input_vector_raw, input_vector, inds, shape,                                                                
                                                        iter_conns_tcp, iter_conns_udp, total_conns_tcp, total_conns_udp, iter_packets_udp, iter_packets_tcp,  total_packets_udp[port_id], total_packets_tcp[port_id], d_max)

                                        if zeroing:
                                            raw_delta_packets=0
                                    elif  pbd_raw_adversary[0, inds[f_id]] != input_vector_raw[0, inds[f_id]] and delta_sign > 0:

                                        delta_to_increase = new_packets - input_vector_raw[0, inds[f_id]] 
                                        #if we have TCP and UDP types of connections then we need to spread tha packets between connections
                                        if iter_conns_tcp != 0 and iter_conns_udp != 0:

                                            tcp_min_packets = total_conns_tcp * 2
                                            tcp_max_packets = total_conns_tcp * self.MAX_PACKET_TCP_CONN
                                            udp_min_packets = total_conns_udp * 2
                                            udp_max_packets = total_conns_udp * self.MAX_PACKET_UDP_CONN
                                                                                                    
                                            lower_rand_udp = round(max(udp_min_packets,delta_to_increase - tcp_max_packets))
                                            upper_rand_udp = round(min(udp_max_packets,delta_to_increase - tcp_min_packets))

                                            iter_packets_udp = random.randint(lower_rand_udp, upper_rand_udp + 1)
                                            iter_packets_tcp = delta_to_increase - iter_packets_udp

                                        #if we have only tcp connections
                                        elif iter_conns_tcp != 0:
                                            iter_packets_udp = 0
                                            iter_packets_tcp = delta_to_increase

                                            #if we have only udp connections   
                                        elif iter_conns_udp != 0 :
                                            iter_packets_tcp = 0
                                            iter_packets_udp = delta_to_increase

                                        #update total number of sent packets and adjust mathematical dependencies(min/ max packets per connection)
                                        pbd_raw_adversary[0, inds[f_id]], pbd_raw_adversary[0, inds[f_id + 1]], pbd_raw_adversary[0, inds[f_id+ 2]]  = self.update_from_total_up(current_raw, input_vector_raw, inds, delta_to_increase * -1, 
                                                                                                                f_id, total_conns_udp, total_conns_tcp, total_packets_udp[port_id], total_packets_tcp[port_id])
                                                
                                        pbd_scaled_adversary = self.adv_scaler.transform(pbd_raw_adversary)
                                        #update bytes and duration
                                        pbd_raw_adversary = update_bytes(self, pbd_scaled_adversary, pbd_raw_adversary, input_vector_raw, input_vector, inds, shape,                                                                
                                                                iter_conns_tcp, iter_conns_udp, total_conns_tcp, total_conns_udp, iter_packets_udp, iter_packets_tcp,  total_packets_udp[port_id], total_packets_tcp[port_id], d_max)
                                    
                                    pbd_scaled_adversary = self.adv_scaler.transform(pbd_raw_adversary)
                                    distance_tmp = np.linalg.norm(pbd_scaled_adversary - input_vector)
                                    current_prob = sigmoid(self.model.model(pbd_scaled_adversary,training=False))
                                    #perform binary search if we are outside L2 norma ball pt resulting probability is > 0.5
                                    if distance_tmp > d_max or current_prob > 0.5:

                                        raw_delta_packets  = math.floor(raw_delta_packets/2)

                                        if delta_sign < 0:

                                            total_packets_udp[port_id] -= iter_packets_udp
                                            total_packets_tcp[port_id] -= iter_packets_tcp

                                        if raw_delta_packets == 0 :

                                            pbd_raw_adversary = raw_adversary
                                            pbd_scaled_adversary = self.adv_scaler.transform(pbd_raw_adversary)

                                    else:
                                        raw_delta_packets = 0

                                        if delta_sign > 0:

                                            total_packets_udp[port_id] = iter_packets_udp
                                            total_packets_tcp[port_id] = iter_packets_tcp

                    else:
                                
                        pbd_raw_adversary = raw_adversary
                        pbd_scaled_adversary = adversary

                    adversary = self.adv_scaler.transform(pbd_raw_adversary)
            after_pred = sigmoid(self.model.model(adversary, training=False).numpy())
            if after_pred < 0.5:
                pred = 0
            else:
                pred = 1
            if pred == self.target:
                return adversary
        return adversary
          
    def update_from_conn_up(self, adversary, inds, raw_delta, conn_id):
        conn_f =  adversary[0, inds[conn_id]]
        max_conn_f = self.max_features[inds[conn_id]]
        new_conn_f = conn_f

        new_conn_f = conn_f - raw_delta

        if new_conn_f > max_conn_f :
            new_conn_f = max_conn_f 

        return new_conn_f

    def update_from_conn_down(self, adversary, input_vector, inds, raw_delta, conn_id):

        conn_f = adversary[0, inds[conn_id]]
        new_conn_f = conn_f

        initial_conn_f = input_vector[0, inds[conn_id]]

        new_conn_f = conn_f - raw_delta

        if new_conn_f <  initial_conn_f:
            new_conn_f = initial_conn_f

        adversary[0, inds[conn_id]] = new_conn_f

        return new_conn_f

    

    

    def update_from_total_up(self, adversary, input_raw, inds, raw_delta, total_id, total_conns_udp, total_conns_tcp, total_packets_udp,  total_packets_tcp):

        #random.seed(500)
        
        total_f = adversary[0, inds[total_id]]
        max_total_f = self.max_features[inds[total_id]]
        new_total_f = total_f

        max_f = input_raw[0, inds[total_id + 2]]
        max_max_f = self.max_features[inds[total_id + 2]]
        new_max_f = max_f

        min_f = input_raw[0, inds[total_id + 1]]
        if min_f < 0:
            min_f = 0
        min_min_f = self.min_features[inds[total_id + 1]]
        new_min_f = min_f

        #update total - increase
  
        new_total_f = total_f - raw_delta

        ###########################################Min/Max

        #get total number of udp and tcp connections OVERALL
        total_conns_tcp = round(total_conns_tcp)
        total_conns_udp = round(total_conns_udp)


        #get total amount of feature added OVERALL
        total_added = new_total_f - input_raw[0, inds[total_id]]

        if total_id == self.PACKETS_ID:
            new_max_f, new_min_f = self.update_max_min_packets(total_added, total_conns_tcp, total_conns_udp, total_packets_tcp, total_packets_udp, new_max_f, new_min_f)
           
        if total_id == self.BYTES_ID:

            new_max_f, new_min_f = update_max_min_bytes(self,total_added, total_conns_tcp, total_conns_udp, total_packets_tcp, total_packets_udp,new_max_f, new_min_f, raw_delta)

        if total_id == self.DURATION_ID:
            new_max_f, new_min_f = update_max_min_duration(self,total_added, total_conns_tcp, total_conns_udp, total_packets_tcp, total_packets_udp, new_max_f, new_min_f, raw_delta)

        return new_total_f, new_min_f, new_max_f

    def update_from_total_down(self, adversary_raw, input_raw, inds, raw_delta, total_id, total_conns_udp, total_conns_tcp, total_packets_udp, total_packets_tcp):

        #random.seed(500)

        input_total = input_raw[0, inds[total_id]]
        total_f = adversary_raw[0, inds[total_id]]
        new_total_f = total_f

        max_f = input_raw[0, inds[total_id + 2]]
        max_max_f = self.max_features[inds[total_id + 2]]
        new_max_f = max_f

        min_f = input_raw[0, inds[total_id + 1]]
        if min_f < 0:
            min_f = 0
        min_min_f = self.min_features[inds[total_id + 1]]
        new_min_f = min_f

        #decrease total
        new_total_f = total_f - raw_delta

        if total_id == self.PACKETS_ID:
            # check if greater than initial + 20 bytes(min bytes per packet)
            if new_total_f < input_total + 2:
                new_total_f = input_total + 2

        elif total_id == self.BYTES_ID:
            # check if greater than initial + 20 bytes(min bytes per packet)
            if new_total_f < input_total + 20:
                new_total_f = input_total + 20

        elif total_id == self.DURATION_ID:
            # check if greater than initial + min duration per packet 
            if new_total_f < input_total + 0.0001:
                new_total_f = input_total + 0.0001

        ###########################################Min/Max
        total_conns_tcp = round(total_conns_tcp)
        total_conns_udp = round(total_conns_udp)

        #get total amount of feature added
        total_added = new_total_f - input_raw[0, inds[total_id]]

        if total_id == self.PACKETS_ID:
            new_max_f, new_min_f = self.update_max_min_packets(total_added, total_conns_tcp, total_conns_udp, total_packets_tcp, total_packets_udp, new_max_f, new_min_f)
       
        if total_id == self.BYTES_ID:
            new_max_f, new_min_f = update_max_min_bytes(self,total_added, total_conns_tcp, total_conns_udp, total_packets_tcp, total_packets_udp, new_max_f, new_min_f, raw_delta)

        if total_id == self.DURATION_ID:
            new_max_f, new_min_f = update_max_min_duration(self,total_added, total_conns_tcp, total_conns_udp, total_packets_tcp, total_packets_udp,new_max_f, new_min_f, raw_delta)

        return new_total_f, new_min_f, new_max_f

    def update_max_min_packets(self, total_added,  total_conns_tcp, total_conns_udp, total_packets_tcp, total_packets_udp, new_max_f, new_min_f):

        total_added = round(total_added)
        
        if total_conns_tcp != 0:

            if total_conns_tcp != 1:
                min_packets_per_connection = math.floor(total_packets_tcp/total_conns_tcp)
                max_packets_per_connection = total_packets_tcp - (total_conns_tcp - 1) * min_packets_per_connection

            else:
                min_packets_per_connection = total_packets_tcp
                max_packets_per_connection = total_packets_tcp

            if max_packets_per_connection > new_max_f:
                new_max_f = max_packets_per_connection

            elif min_packets_per_connection < new_min_f:
                new_min_f = min_packets_per_connection
        
        if total_conns_udp != 0:

            if total_conns_udp != 1:
                min_packets_per_connection = math.floor(total_packets_udp/total_conns_udp)
                max_packets_per_connection = total_packets_udp - (total_conns_udp - 1) * min_packets_per_connection
        
            else:
                min_packets_per_connection = total_packets_udp
                max_packets_per_connection = total_packets_udp 
        
            if max_packets_per_connection > new_max_f:
                new_max_f = max_packets_per_connection
        
            elif min_packets_per_connection < new_min_f:
                new_min_f = min_packets_per_connection

        return new_max_f, new_min_f
   
    def run_attack(self, sample, label):
        adv = self.attack_round( input_vector=sample, label=label)
        return adv
