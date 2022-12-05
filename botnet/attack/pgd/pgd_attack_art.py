import numpy as np
import tensorflow as tf
from art.attacks.evasion import ProjectedGradientDescent as PGD
from art.estimators.classification.tensorflow import TensorFlowV2Classifier as kc

#from tensorflow.random import set_seed
#set_seed(2)

#from numpy.random import seed
#seed(2)


class PgdRandomRestart():
    def __init__(self, model, eps, alpha, num_iter, restarts, scaler, mins, maxs, mask_idx, eq_min_max):
        """
        :param model: instance of tf.keras.Model that is used to generate adversarial examples
        :param eps: float number - maximum perturbation size for adversarial attack
        :param alpha: float number - step size in adversarial attack
        :param num_iter: integer - number of iterations of pgd during one restart iteration
        :param restarts: integer - number of restarts
        """
        #self.model =  MDModel(model_path) 
        self.name = "PGD With Random Restarts"
        self.specifics = "PGD With Random Restarts - " \
                         f"eps: {eps} - alpha: {alpha} - " \
                         f"num_iter: {num_iter} - restarts: {restarts}"
        self.alpha = alpha
        self.num_iter = num_iter
        self.restarts = restarts
        self.eps = eps
        self.model = model
        self.scaler = scaler
        clip_min = mins
        clip_max = maxs
        self.clip_min = self.scaler.transform(np.array(clip_min).reshape(1, -1))
        self.clip_max = self.scaler.transform(np.array(clip_max).reshape(1, -1))
        self.clip_max[0][eq_min_max] = self.clip_max[0][eq_min_max] + 0.000000001
        self.mask_idx = mask_idx
        
    def run_attack (self, clean_samples, true_labels):
        n = clean_samples.shape[0]
        mask_feat = np.zeros((clean_samples.shape[1],))
        mask_feat[self.mask_idx]=1

        # ----- Attack
    
        true_labels = np.squeeze(true_labels)
        target = 1 - true_labels
        #target = to_categorical(target, num_classes=2)
        kc_classifier = kc(self.model, clip_values=(self.clip_min, self.clip_max), nb_classes=2, input_shape=(756),  loss_object=tf.keras.losses.BinaryCrossentropy())
        pgd = PGD(kc_classifier)
        pgd.set_params( eps=self.eps,  verbose=False, max_iter=self.num_iter, num_random_init=self.restarts, norm=2, eps_step=self.alpha, targeted=True)
        attacks = pgd.generate(x=clean_samples, 
                                y=target,
                                mask=mask_feat)

        return attacks




if __name__ == "__main__":
    run()