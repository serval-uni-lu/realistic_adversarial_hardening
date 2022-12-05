import sys
sys.path.append('..')

import numpy as np
from attack.fence.neris_attack_tf2_modified import Neris_attack
from attack.pgd.pgd_attack_art import PgdRandomRestart

def generate_adversarial_batch_fence(model, total, samples, labels,  distance, iterations, scaler, mins, maxs, model_path):

	while True:
		model.save(model_path)

		attack_generator = Neris_attack(model_path = model_path, iterations=iterations, distance=distance, scaler=scaler, mins=mins, maxs=maxs)

		# initialize our perturbed data and labels
		perturbSamples = []
		perturbLabels = []
		idxs = np.random.choice(range(0, len(samples)), size=total, replace=False)
		for i in idxs:
			sample = samples[i]
			sample = np.expand_dims(sample, axis=0)
			label = labels[i]
			# generate an adversarial sample
			adversary = attack_generator.run_attack(sample,label)
			perturbSamples.append(adversary)
			perturbLabels.append(label)

		perturbSamples = np.squeeze(np.array(perturbSamples))

		yield (np.array(perturbSamples), np.array(perturbLabels))


def generate_adversarial_batch_pgd(model, total, samples, labels, distance, iterations, scaler, mins, maxs):

	attack_generator = PgdRandomRestart(model, eps = distance, alpha = 1, num_iter = iterations, restarts = 5, scaler=scaler, mins=mins, maxs=maxs )

	while True:
		idxs = np.random.choice(range(0, len(samples)), size=total, replace=False)
		batch_samples = samples[idxs] 
		perturbLabels = labels[idxs]
		perturbLabels = np.expand_dims(perturbLabels, axis=1)

		perturbSamples = attack_generator.run_attack(batch_samples,perturbLabels)

		yield (np.array(perturbSamples), np.array(perturbLabels))