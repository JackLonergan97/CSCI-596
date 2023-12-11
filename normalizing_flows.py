# Importing packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import h5py
import os
from scipy import stats

import warnings
warnings.filterwarnings('ignore')

# Reading in and normalizing a single realization of a Galacticus subhalo population
f = h5py.File('mini_darkMatterOnlySubHalos.hdf5', 'r')
mergerTreeBuildMassesGroup = f['Parameters/mergerTreeBuildMasses']
massResolutionGroup = f['Parameters/mergerTreeMassResolution']
massTree = mergerTreeBuildMassesGroup.attrs['massTree'][0]
countTree = mergerTreeBuildMassesGroup.attrs['treeCount'][0]
massResolution = massResolutionGroup.attrs['massResolution']
weight = f['Outputs/Output1/nodeData/nodeSubsamplingWeight']
treeIndex = f['Outputs/Output1/nodeData/mergerTreeIndex']
isCentral = f['Outputs/Output1/nodeData/nodeIsIsolated']
massInfall = f['Outputs/Output1/nodeData/basicMass']
massBound = f['Outputs/Output1/nodeData/satelliteBoundMass']
concentration = f['Outputs/Output1/nodeData/concentration']

redshiftLastIsolated = f['Outputs/Output1/nodeData/redshiftLastIsolated']
positionOrbitalX = f['Outputs/Output1/nodeData/positionOrbitalX']
positionOrbitalY = f['Outputs/Output1/nodeData/positionOrbitalY']
positionOrbitalZ = f['Outputs/Output1/nodeData/positionOrbitalZ']
satelliteTidalHeating = f['Outputs/Output1/nodeData/satelliteTidalHeatingNormalized']
radiusVirial = f['Outputs/Output1/nodeData/darkMatterOnlyRadiusVirial']
velocityVirial = f['Outputs/Output1/nodeData/darkMatterOnlyVelocityVirial']
subhalos = (isCentral[:] == 0) & (massInfall[:] > 2.0*massResolution)
centrals = (isCentral[:] == 1)
countSubhalos = np.zeros(countTree)
for i in range(countTree):
    selectTree = (isCentral[:] == 0) & (treeIndex[:] == i+1)
    countSubhalos[i] = np.sum(weight[selectTree])
countSubhalosMean = np.mean(countSubhalos)
countSubhalosSquaredMean = np.mean(countSubhalos**2)
variance = countSubhalosSquaredMean-countSubhalosMean**2
standardDeviation = np.sqrt(variance)
standardDeviationFractional = standardDeviation/countSubhalosMean
countSubhalosMin = np.min(countSubhalos)
countSubhalosMax = np.max(countSubhalos)
massHost = massInfall[centrals][0]
radiusVirialHost = radiusVirial[centrals][0]
velocityVirialHost = velocityVirial[centrals][0]
massInfallNormalized = np.log10(massInfall[subhalos]/massHost)
massBoundNormalized = np.log10(massBound[subhalos]/massInfall[subhalos])
concentrationNormalized = concentration[subhalos]
redshiftLastIsolatedNormalized = redshiftLastIsolated[subhalos]
radiusOrbitalNormalized = np.log10(np.sqrt(+positionOrbitalX[subhalos]**2+positionOrbitalY[subhalos]**2+positionOrbitalZ[subhalos]**2)/radiusVirialHost)
satelliteTidalHeatingNormalized = np.log10(1.0e-6+satelliteTidalHeating[subhalos]/velocityVirial[subhalos]**2*radiusVirial[subhalos]**2)

# defining the 6 variables of the parameter space to a single data object
data=np.array(
    list(
        zip(
            massInfallNormalized,
            concentrationNormalized,
            massBoundNormalized,
            redshiftLastIsolatedNormalized,
            radiusOrbitalNormalized,
            satelliteTidalHeatingNormalized
        )
    )
)

# Defining a transformation (and its inverse) normalizing the data to a 6D hypercube to make it easier for the 
# normalizing flows algorithm to implement
def norm_transform(data, min_val, max_val):
    data_min = np.nanmin(data, axis = 0)
    data_max = np.nanmax(data, axis = 0)
    sigma_data = (data - data_min)/(data_max - data_min)
    return data_min, data_max, sigma_data*(max_val - min_val) + min_val

def norm_transform_inv(norm_data, data_min, data_max, min_val, max_val):
    data_min = np.nanmin(data, axis = 0)
    data_max = np.nanmax(data, axis = 0)
    sigma_data = (norm_data - min_val)/(max_val - min_val)
    return sigma_data*(data_max - data_min) + data_min

# Creating a custom layer with keras API.
output_dim = 256
reg = 0.01


def Coupling(input_shape):
    input = keras.layers.Input(shape=input_shape)

    t_layer_1 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(input)
    t_layer_2 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(t_layer_1)
    t_layer_3 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(t_layer_2)
    t_layer_4 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(t_layer_3)
    t_layer_5 = keras.layers.Dense(
        input_shape, activation="tanh", kernel_regularizer=regularizers.l2(reg)
    )(t_layer_4)

    s_layer_1 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(input)
    s_layer_2 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(s_layer_1)
    s_layer_3 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(s_layer_2)
    s_layer_4 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(s_layer_3)
    s_layer_5 = keras.layers.Dense(
        input_shape, activation="tanh", kernel_regularizer=regularizers.l2(reg)
    )(s_layer_4)

    return keras.Model(inputs=input, outputs=[s_layer_5, t_layer_5])

# defining class which sets up the framework of the algorithm
class RealNVP(keras.Model):
    def __init__(self, num_coupling_layers):
        super(RealNVP, self).__init__()

        self.num_coupling_layers = num_coupling_layers

        # Distribution of the latent space.
        self.distribution = tfp.distributions.MultivariateNormalDiag(
            loc=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], scale_diag=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        )
        self.masks = np.array(
            [[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]] * (num_coupling_layers // 2), dtype="float32"
        )
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.layers_list = [Coupling(6) for i in range(num_coupling_layers)]

    @property
    def metrics(self):
        """List of the model's metrics.
        We make sure the loss tracker is listed as part of `model.metrics`
        so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
        at the start of each epoch and at the start of an `evaluate()` call.
        """
        return [self.loss_tracker]

    def call(self, x, training=True):
        log_det_inv = 0
        direction = 1
        if training:
            direction = -1
        for i in range(self.num_coupling_layers)[::direction]:
            x_masked = x * self.masks[i]
            reversed_mask = 1 - self.masks[i]
            s, t = self.layers_list[i](x_masked)
            s *= reversed_mask
            t *= reversed_mask
            gate = (direction - 1) / 2
            x = (
                reversed_mask
                * (x * tf.exp(direction * s) + direction * t * tf.exp(gate * s))
                + x_masked
            )
            log_det_inv += gate * tf.reduce_sum(s, [1])

        return x, log_det_inv

    # Log likelihood of the normal distribution plus the log determinant of the jacobian.

    def log_loss(self, data):
        # Extract the actual data here as "x", and the final weight column as "w".
        x = data[:,0:-1]
        w = data[:,-1]
        m = data[:,0]
        y, logdet = self(x)
        # Suppose the weight of the subhalo is "N". This means that this subhalo actually represents N such subhalos.
        # Treating these as independent contributions to the likelihood, we should multiply the probability, p, of this point
        # together N times, i.e. p^N. Since we compute a log-likelihood this corresponds to multiplying the likelihood by the weight.
        log_likelihood = (self.distribution.log_prob(y) + logdet)*w
        return -tf.reduce_mean(log_likelihood)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self.log_loss(data)

        g = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(g, self.trainable_variables))
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self.log_loss(data)
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}

model = RealNVP(num_coupling_layers=12)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001))

# executing the normalizing flows algorithm on the 6D hypercube data defined earlier in the script
history = model.fit(
    augmented_normalized_data, batch_size=256, epochs=100, verbose=2, validation_split=0.2
)

# saving the weights (the output) of the algorithm
model.save_weights('../data/mini_emulatorModel')

emulator = RealNVP(num_coupling_layers=12)
emulator.load_weights('../data/mini_emulatorModel')

# From data to latent space.
z, _ = emulator(normalized_data)

# From latent space to data.
samples = emulator.distribution.sample(3000)
x, _ = emulator.predict(samples)
xt = norm_transform_inv(x, np.nanmin(x, axis = 0), np.nanmax(x, axis = 0), -1, 1)
clip = (xt[:,0] > np.log10(2.0*massResolution/massTree)) & (xt[:,2] <= 0.0) & (xt[:,2] > -xt[:,0]+np.log10(massResolution/massTree)) & (xt[:,3] >= 0.0)

# Generate a weighted subsample of the original data.
w = weight[subhalos]
i = np.arange(0, w.size, 1, dtype=int)
subsample = np.random.choice(i, size=len(xt[clip]), replace=True, p=w/np.sum(w))

# create a mask that eliminates the tidal heating outlier data points from Galacticus
tidal_clip = (data[subsample, 5] > -6)

# Compute and compare the ratio of low-to-high-mass subhalos in the original data and in the emulated data.
# For the original data we weight by the subsampling weight. If this was included correctly in the training
# then the emulated data should have effectively learned these weights and produce a ratio similar to that in
# the original data.
s6 = data[:,0] > -6.0
s4 = data[:,0] > -4.0
ratioOriginal = np.sum(data[s6,0]*w[s6])/np.sum(data[s4,0]*w[s4])
print("Ratio of low-to-high-mass subhalos in original data (weighted): "+str(ratioOriginal))

ratioEmulator = np.sum(xt[clip,0] > -6.0)/np.sum(xt[clip,0] > -4.0)
print("Ratio of low-to-high-mass subhalos in emulated data: "+str(ratioEmulator))

# Setting up code for plots
# Testing now to create Density plots
from scipy.stats import gaussian_kde

concentration_density_galacticus = np.vstack([data[subsample, 0][tidal_clip], data[subsample, 1][tidal_clip]])
z1_galacticus = gaussian_kde(concentration_density_galacticus)(concentration_density_galacticus)
concentration_density_generated = np.vstack([xt[clip, 0][tidal_clip], xt[clip, 1][tidal_clip]])
z1_generated = gaussian_kde(concentration_density_generated)(concentration_density_generated)

mass_bound_density_galacticus = np.vstack([data[subsample, 0][tidal_clip], data[subsample, 2][tidal_clip]])
z2_galacticus = gaussian_kde(mass_bound_density_galacticus)(mass_bound_density_galacticus)
mass_bound_density_generated = np.vstack([xt[clip, 0][tidal_clip], xt[clip, 2][tidal_clip]])
z2_generated = gaussian_kde(mass_bound_density_generated)(mass_bound_density_generated)

mass_bound_density_galacticus = np.vstack([data[subsample, 0][tidal_clip], data[subsample, 2][tidal_clip]])
z2_galacticus = gaussian_kde(mass_bound_density_galacticus)(mass_bound_density_galacticus)
mass_bound_density_generated = np.vstack([xt[clip, 0][tidal_clip], xt[clip, 2][tidal_clip]])
z2_generated = gaussian_kde(mass_bound_density_generated)(mass_bound_density_generated)

redshift_infall_density_galacticus = np.vstack([data[subsample, 0][tidal_clip], data[subsample, 3][tidal_clip]])
z3_galacticus = gaussian_kde(redshift_infall_density_galacticus)(redshift_infall_density_galacticus)
redshift_infall_density_generated = np.vstack([xt[clip, 0][tidal_clip], xt[clip, 3][tidal_clip]])
z3_generated = gaussian_kde(redshift_infall_density_generated)(redshift_infall_density_generated)

orbital_radius_density_galacticus = np.vstack([data[subsample, 0][tidal_clip], data[subsample, 4][tidal_clip]])
z4_galacticus = gaussian_kde(orbital_radius_density_galacticus)(orbital_radius_density_galacticus)
orbital_radius_density_generated = np.vstack([xt[clip, 0][tidal_clip], xt[clip, 4][tidal_clip]])
z4_generated = gaussian_kde(orbital_radius_density_generated)(orbital_radius_density_generated)

tidal_heating_density_galacticus = np.vstack([data[subsample, 0][tidal_clip], data[subsample, 5][tidal_clip]])
z5_galacticus = gaussian_kde(tidal_heating_density_galacticus)(tidal_heating_density_galacticus)
tidal_heating_density_generated = np.vstack([xt[clip, 0][tidal_clip], xt[clip, 5][tidal_clip]])
z5_generated = gaussian_kde(tidal_heating_density_generated)(tidal_heating_density_generated)

f, axes = plt.subplots(5, 2)
f.set_size_inches(15, 18)

axes[0, 0].scatter(data[subsample, 0][tidal_clip], data[subsample, 1][tidal_clip], c = z1_galacticus, s=9)
axes[0, 0].set(title="Galacticus", xlabel="Mass infall", ylabel="concentration")
axes[0, 0].set_xlim([-6, 0])
axes[0, 0].set_ylim([0, 23])
axes[0, 1].scatter(xt[clip, 0][tidal_clip], xt[clip, 1][tidal_clip], c = z1_generated, s=9)
axes[0, 1].set(title="Generated", xlabel="Mass infall", ylabel="concentration")
axes[0, 1].set_xlim([-6, 0])
axes[0, 1].set_ylim([0, 23])
axes[1, 0].scatter(data[subsample, 0][tidal_clip], data[subsample, 2][tidal_clip], c = z2_galacticus, s=9)
axes[1, 0].set(title="Galacticus", xlabel="Mass infall", ylabel="Mass bound")
axes[1, 0].set_xlim([-6, 0])
axes[1, 0].set_ylim([-5.0, 0.2])
axes[1, 1].scatter(xt[clip, 0][tidal_clip], xt[clip, 2][tidal_clip], c = z2_generated, s=9)
axes[1, 1].set(title="Generated", xlabel="Mass infall", ylabel="Mass bound")
axes[1, 1].set_xlim([-6, 0])
axes[1, 1].set_ylim([-5.0, 0.2])
axes[2, 0].scatter(data[subsample, 0][tidal_clip], data[subsample, 3][tidal_clip], c = z3_galacticus, s=9)
axes[2, 0].set(title="Galacticus", xlabel="Mass infall", ylabel="Redshift infall")
axes[2, 0].set_xlim([-6, 0])
axes[2, 0].set_ylim([-0.2, 6.0])
axes[2, 1].scatter(xt[clip, 0][tidal_clip], xt[clip, 3][tidal_clip], c = z3_generated, s=9)
axes[2, 1].set(title="Generated", xlabel="Mass infall", ylabel="Redshift infall")
axes[2, 1].set_xlim([-6, 0])
axes[2, 1].set_ylim([-0.2, 6.0])
axes[3, 0].scatter(data[subsample, 0][tidal_clip], data[subsample, 4][tidal_clip], c = z4_galacticus, s=9)
axes[3, 0].set(title="Galacticus", xlabel="Mass infall", ylabel="Orbital radius")
axes[3, 0].set_xlim([-6, 0])
axes[3, 0].set_ylim([-2.0, 1.0])
axes[3, 1].scatter(xt[clip, 0][tidal_clip], xt[clip, 4][tidal_clip], c = z4_generated, s=9)
axes[3, 1].set(title="Generated", xlabel="Mass infall", ylabel="Orbital radius")
axes[3, 1].set_xlim([-6, 0])
axes[3, 1].set_ylim([-2.0, 1.0])
axes[4, 0].scatter(data[subsample, 0][tidal_clip], data[subsample, 5][tidal_clip], c = z5_galacticus, s=9)
axes[4, 0].set(title="Galacticus", xlabel="Mass infall", ylabel="Tidal heating")
axes[4, 0].set_xlim([-6, 0])
axes[4, 0].set_ylim([-3.0, 5.0])
axes[4, 1].scatter(xt[clip, 0][tidal_clip], xt[clip, 5][tidal_clip], c = z5_generated, s=9)
axes[4, 1].set(title="Generated", xlabel="Mass infall", ylabel="Tidal heating")
axes[4, 1].set_xlim([-6, 0])
axes[4, 1].set_ylim([-3.0, 5.0])
