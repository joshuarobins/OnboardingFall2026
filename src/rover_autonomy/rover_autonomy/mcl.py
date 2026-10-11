"""Monte Carlo localization: a particle filter over the rover's pose (x, y, yaw) in the map.

Every update moves the particles by the change in odometry (motion model), weights them by how well
the scan fits the map from each particle's pose (sensor model), and resamples when the weights get
uneven. You write the four functions above ParticleFilter; ParticleFilter ties them together.
"""
import math

import numpy as np

from rover_autonomy.geometry import wrap

NUM_PARTICLES = 1000
ALPHAS = (0.02, 0.02, 0.02, 0.02)  # motion noise: turn from turning, turn from driving, drive from driving, drive from turning
SIGMA_HIT = 0.15                   # m: how far a beam endpoint lands from the obstacle it hit
Z_HIT, Z_RAND = 0.9, 0.1           # odds a beam hit a mapped obstacle vs anything else (like an unmapped rock)
MAX_RANGE = 12.0                   # m: the lidar's range
BEAM_STEP = 6                      # use every 6th beam, so 60 of the 360
MIN_MOVE, MIN_TURN = 0.02, 0.02    # m, rad: skip the update while the rover is parked


def sample_motion_model(particles, odom_prev, odom_now, rng):
    """Move every particle by the odometry change, with noise. Returns a new (N, 3) array.

    particles: (N, 3) array of (x, y, yaw) in the map frame.
    odom_prev, odom_now: odometry poses (x, y, yaw) in the odom frame, before and after.
    rng: a numpy Generator; draw noise with rng.normal(0.0, std, N).

    Express the change as rot1 (turn to face where the rover went), trans (drive straight) and rot2
    (turn to the final heading). Give every particle its own noisy copy, with standard deviations
        rot1:  sqrt(a1 * rot1**2 + a2 * trans**2)
        trans: sqrt(a3 * trans**2 + a4 * (rot1**2 + rot2**2))
        rot2:  sqrt(a1 * rot2**2 + a2 * trans**2)
    and replay it from the particle's own heading. Wrap the yaws. Reference: Probabilistic Robotics,
    table 5.6. Watch out when trans is nearly zero (turning in place) and when driving backward.
    """

    distance = np.hypot(odom_now[0] - odom_prev[0], odom_now[1] - odom_prev[1])

    if distance > MIN_MOVE:
        initial_yaw = wrap(math.atan2((odom_now[1]-odom_prev[1]), (odom_now[0]-odom_prev[0])) - odom_prev[2])
        if abs(initial_yaw) > math.pi/2:
            distance *= -1
            initial_yaw = wrap(initial_yaw + math.pi)
    else:
        initial_yaw = 0
        distance = 0
        
    final_yaw = wrap(wrap((odom_now[2]-odom_prev[2])) - initial_yaw)

    (a1, a2, a3, a4) = ALPHAS

    rot1_sd = np.sqrt(a1* initial_yaw**2 + a2 * distance**2)
    trans_sd = np.sqrt(a3 * distance**2 + a4 * (initial_yaw**2 + final_yaw**2))
    rot2_sd = np.sqrt(a1 * final_yaw**2 + a2 * distance**2)

    rot1_noise = rng.normal(0.0, rot1_sd, len(particles))
    trans_noise = rng.normal(0.0, trans_sd, len(particles))
    rot2_noise = rng.normal(0.0, rot2_sd, len(particles))

    noisy_rot1 = initial_yaw + rot1_noise
    noisy_rot2 = final_yaw + rot2_noise
    noisy_trans = distance + trans_noise

    translated_particles = particles.copy()
    translated_particles[:, 2] += noisy_rot1
    translated_particles[:, 0] += noisy_trans * np.cos(translated_particles[:, 2])
    translated_particles[:, 1] += noisy_trans * np.sin(translated_particles[:, 2])
    translated_particles[:, 2] += noisy_rot2
    translated_particles[:, 2] = wrap(translated_particles[:, 2])

    return translated_particles

def scan_log_likelihood(particles, ranges, angles, grid, laser_offset):
    """How well the scan fits the map from each particle. Returns an (N,) array of log-likelihoods.

    ranges, angles: (B,) arrays of beam lengths and their angles in the laser frame. All are valid hits.
    grid: the map. grid.distance_at(x, y, outside=...) gives the distance to the nearest mapped
        obstacle at world points, for arrays of any shape.
    laser_offset: (x, y, yaw) of the laser in the rover's frame.

    For every particle and beam, find where the beam ends in the map, look up the distance d from
    that endpoint to the nearest obstacle, and score it as
        Z_HIT * N(d; 0, SIGMA_HIT) + Z_RAND / MAX_RANGE
    where N is the Gaussian density. Sum the logs of the scores over the beams. Endpoints off the map
    should get the lowest score, not the highest. Try to do it without Python loops.
    """
    raise NotImplementedError('scan_log_likelihood')


def low_variance_resample(particles, weights, rng):
    """Draw N particles with probability proportional to their weights. Returns a new (N, 3) array.

    Use low-variance (systematic) resampling: one random number r in [0, 1/N), then pointers at
    r, r + 1/N, r + 2/N, ... into the cumulative sum of the weights. Probabilistic Robotics, table 4.4.
    """
    raise NotImplementedError('low_variance_resample')


def estimate_pose(particles, weights):
    """The weighted mean pose, as a (3,) array. Average the yaws as angles: the mean of 179 degrees
    and -179 degrees is 180, not 0."""
    raise NotImplementedError('estimate_pose')


class ParticleFilter:
    """Ties the four functions together. Call step() on every scan."""

    def __init__(self, grid, laser_offset, seed=None):
        self.grid = grid
        self.laser_offset = laser_offset
        self.rng = np.random.default_rng(seed)
        self.particles = None
        self.weights = None
        self.odom = None

    def initialize(self, pose, std=(0.25, 0.25, 0.1)):
        """Spread the particles around a pose."""
        self.particles = np.asarray(pose, dtype=float) + self.rng.normal(0.0, std, (NUM_PARTICLES, 3))
        self.particles[:, 2] = wrap(self.particles[:, 2])
        self.weights = np.full(NUM_PARTICLES, 1.0 / NUM_PARTICLES)
        self.odom = None

    def step(self, odom, ranges, angles):
        """Update with the odometry pose at the scan's time and the scan.

        Returns (estimate, odom_used): the pose estimate in the map, and the odometry pose it goes with.
        """
        if self.odom is not None:
            moved = math.hypot(odom[0] - self.odom[0], odom[1] - self.odom[1])
            turned = abs(float(wrap(odom[2] - self.odom[2])))
            if moved < MIN_MOVE and turned < MIN_TURN:
                return estimate_pose(self.particles, self.weights), self.odom
            self.particles = sample_motion_model(self.particles, self.odom, odom, self.rng)
        self.odom = np.asarray(odom, dtype=float)

        ranges = np.asarray(ranges, dtype=float)[::BEAM_STEP]
        angles = np.asarray(angles, dtype=float)[::BEAM_STEP]
        hit = np.isfinite(ranges) & (ranges < MAX_RANGE)
        log_w = np.log(self.weights + 1e-300) + scan_log_likelihood(
            self.particles, ranges[hit], angles[hit], self.grid, self.laser_offset)
        w = np.exp(log_w - log_w.max())
        self.weights = w / w.sum()

        if 1.0 / np.sum(self.weights ** 2) < NUM_PARTICLES / 2:
            self.particles = low_variance_resample(self.particles, self.weights, self.rng)
            self.weights = np.full(NUM_PARTICLES, 1.0 / NUM_PARTICLES)
        return estimate_pose(self.particles, self.weights), self.odom
