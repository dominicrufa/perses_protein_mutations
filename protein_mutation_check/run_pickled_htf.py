#!/usr/bin/env python
from __future__ import absolute_import 
from perses.annihilation.lambda_protocol import LambdaProtocol
from openmmtools.multistate import MultiStateReporter
from perses.samplers.multistate import HybridRepexSampler
import pickle
import simtk.unit as unit
import sys
from openmmtools import mcmc, utils

def create_hss(pkl, suffix, selection, checkpoint_interval, n_states):
    with open(pkl, 'rb') as f:
        htf = pickle.load(f)
    lambda_protocol = LambdaProtocol(functions='default')
    reporter_file = pkl[:-3]+suffix+'.nc'
    reporter = MultiStateReporter(reporter_file, analysis_particle_indices = htf.hybrid_topology.select(selection), checkpoint_interval = checkpoint_interval)
    hss = HybridRepexSampler(mcmc_moves=mcmc.LangevinSplittingDynamicsMove(timestep= 4.0 * unit.femtoseconds,
                                                                                  collision_rate=5.0 / unit.picosecond,
                                                                                  n_steps=250,
                                                                                  reassign_velocities=False,
                                                                                  n_restart_attempts=20,
                                                                                  splitting="V R R R O R R R V",
                                                                                  constraint_tolerance=1e-06),
                                                                                  hybrid_factory=htf, online_analysis_interval=10)
    hss.setup(n_states=n_states, temperature=300*unit.kelvin, storage_file = reporter, lambda_protocol = lambda_protocol, endstates=False)
    return hss, reporter

def run_sim(pkl, suffix = 't2', selection = 'protein', checkpoint_interval = 10, n_states = 11, num_extensions = 1000):
    hss, reporter = create_hss(pkl, suffix, selection, checkpoint_interval, n_states)
    hss.extend(num_extensions)
            
run_sim(str(sys.argv[1]))

