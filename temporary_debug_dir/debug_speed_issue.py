#!/usr/bin/env python
# coding: utf-8

# here, i i will attempt to find the bug that is preventing certain protein mutations from running quickly...

# In[1]:


import simtk.openmm as openmm
from openmmtools.constants import kB
import simtk.unit as unit
temperature = 300 * unit.kelvin
kT = kB * temperature
beta = 1.0/kT
from perses.rjmc.topology_proposal import TopologyProposal, NetworkXMolecule
from openmmtools.states import ThermodynamicState, SamplerState, CompoundThermodynamicState
from openmmtools import mcmc, utils
import openmmtools.cache as cache
from perses.dispersed.utils import configure_platform
#cache.global_context_cache.platform = configure_platform(utils.get_fastest_platform().getName())
from perses.annihilation.lambda_protocol import LambdaProtocol
from perses.annihilation.lambda_protocol import RelativeAlchemicalState, LambdaProtocol
import openmmtools.integrators as integrators
from protein_test_cycle import *


# In[2]:


def create_hss(reporter_name, hybrid_factory, selection_string ='all', checkpoint_interval = 1, n_states = 13):
    lambda_protocol = LambdaProtocol(functions='default')
    reporter = MultiStateReporter(reporter_name, analysis_particle_indices = hybrid_factory.hybrid_topology.select(selection_string), checkpoint_interval = checkpoint_interval)
    hss = HybridRepexSampler(mcmc_moves=mcmc.LangevinSplittingDynamicsMove(timestep= 4.0 * unit.femtoseconds,
                                                                                 collision_rate=5.0 / unit.picosecond,
                                                                                 n_steps=250,
                                                                                 reassign_velocities=False,
                                                                                 n_restart_attempts=20,
                                                                                 splitting="V R R R O R R R V",
                                                                                 constraint_tolerance=1e-06),
                                                                                 hybrid_factory=hybrid_factory, online_analysis_interval=10)
    hss.setup(n_states=n_states, temperature=300*unit.kelvin,storage_file=reporter,lambda_protocol=lambda_protocol,endstates=False)
    return hss, reporter


# In[3]:


def deserialize(filename):
    import pickle
    with open(filename, 'rb') as f:
        htf = pickle.load(f)
    return htf


# In[8]:


fast_htf = deserialize('ALA_SER.vac.pkl')
slow_htf = deserialize('ALA_ILE.vac.pkl') 
fast_hss, _ = create_hss(f"fast_debug.nc", fast_htf)
slow_hss, _ = create_hss(f"slow_debug.nc", slow_htf)


# In[9]:


fast_hss.extend(5)
print(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
slow_hss.extend(5)


# as you can see above, even in vacuum, there is a big difference in the replica propagation time between the fast and slow systems, even in vacuum

# In[ ]:


# fast_lambda_alchemical_state = RelativeAlchemicalState.from_system(fast_htf.hybrid_system)
# fast_lambda_alchemical_state.set_alchemical_parameters(0.0, LambdaProtocol(functions = 'default'))

# slow_lambda_alchemical_state = RelativeAlchemicalState.from_system(slow_htf.hybrid_system)
# slow_lambda_alchemical_state.set_alchemical_parameters(0.0, LambdaProtocol(functions = 'default'))

# fast_thermodynamic_state = CompoundThermodynamicState(ThermodynamicState(fast_htf.hybrid_system, temperature = temperature),composable_states = [fast_lambda_alchemical_state])
# slow_thermodynamic_state = CompoundThermodynamicState(ThermodynamicState(slow_htf.hybrid_system, temperature = temperature),composable_states = [slow_lambda_alchemical_state])

# fast_sampler_state = SamplerState(positions = fast_htf._hybrid_positions, box_vectors=fast_htf.hybrid_system.getDefaultPeriodicBoxVectors())
# slow_sampler_state = SamplerState(positions = slow_htf._hybrid_positions, box_vectors=slow_htf.hybrid_system.getDefaultPeriodicBoxVectors())

# integrator_1 = integrators.LangevinIntegrator(temperature = temperature,
#                                                  timestep = 4.0* unit.femtoseconds,
#                                                  splitting = 'V R O R V',
#                                                  measure_shadow_work = False,
#                                                  measure_heat = False,
#                                                  constraint_tolerance = 1e-6,
#                                                  collision_rate = 5.0 / unit.picoseconds)

# integrator_2 = integrators.LangevinIntegrator(temperature = temperature,
#                                                  timestep = 4.0* unit.femtoseconds,
#                                                  splitting = 'V R O R V',
#                                                  measure_shadow_work = False,
#                                                  measure_heat = False,
#                                                  constraint_tolerance = 1e-6,
#                                                  collision_rate = 5.0 / unit.picoseconds)


# fast_context, fast_integrator = cache.global_context_cache.get_context(fast_thermodynamic_state, integrator_1)
# slow_context, slow_integrator = cache.global_context_cache.get_context(slow_thermodynamic_state, integrator_2)

# fast_sampler_state.apply_to_context(fast_context)
# slow_sampler_state.apply_to_context(slow_context)


# perhaps if we remove the constraints, it will run faster?
# 

# In[ ]:




