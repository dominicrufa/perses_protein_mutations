#!/usr/bin/env python
# coding: utf-8

# here, i i will attempt to find the bug that is preventing certain protein mutations from running quickly...

# In[12]:


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


# In[ ]:





# and we should be using CUDA...

# In[4]:


print(cache.global_context_cache.platform.getName())
print(cache.global_context_cache.platform.getPropertyDefaultValue('CudaPrecision'))


# In[3]:


def deserialize(filename):
    import pickle
    with open(filename, 'rb') as f:
        htf = pickle.load(f)
    return htf


# as you can see above, even in vacuum, there is a big difference in the replica propagation time between the fast and slow systems, even in vacuum;
# perhaps if we remove the constraints in the slow system, we can fix the problem

# In[4]:


import copy


# In[52]:


def create_integrator(htf, constraint_tol):
    """
    create lambda alchemical states, thermodynamic states, sampler states, integrator, and return context, thermostate, sampler_state, integrator
    """
    fast_lambda_alchemical_state = RelativeAlchemicalState.from_system(htf.hybrid_system)
    fast_lambda_alchemical_state.set_alchemical_parameters(0.0, LambdaProtocol(functions = 'default'))
    
    fast_thermodynamic_state = CompoundThermodynamicState(ThermodynamicState(htf.hybrid_system, temperature = temperature),composable_states = [fast_lambda_alchemical_state])
    
    fast_sampler_state = SamplerState(positions = htf._hybrid_positions, box_vectors = htf.hybrid_system.getDefaultPeriodicBoxVectors())
    
#     integrator_1 = integrators.LangevinIntegrator(temperature = temperature,
#                                                      timestep = 0.5* unit.femtoseconds,
#                                                      splitting = 'V R O R V',
#                                                      measure_shadow_work = False,
#                                                      measure_heat = False,
#                                                      constraint_tolerance = constraint_tol,
#                                                      collision_rate = 5.0 / unit.picoseconds)
    mcmc_moves=mcmc.LangevinSplittingDynamicsMove(timestep = 4.0 * unit.femtoseconds,
                                                             collision_rate=5.0 / unit.picosecond,
                                                             n_steps=1,
                                                             reassign_velocities=False,
                                                             n_restart_attempts=20,
                                                             splitting="V R O R V",
                                                             constraint_tolerance=constraint_tol)
    
    
    #print(integrator_1.getConstraintTolerance())
    
#     fast_context, fast_integrator = cache.global_context_cache.get_context(fast_thermodynamic_state, integrator_1)
    
    
#     fast_sampler_state.apply_to_context(fast_context)
    
    return mcmc_moves, fast_thermodynamic_state, fast_sampler_state


    
    
    
    


# In[42]:


def create_langevin_integrator(htf, constraint_tol):
    """
    create lambda alchemical states, thermodynamic states, sampler states, integrator, and return context, thermostate, sampler_state, integrator
    """
    fast_lambda_alchemical_state = RelativeAlchemicalState.from_system(htf.hybrid_system)
    fast_lambda_alchemical_state.set_alchemical_parameters(0.0, LambdaProtocol(functions = 'default'))
    
    fast_thermodynamic_state = CompoundThermodynamicState(ThermodynamicState(htf.hybrid_system, temperature = temperature),composable_states = [fast_lambda_alchemical_state])
    
    fast_sampler_state = SamplerState(positions = htf._hybrid_positions, box_vectors = htf.hybrid_system.getDefaultPeriodicBoxVectors())
    
    integrator_1 = integrators.LangevinIntegrator(temperature = temperature,
                                                     timestep = 4.0* unit.femtoseconds,
                                                     splitting = 'V R O R V',
                                                     measure_shadow_work = False,
                                                     measure_heat = False,
                                                     constraint_tolerance = constraint_tol,
                                                     collision_rate = 5.0 / unit.picoseconds)
#     mcmc_moves=mcmc.LangevinSplittingDynamicsMove(timestep = 4.0 * unit.femtoseconds,
#                                                              collision_rate=5.0 / unit.picosecond,
#                                                              n_steps=1,
#                                                              reassign_velocities=False,
#                                                              n_restart_attempts=20,
#                                                              splitting="V R R R O R R R V",
#                                                              constraint_tolerance=constraint_tol)
    
    
    #print(integrator_1.getConstraintTolerance())
    
    fast_context, fast_integrator = cache.global_context_cache.get_context(fast_thermodynamic_state, integrator_1)
    
    
    fast_sampler_state.apply_to_context(fast_context)
    
    return fast_context, fast_thermodynamic_state, fast_sampler_state, fast_integrator


# In[31]:


def time_integrator(move, thermostate, sstate, num_steps = 10):
    import time
    _time = []
    #integrator.step(1)
    move.apply(thermostate, sstate)
    for i in range(num_steps):
        start = time.time()
        #integrator.step(1)
        move.apply(thermostate, sstate)
        end = time.time() - start
        _time.append(end)
    return np.array(_time)


# In[43]:


def time_lan_integrator(integrator, num_steps = 10):
    import time
    _time = []
    integrator.step(1)
    #move.apply(thermostate, sstate)
    for i in range(num_steps):
        start = time.time()
        integrator.step(1)
        #move.apply(thermostate, sstate)
        end = time.time() - start
        _time.append(end)
    return np.array(_time)


# let's compare the speeds of all the atom maps right now...

# In[44]:


from openmmtools.utils import get_available_platforms


# In[45]:


plats = get_available_platforms()
for plat in plats:
    print(plat.getName())


# In[46]:


dipeptides = ['ALA', 'SER', 'THR', 'CYS']


# In[55]:


print(f"this is for platform {cache.global_context_cache.platform.getName()}")

for mapping in ['weak', 'default', 'strong']:
    for i in ['SER']:
        for j in ['CYS']:
            if i != j:
                for phase in ['sol']:
                    print(f"mapping {mapping} for phase {phase} with dipeptides: {i, j}")
                    htf = deserialize(f'{i}_{j}.{phase}.{mapping}_map.pkl')
                    _, _, _, integrator = create_langevin_integrator(htf, 1e-6)
                    fast = time_lan_integrator(integrator, 250)
                    print(np.average(fast), np.std(fast))
    print()
                    
    


# In[56]:


print(f"this is for platform {cache.global_context_cache.platform.getName()}")

for mapping in ['weak', 'default', 'strong']:
    for i in ['SER']:
        for j in ['CYS']:
            if i != j:
                for phase in ['sol']:
                    print(f"mapping {mapping} for phase {phase} with dipeptides: {i, j}")
                    htf = deserialize(f'{i}_{j}.{phase}.{mapping}_map.pkl')
                    mcmc_moves, thermostate, sstate = create_integrator(htf, 1e-6)
                    fast = time_integrator(mcmc_moves, thermostate, sstate, 250)
                    print(np.average(fast), np.std(fast))
    print()
                    
    


# In[ ]:





# In[ ]:


# plats = get_available_platforms()
# for plat in plats:
#     print(plat.getName())
# cache.global_context_cache.empty()
# print(f"this is for platform {cache.global_context_cache.platform.getName()}")
# for mapping in ['weak', 'default', 'strong']:
#     for i in dipeptides:
#         for j in dipeptides:
#             if i != j:
#                 for phase in ['vac']:
#                     print(f"mapping {mapping} for phase {phase} with dipeptides: {i, j}")
#                     htf = deserialize(f'{i}_{j}.{phase}.{mapping}_map.pkl')
#                     try:
#                         _, _, _, _int = create_integrator(htf, 1e-6)
#                         fast = time_integrator(_int, num_steps = 100)
#                         print(np.average(fast), np.std(fast))
#                     except Exception as e:
#                         print(e)
#     print()


# In[ ]:


htf = deserialize(f'SER_CYS.vac.weak_map.pkl')


# the atom map names

# In[ ]:


[(htf._hybrid_to_old_map[hybrid_idx],  list(htf._topology_proposal._old_topology.atoms())[htf._hybrid_to_old_map[hybrid_idx]].name, htf._hybrid_to_new_map[hybrid_idx],  list(htf._topology_proposal._new_topology.atoms())[htf._hybrid_to_new_map[hybrid_idx]].name) for hybrid_idx in htf._atom_classes['core_atoms']]


# In[ ]:


[(htf._hybrid_to_old_map[hybrid_idx], list(htf._topology_proposal._old_topology.atoms())[htf._hybrid_to_old_map[hybrid_idx]].name, hybrid_idx) for hybrid_idx in htf._atom_classes['unique_old_atoms']]


# In[ ]:


[(htf._hybrid_to_new_map[hybrid_idx], list(htf._topology_proposal._new_topology.atoms())[htf._hybrid_to_new_map[hybrid_idx]].name, hybrid_idx) for hybrid_idx in htf._atom_classes['unique_new_atoms']]


# In[ ]:


num_constraints = htf._hybrid_system.getNumConstraints()


# In[ ]:


for idx in reversed(range(num_constraints)):
    p1, p2, r = htf._hybrid_system.getConstraintParameters(idx)
    
#     old_idx, new_idx = htf._hybrid_to_old_map[p1], htf._hybrid_to_new_map[p2]
#     old_name, new_name = list(htf._topology_proposal._old_topology.atoms())[old_idx].name, list(htf._topology_proposal._new_topology.atoms())[new_idx].name
    print(p1, p2, r)


# so we know that the constraint distance changes, which is why we are not mapping the terminal hydrogen
# 

# out of curiosity, can we make a CCCCO --> CCCCS with different mappings and see how it performs?
# 

# In[ ]:


from perses.annihilation.relative import HybridTopologyFactory
from perses.annihilation.lambda_protocol import RelativeAlchemicalState, LambdaProtocol
from perses.rjmc.topology_proposal import TopologyProposal, TwoMoleculeSetProposalEngine, SystemGenerator,SmallMoleculeSetProposalEngine, PointMutationEngine
from perses.rjmc.geometry import FFAllAngleGeometryEngine
from perses.utils.openeye import createSystemFromSMILES


# In[ ]:


system_generator = SystemGenerator(['gaff2.xml'],
                               barostat = None,
                               forcefield_kwargs = {'removeCMMotion': False, 
                                                    'ewaldErrorTolerance': 1e-4, 
                                                    'nonbondedMethod': app.NoCutoff,
                                                    'constraints' : app.HBonds, 
                                                    'hydrogenMass' : 4 * unit.amus})


# In[ ]:





# In[ ]:


old_mol, old_system, old_pos, old_top = createSystemFromSMILES('CCCCO')
old_system = system_generator.build_system(old_top)

new_mol, new_system, new_pos, new_top = createSystemFromSMILES('CCCCS')
new_system = system_generator.build_system(new_top)


# In[ ]:


num_old_constraints = old_system.getNumConstraints()
print(num_old_constraints)
for i in range(num_old_constraints):
    print(old_system.getConstraintParameters(i))


# In[ ]:


num_old_constraints = new_system.getNumConstraints()
print(num_old_constraints)
for i in range(num_old_constraints):
    print(new_system.getConstraintParameters(i))


# In[ ]:


def generate_top_pos_sys(topology, old_oemol, new_oemol, system, positions, system_generator, map_strength):
    """generate point mutation engine, geometry_engine, and conduct topology proposal, geometry propsal, and hybrid factory generation"""
    #create the point mutation engine
    print(f"generating point mutation engine")
    proposal_engine = SmallMoleculeSetProposalEngine(['CCCCO', 'CCCCS'], system_generator, map_strength=map_strength, residue_name='MOL')
    

    #create a geometry engine
    print(f"generating geometry engine")
    geometry_engine = FFAllAngleGeometryEngine(metadata=None, 
                                           use_sterics=False, 
                                           n_bond_divisions=100, 
                                           n_angle_divisions=180, 
                                           n_torsion_divisions=360, 
                                           verbose=True, 
                                           storage=None, 
                                           bond_softening_constant=1.0, 
                                           angle_softening_constant=1.0, 
                                           neglect_angles = False, 
                                           use_14_nonbondeds = False)

    #create a top proposal
    print(f"making topology proposal")
    topology_proposal = proposal_engine.propose(system, topology, old_oemol, new_oemol)

    #make a geometry proposal forward
    print(f"making geometry proposal")
    forward_new_positions, logp_proposal = geometry_engine.propose(topology_proposal, positions, beta)


    #create a hybrid topology factory
    f"making forward hybridtopologyfactory"
    forward_htf = HybridTopologyFactory(topology_proposal = topology_proposal,
                 current_positions =  positions,
                 new_positions = forward_new_positions,
                 use_dispersion_correction = False,
                 functions=None,
                 softcore_alpha = None,
                 bond_softening_constant=1.0,
                 angle_softening_constant=1.0,
                 soften_only_new = False,
                 neglected_new_angle_terms = [],
                 neglected_old_angle_terms = [],
                 softcore_LJ_v2 = True,
                 softcore_electrostatics = True,
                 softcore_LJ_v2_alpha = 0.85,
                 softcore_electrostatics_alpha = 0.3,
                 softcore_sigma_Q = 1.0,
                 interpolate_old_and_new_14s = False,
                 omitted_terms = None)
    
    return topology_proposal, forward_new_positions, forward_htf



# In[ ]:


_, weak_sm_new_pos, weak_sm_htf = generate_top_pos_sys(old_top, old_mol, new_mol, old_system, old_pos, system_generator, 'weak')
_, default_sm_new_pos, default_sm_htf = generate_top_pos_sys(old_top, old_mol, new_mol, old_system, old_pos, system_generator, 'default')
_, strong_sm_new_pos, strong_sm_htf = generate_top_pos_sys(old_top, old_mol, new_mol, old_system, old_pos, system_generator, 'strong')


# In[ ]:





# In[ ]:


print(weak_sm_htf._topology_proposal._unique_new_atoms)
print(weak_sm_htf._topology_proposal._unique_old_atoms)

print(default_sm_htf._topology_proposal._unique_new_atoms)
print(default_sm_htf._topology_proposal._unique_old_atoms)

print(strong_sm_htf._topology_proposal._unique_new_atoms)
print(strong_sm_htf._topology_proposal._unique_old_atoms)


# In[ ]:



for htf in [weak_sm_htf, default_sm_htf, strong_sm_htf]:
    _, _, _, _int = create_integrator(htf, 1e-6)
    fast = time_integrator(_int, num_steps = 100)
    print(np.average(fast), np.std(fast))


# In[ ]:




