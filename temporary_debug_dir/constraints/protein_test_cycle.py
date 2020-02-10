#!/usr/bin/env python
# coding: utf-8

# # Here, I document an attempt to validate a small set of protein mutations in vacuum and solvent with the following checks...

# 1. generate alanine dipeptide --> valine dipeptide in vac/solvent and conduct a forward _and_ reverse parallel tempering FEP calculation; the check passes if the forward free energy is equal to the reverse free energy within an error tolerance
# 2. generate alanine dipeptide --> valine dipeptide --> isoleucine dipeptide --> glycine dipeptide and attempt to close the thermodynamic cycle within an error tolerance

# In[ ]:


from __future__ import absolute_import

import networkx as nx
from perses.dispersed import feptasks
from perses.utils.openeye import *
from perses.utils.data import load_smi
from perses.annihilation.relative import HybridTopologyFactory
from perses.annihilation.lambda_protocol import RelativeAlchemicalState, LambdaProtocol
from perses.rjmc.topology_proposal import TopologyProposal, TwoMoleculeSetProposalEngine, SystemGenerator,SmallMoleculeSetProposalEngine, PointMutationEngine
from perses.rjmc.geometry import FFAllAngleGeometryEngine
import simtk.openmm.app as app
import sys

from openmmtools.states import ThermodynamicState, CompoundThermodynamicState, SamplerState

import pymbar
import simtk.openmm as openmm
import simtk.openmm.app as app
import simtk.unit as unit
import numpy as np
from openmoltools import forcefield_generators
import copy
import pickle
import mdtraj as md
from io import StringIO
from openmmtools.constants import kB
import logging
import os
import dask.distributed as distributed
import parmed as pm
from collections import namedtuple
from typing import List, Tuple, Union, NamedTuple
from collections import namedtuple
import random
#beta = 1.0/(kB*temperature)
import itertools
import os
from openeye import oechem
from perses.utils.smallmolecules import render_atom_mapping
from perses.tests.utils import validate_endstate_energies

ENERGY_THRESHOLD = 1e-6
temperature = 300 * unit.kelvin
kT = kB * temperature
beta = 1.0/kT


# In[ ]:


from perses.samplers.multistate import HybridSAMSSampler, HybridRepexSampler
from openmmtools.multistate import MultiStateReporter, MultiStateSamplerAnalyzer
from openmmtools import mcmc, utils
from perses.annihilation.lambda_protocol import LambdaProtocol


# In[ ]:


def generate_atp(phase = 'vacuum'):
    """
    modify the AlanineDipeptideVacuum test system to be parametrized with amber14ffsb in vac or solvent (tip3p)
    """
    import openmmtools.testsystems as ts
    atp = ts.AlanineDipeptideVacuum(constraints = app.HBonds, hydrogenMass = 4 * unit.amus)

    forcefield_files = ['gaff.xml', 'amber14/protein.ff14SB.xml', 'amber14/tip3p.xml']
    
    if phase == 'vacuum':
        barostat = None
        system_generator = SystemGenerator(forcefield_files,
                                       barostat = barostat,
                                       forcefield_kwargs = {'removeCMMotion': False, 
                                                            'ewaldErrorTolerance': 1e-4, 
                                                            'nonbondedMethod': app.NoCutoff,
                                                            'constraints' : None, 
                                                            'hydrogenMass' : 4 * unit.amus})
        atp.system = system_generator.build_system(atp.topology) #update the parametrization scheme to amberff14sb
        
    elif phase == 'solvent':
        barostat = openmm.MonteCarloBarostat(1.0 * unit.atmosphere, 300 * unit.kelvin, 50)
        system_generator = SystemGenerator(forcefield_files,
                                   barostat = barostat,
                                   forcefield_kwargs = {'removeCMMotion': False, 
                                                        'ewaldErrorTolerance': 1e-4, 
                                                        'nonbondedMethod': app.PME,
                                                        'constraints' : None, 
                                                        'hydrogenMass' : 4 * unit.amus})
    
    if phase == 'solvent':
        modeller = app.Modeller(atp.topology, atp.positions)
        modeller.addSolvent(system_generator._forcefield, model='tip3p', padding=9*unit.angstroms, ionicStrength=0.15*unit.molar)
        solvated_topology = modeller.getTopology()
        solvated_positions = modeller.getPositions()

        # canonicalize the solvated positions: turn tuples into np.array
        atp.positions = unit.quantity.Quantity(value = np.array([list(atom_pos) for atom_pos in solvated_positions.value_in_unit_system(unit.md_unit_system)]), unit = unit.nanometers)
        atp.topology = solvated_topology

        atp.system = system_generator.build_system(atp.topology)
    
    
    return atp, system_generator


# In[ ]:


def generate_top_pos_sys(topology, new_res, system, positions, system_generator):
    """generate point mutation engine, geometry_engine, and conduct topology proposal, geometry propsal, and hybrid factory generation"""
    #create the point mutation engine
    print(f"generating point mutation engine")
    point_mutation_engine = PointMutationEngine(wildtype_topology = topology,
                                                system_generator = system_generator,
                                                chain_id = '1', #denote the chain id allowed to mutate (it's always a string variable)
                                                max_point_mutants = 1,
                                                residues_allowed_to_mutate = ['2'], #the residue ids allowed to mutate
                                                allowed_mutations = [('2', new_res)], #the residue ids allowed to mutate with the three-letter code allowed to change
                                                aggregate = True) #always allow aggregation

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
    topology_proposal, local_map_stereo_sidechain, new_oemol_sidechain, old_oemol_sidechain = point_mutation_engine.propose(current_system = system,
                                  current_topology = topology)

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
    
    return topology_proposal, forward_new_positions, forward_htf, local_map_stereo_sidechain, old_oemol_sidechain, new_oemol_sidechain


# In[ ]:


def create_hss(reporter_name, hybrid_factory, selection_string ='all', checkpoint_interval = 1, n_states = 13):
    lambda_protocol = LambdaProtocol(functions='default')
    reporter = MultiStateReporter(reporter_name, analysis_particle_indices = hybrid_factory.hybrid_topology.select(selection_string), checkpoint_interval = checkpoint_interval)
    hss = HybridRepexSampler(mcmc_moves=mcmc.LangevinSplittingDynamicsMove(timestep= 0.5 * unit.femtoseconds,
                                                                                 collision_rate=5.0 / unit.picosecond,
                                                                                 n_steps=500,
                                                                                 reassign_velocities=False,
                                                                                 n_restart_attempts=20,
                                                                                 splitting="V R R R O R R R V",
                                                                                 constraint_tolerance=1e-06),
                                                                                 hybrid_factory=hybrid_factory, online_analysis_interval=10)
    hss.setup(n_states=n_states, temperature=300*unit.kelvin,storage_file=reporter,lambda_protocol=lambda_protocol,endstates=False)
    return hss, reporter


# let's make a function to generate an n node graph and run a computation on it...

# In[ ]:


def run_wrapper(system, positions, topology, system_generator, dipeptide_name, reporter_name):
    top_prop, new_positions, htf = generate_top_pos_sys(topology, dipeptide_name, system, positions, system_generator)
    hss, reporter = create_hss(reporter_name, htf, selection_string = 'protein', checkpoint_interval = 10, n_states = 13)
    return htf._new_system, new_positions, top_prop._new_topology, hss
    


# In[ ]:


def generate_fully_connected_perturbation_graph(dipeptides = ['ALA', 'SER', 'THR', 'CYS']):
    # generate a fully connected solvation energy graph for the dipeptides specified...
    graph = nx.DiGraph()
    for dipeptide in dipeptides:
        graph.add_node(dipeptide)
    
    #now for edges...
    for i in graph.nodes():
        for j in graph.nodes():
            if i != j:
                graph.add_edge(i, j)
    
    
    #start with ala
    vac_atp, vac_system_generator = generate_atp(phase = 'vacuum')
    sol_atp, sol_system_generator = generate_atp(phase = 'solvent')
    
    graph.nodes['ALA']['vac_sys_pos_top'] = (vac_atp.system, vac_atp.positions, vac_atp.topology)
    graph.nodes['ALA']['sol_sys_pos_top'] = (sol_atp.system, sol_atp.positions, sol_atp.topology)
    
    #turn ala into all of the other dipeptides
    for dipeptide in [pep for pep in dipeptides if pep != 'ALA']:
        for phase, testcase, sys_gen in zip(['vac', 'sol'], [vac_atp, sol_atp], [vac_system_generator, sol_system_generator]):
            top_prop, new_positions, htf, local_map_stereo_sidechain, old_oemol, new_oemol =  generate_top_pos_sys(testcase.topology, dipeptide, testcase.system, testcase.positions, sys_gen)
            new_sys, new_pos, new_top = htf._new_system, htf._new_positions, top_prop._new_topology
            graph.nodes[dipeptide][f"{phase}_sys_pos_top"] = (new_sys, new_pos, new_top)
            graph.edges[('ALA', dipeptide)][f"map_oldmol_newmol"] = (local_map_stereo_sidechain, old_oemol, new_oemol)
            graph.edges[('ALA', dipeptide)][f'{phase}_htf'] = htf

        
        
    #now we can turn all of the other states in to each other!!!
    for edge_start, edge_end in list(graph.edges()):
        if edge_start == 'ALA': #we already did ALA
            continue
        
        for phase, sys_gen in zip(['vac', 'sol'], [vac_system_generator, sol_system_generator]):
            sys, pos, top = graph.nodes[edge_start][f"{phase}_sys_pos_top"]
            top_prop, new_positions, htf, local_map_stereo_sidechain, old_oemol, new_oemol = generate_top_pos_sys(top, edge_end, sys, pos, sys_gen)
            new_sys, new_pos, new_top = htf._new_system, htf._new_positions, top_prop._new_topology
            graph.nodes[edge_end][f"{phase}_sys_pos_top"] = (new_sys, new_pos, new_top)
            graph.edges[(edge_start, edge_end)][f"{phase}_htf"] = htf
            graph.edges[(edge_start, edge_end)][f"map_oldmol_newmol"] = (local_map_stereo_sidechain, old_oemol, new_oemol)
            
    print(f"graph_edges: {graph.edges()}")
    
    return graph
        


# In[ ]:


#os.system(f"rm *.nc")


# In[ ]:


# graph = generate_fully_connected_perturbation_graph()
# print(f"graph edges: {graph.edges()}")

# def run_sim(_graph, start_protein, end_protein):
#     for phase in ['vac', 'sol']:
#         hss, reporter = create_hss(f"{start_protein}_{end_protein}.{phase}.nc", _graph.edges[(start_protein, end_protein)][f"{phase}_htf"], selection_string = 'protein', checkpoint_interval = 10, n_states = 13)
#         hss.extend(5000)
        
# run_sim(graph, str(sys.argv[1]), str(sys.argv[2]))

    


