#!/usr/bin/env python
from __future__ import absolute_import 
from protein_test_cycle import *

graph = generate_fully_connected_perturbation_graph()
#print(f"graph edges: {graph.edges()}")

def run_sim(_graph, start_protein, end_protein):
    for phase in ['sol', 'vac']:
        print(f"running phase: {phase}")
        hss, reporter = create_hss(f"{start_protein}_{end_protein}.{phase}.default_map.nc", _graph.edges[(start_protein, end_protein)][f"{phase}_htf"], selection_string = 'protein', checkpoint_interval = 100, n_states = 11)
        hss.extend(10000)
        
run_sim(graph, str(sys.argv[1]), str(sys.argv[2]))

