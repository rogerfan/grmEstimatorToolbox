#!/usr/bin/env python
''' Template that shows an example on how to use calcSimTreatEffects_mpi()
    from the the grmEstimatorToolbox.
'''

# standard library
import os
import sys
from mpi4py import MPI

# edit pythonpath
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir) 

# project library
import grmToolbox


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

#
# Calculating Treatment Effects
# 
treat_effects = grmToolbox.calcSimTreatEffects_mpi(20, outputfile = 'grmTreatEffects.txt')

if rank == 0:
    print treat_effects