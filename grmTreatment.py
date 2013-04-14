#!/usr/bin/env python
''' ---------------------------------------------------------------------------

    Econ 419: Final Project
    Roger Fan

    This file is part of the Generalized Roy Toolbox. 
    
    The Generalized Roy Toolbox is free software: you can redistribute it 
    and/or modify it under the terms of the GNU General Public License as 
    published by the Free Software Foundation, either version 3 of the 
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.    
 
    ---------------------------------------------------------------------------
 
    This module contains the capabilities required for the calculation
    of simulated treatment effects of the Generalized Roy Model.
 
'''

# standard library
import os
import numpy as np
import json
from mpi4py import MPI


#
# Public Functions
#

def read_rslt():
    ''' Reads in estimates from file. 

        Returns
        ---------
        Dict with keys 'Y0_beta', 'Y1_beta', 'D_gamma'.

    '''    
    assert (os.path.exists('grmRslt.json')) 

    with open('grmRslt.json') as file_:
        rslt = json.load(file_)

    rslt['Y0_beta'] = np.array(rslt['Y0_beta'])
    rslt['Y1_beta'] = np.array(rslt['Y1_beta'])
    rslt['D_gamma'] = np.array(rslt['D_gamma'])

    return rslt

def read_data(num_x):
    ''' Reads in data from file. 

        Argument
        ---------
        num_x : int
            Number of X variables in the model.

        Returns
        ---------
        Dict with keys 'Y', 'D', 'X', 'Z'.

    '''
    assert (os.path.exists('grmData.dat')) 
    raw = np.genfromtxt('grmData.dat')

    Y = raw[:, 0]
    D = raw[:, 1]
    X = raw[:, 2:2+num_x]
    Z = raw[:, 2+num_x:]

    data = {}
    data['Y'] = Y
    data['D'] = D
    data['X'] = X
    data['Z'] = Z

    return data

def calcSimTreatEffects(simnum, outputfile = False):
    ''' Simulates data and calculates treatment effects.

        Arguments
        ---------
        simnum : int
            Number of simulations to perform.
        outputfile : str, optional
            Saves output to the given file is provided.

        Returns
        ---------
        Dict with keys 'ATE', 'TT', 'TUT', 'simnum'.

    '''

    # Load Data
    rslt = read_rslt()
    data = read_data(rslt['Y1_beta'].shape[0])

    # Perform Simulations and Calculate Treatment EFfects
    treat_effects_t = []
    for i in range(simnum):
        simdata = _genSimData(rslt, data)
        treat_effects_iter = _calcTreatEffects(simdata)
        treat_effects_t.append(treat_effects_iter)

    # Avg Treatment Effects
    treat_effects = np.vstack(treat_effects_t)
    avg_treat_effects = np.mean(treat_effects, axis=0)

    # Output
    avg_treat_effects_d = {}
    avg_treat_effects_d['ATE'] = avg_treat_effects[0]
    avg_treat_effects_d['TT']  = avg_treat_effects[1]
    avg_treat_effects_d['TUT'] = avg_treat_effects[2]
    avg_treat_effects_d['sim_num'] = simnum

    if outputfile is not False:
        with open(outputfile, 'w') as file_:
            json.dump(avg_treat_effects_d, file_)
        print "Treatment Effects ({} Total Simulations) saved to \'{}\'.".format(simnum, outputfile)

    return avg_treat_effects_d

def calcSimTreatEffects_mpi(simnum, outputfile = False):
    ''' Simulates data and calculates treatment effects.
        Allows usage of MPI for parallel processing.

        Arguments
        ---------
        simnum : int
            Number of simulations to perform per processor.
        outputfile : str, optional
            Saves output to the given file is provided.

        Returns
        ---------
        Dict with keys 'ATE', 'TT', 'TUT', 'simnum'.

    '''

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Load Data
    rslt = read_rslt()
    data = read_data(rslt['Y1_beta'].shape[0])

    # Perform Simulations and Calculate Treatment EFfects
    treat_effects_t = []
    for i in range(simnum):
        simdata = _genSimData(rslt, data)
        treat_effects_iter = _calcTreatEffects(simdata)
        treat_effects_t.append(treat_effects_iter)

    # Avg Treatment Effects
    treat_effects = np.vstack(treat_effects_t)
    avg_treat_effects = np.mean(treat_effects, axis=0)

    # Collect Effects 
    avg_treat_effects_all = np.array(comm.gather(avg_treat_effects,root=0))

    if rank == 0:

        # Average Effects over Processors
        avg_treat_effects = np.mean(avg_treat_effects_all, axis=0)

        # Output
        totsimnum = size * simnum

        avg_treat_effects_d = {}
        avg_treat_effects_d['ATE'] = avg_treat_effects[0]
        avg_treat_effects_d['TT']  = avg_treat_effects[1]
        avg_treat_effects_d['TUT'] = avg_treat_effects[2]
        avg_treat_effects_d['sim_num'] = totsimnum

        if outputfile is not False:
            with open(outputfile, 'w') as file_:
                json.dump(avg_treat_effects_d, file_)
            print "Treatment Effects ({} Total Simulations) saved to \'{}\'.".format(totsimnum, outputfile)

        return avg_treat_effects_d


#
# Private Functions
#

def _genSimData(rslt, data):

    # Distribute Data and Parameters
    X = data['X']
    Z = data['Z']
    numAgents  = X.shape[0]
    
    Y1_beta    = np.array(rslt['Y1_beta'])
    Y0_beta    = np.array(rslt['Y0_beta'])
    D_gamma    = np.array(rslt['D_gamma'])
    
    U1_var     = rslt['U1_var'] 
    U0_var     = rslt['U0_var'] 
    V_var      = 1.
    U1V_rho    = rslt['U1V_rho']  
    U0V_rho    = rslt['U0V_rho']  

    # Construct Auxiliary Objects
    U1V_cov      = U1V_rho*np.sqrt(U1_var)*np.sqrt(V_var)
    U0V_cov      = U0V_rho*np.sqrt(U0_var)*np.sqrt(V_var)

    # Draw Errors
    covs  = np.diag([U1_var, U0_var, V_var])
    covs[0,2] = U1V_cov 
    covs[2,0] = covs[0,2]
    covs[1,2] = U0V_cov
    covs[2,1] = covs[1,2]
    
    U = np.random.multivariate_normal(np.tile(0.0, 3), covs, numAgents)

    U1 = U[:,0]
    U0 = U[:,1]
    V  = U[:,2]

    # Construct Level Indicators
    Y1_level = np.dot(Y1_beta, X.T)
    Y0_level = np.dot(Y0_beta, X.T)
    D_level  = np.dot(D_gamma, Z.T)

    # Simulate
    Y1 = np.tile(np.nan, (numAgents))
    Y0 = np.tile(np.nan, (numAgents))
    Y  = np.tile(np.nan, (numAgents))
    D  = np.tile(np.nan, (numAgents))
    
    expectedBenefits = Y1_level - Y0_level
    cost             = D_level  + V 

    def decisionRule(expBen, cost):
        return np.float(expBen - cost > 0)

    D = np.array(map(decisionRule, expectedBenefits, cost))

    Y1 = Y1_level + U1
    Y0 = Y0_level + U0

    Y = D*Y1 + (1.0-D)*Y0

    # Check quality of simulated sample. 
    assert (np.all(np.isfinite(Y1)))
    assert (np.all(np.isfinite(Y0)))
    
    assert (np.all(np.isfinite(Y)))
    assert (np.all(np.isfinite(D)))
    
    assert (Y1.shape == (numAgents, ))
    assert (Y0.shape == (numAgents, ))
    
    assert (Y.shape  == (numAgents, ))
    assert (D.shape  == (numAgents, ))
    
    assert (Y1.dtype == 'float')
    assert (Y0.dtype == 'float')
    assert (Y.dtype == 'float')
    assert (D.dtype == 'float')

    # Output
    simdata = {}
    simdata['Y']  = Y
    simdata['Y0'] = Y0
    simdata['Y1'] = Y1
    simdata['D']  = D
    simdata['X']  = X
    simdata['Z']  = Z

    return simdata

def _calcTreatEffects(simdata):
    Y1 = simdata['Y1']
    Y0 = simdata['Y0']
    D  = simdata['D']

    Y1_T  = Y1[D == 1.]
    Y0_T  = Y0[D == 1.]
    Y1_UT = Y1[D == 0.]
    Y0_UT = Y0[D == 0.]

    ATE = np.sum(Y1    - Y0   ) / Y1.shape[0]
    TT  = np.sum(Y1_T  - Y0_T ) / Y1_T.shape[0]
    TUT = np.sum(Y1_UT - Y0_UT) / Y1_UT.shape[0]

    return np.array([ATE, TT, TUT])





