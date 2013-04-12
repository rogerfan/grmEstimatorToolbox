''' ---------------------------------------------------------------------------

    Copyright 2013    Philipp Eisenhauer, Stefano Mosso
    Modified by: Roger Fan
    
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
 
    This module contains the capabilities required for the processing of
    an initialization file for the Generalized Roy Model.
 
'''

# standard library
import json

import numpy as np


#
# Public Interface.
#

def read():
    ''' Public interface to request reading of an initFile. '''   

    # Read File from Disk.
    initDict = _read()

    # Quality checks.
    assert (_check(initDict) == True)
    
    # Write File to Disk.
    _write(initDict)
    
    # Finishing.
    return initDict
    

#
# Private Functions
#

def _write(initDict):
    ''' Write dictionary that results from the processing of the *.ini
        file.

    '''
    # Write out the *.json file.
    with open('grmInit.json', 'w') as file_:
        
        json.dump(initDict, file_)
    
def _read():
    ''' This function reads the *.ini file, collects the information in a 
        dictionary and outputs it to a *.json file.

    '''
    # Initialization
    dict_ = {}
    
    with open('grmInit.ini', 'r') as file_:
        
        for count in range(35):
            
            str_ = file_.readline()
            
            # Number of Agents.
            if(count == 6):
                
                str_ = str_.split('!')[0]     
                dict_['numAgents'] = int(str_)

            # File Name.    
            if(count == 7):
                
                str_ = str_.split("'")[1]
                dict_['fileName'] = str_    
            
            # Coefficients for Treated State .   
            if(count == 11):
                
                str_ = _processLine(str_)
                
                for i in range(len(str_)):
                
                    str_[i] = float(str_[i])

                dict_['Y1_beta'] = list(str_)       
            
            # Coefficients for Untreated State.   
            if(count == 12):
                
                str_ = _processLine(str_)
                
                for i in range(len(str_)):
                
                    str_[i] = float(str_[i])
                
                dict_['Y0_beta'] = list(str_)  
            
            # Coefficients in Cost Equation. 
            if(count == 16):
                
                str_ = _processLine(str_)
                
                for i in range(len(str_)):
                
                    str_[i] = float(str_[i])
                
                dict_['D_gamma'] = list(str_)  
            
            # Variance of Disturbance in Treated State. 
            if(count == 20):
                
                str_ = _processLine(str_)                
                dict_['U1_var'] = float(str_[0])  
            
            # Variance of Disturbance in Untreated State.
            if(count == 21):
                
                str_ = _processLine(str_)
                dict_['U0_var'] = float(str_[0])  
            
            # Variance of Disturbance in Cost Equation.
            if(count == 22):
                
                str_ = _processLine(str_)
                dict_['V_var'] = float(str_[0])  
                            
            # Correlation Coefficients (U1, V)            
            if(count == 23):
                
                str_ = _processLine(str_)
                dict_['U1V_rho'] = float(str_[0])  
 
            # Correlation Coefficients (U1, V)            
            if(count == 24):
                
                str_ = _processLine(str_)
                dict_['U0V_rho'] = float(str_[0])     

            # Maximum Number of Iterations          
            if(count == 28):
                
                str_ = _processLine(str_)
                dict_['maxiter'] = int(str_[0])   

            # Maximum Number of Iterations          
            if(count == 32):
                
                str_ = _processLine(str_)
                dict_['randomSeed'] = int(str_[0])   
                    
    # Finishing.
    return dict_

def _processLine(str_):
    ''' Basic processing of line from initialization file. '''

    # Antibugging.
    assert (isinstance(str_, str))
    
    # Process string.
    str_ = str_.split('!')[0]
    str_ = str_.replace('\t', '')
    str_ = str_.replace('\n', '')
    str_ = str_.split(',')    
    
    # Finishing.
    return str_

def _check(initDict):
    ''' Check integrity of initFile dict. '''

    # Antibugging.
    assert (isinstance(initDict, dict))
    
    # Check number of agents.
    assert (initDict['numAgents'] > 0)
    assert (isinstance(initDict['numAgents'], int))
    
    # Filename
    assert (isinstance(initDict['fileName'], str))
    
    # Coefficients in treated state.
    assert (isinstance(initDict['Y1_beta'], list))
    assert (np.all(np.isfinite(initDict['Y1_beta'])))

    # Coefficients in untreated state.
    assert (isinstance(initDict['Y0_beta'], list))
    assert (np.all(np.isfinite(initDict['Y0_beta'])))
    
    # Implications of outcome coefficient vectors.
    assert (len(initDict['Y0_beta']) == len(initDict['Y1_beta']))
    
    # Coefficients for choice model.
    assert (isinstance(initDict['D_gamma'], list))
    assert (np.all(np.isfinite(initDict['D_gamma'])))
    
    # Variance in treated state.
    assert (isinstance(initDict['U1_var'], float))
    assert (initDict['U1_var'] > 0.0)

    # Variance in untreated state.
    assert (isinstance(initDict['U0_var'], float))
    assert (initDict['U0_var'] > 0.0)    

    # Variance in choice model
    assert (isinstance(initDict['V_var'], float))
    assert (initDict['V_var'] > 0.0)    
    
    # Correlation choice and treated outcome.
    assert (isinstance(initDict['U1V_rho'], float))
    assert (-1.0 < initDict['U1V_rho'] < 1.0)   
    
    # Correlation choice and untreated outcome.
    assert (isinstance(initDict['U0V_rho'], float))
    assert (-1.0 < initDict['U0V_rho'] < 1.0)  
    
    # Maximum number of iterations.
    assert (isinstance(initDict['maxiter'], int))
    assert (initDict['maxiter'] > 0)
    
    # Finishing.
    return True