## @package halo
#  The package containing the general information for all DM halo models.

import numpy as np
import pyfiles.data_models.constants as constants
import pyfiles.models.cdm.cdm_params as cdm_params
import pyfiles.models.cdm.cdm_funcs as cdm_funcs
import pyfiles.models.alps.alp_params as alp_params
import pyfiles.models.alps.alp_funcs as alp_funcs

## @var standard_consts_in
#  dictionary \n
#  Dictionary containing standard constants.
#  @var kmTOGeV
#  float \n
#  Conversion from km to GeV
#  @var sTOGeV
#  float \n
#  Conversion from s to GeV
#  @var kpcTOGeV
#  float \n
#  Conversion from kpc to GeV
#  @var msun
#  float \n
#  Mass of sun in GeV
#  @var MP
#  float \n
#  Planck mass in GeV
#  @var fitting_dict_in
#  constants.fitting_dict instance \n
#  Instance of the constants.fitting_dict class using all default values
standard_consts_in=constants.standard()
kmTOGeV=standard_consts_in['kmTOGeV']
sTOGeV=standard_consts_in['sTOGeV']
kpcTOGeV=standard_consts_in['kpcTOGeV']
msun=standard_consts_in['msun']
MP=standard_consts_in['MP']
fitting_dict_in=constants.fitting_dict()

## The class containing definitions to describe a DM halo.
class halo:

    ## Define the constructor of the halo class.
    #  This defines the constructor of the halo class.
    #  @param self
    #  object pointer
    #  @param model
    #  str \n
    #  Model to assume for DM halo.
    #  Can be equal to :
    #  - Burkert
    #  - DC14
    #  - Einasto
    #  - NFW
    #  - psi_single
    #  - psi_multi
    #  @param data
    #  dictionary \n
    #  Dictionary containing all the necessary data for a given galaxy.
    #  This can be created using galaxy.galaxy.
    #  @param fit_dict_in (optional)
    #  constants.fitting_dict instance \n
    #  Instance of the constants.fitting_dict class.
    #  Contains all the necessary rules and values to be used in fitting.
    def __init__(self,model,data,fit_dict_in=fitting_dict_in):

        ## @var model
        #  str \n
        #  Model to assume for DM halo.
        #  @var data
        #  dictionary \n
        #  Dictionary containing all the necessary data for a given galaxy.
        #  This can be created using galaxy.galaxy.
        #  @var fit_dict_in
        #  constants.fitting_dict instance \n
        #  Instance of the constants.fitting_dict class.
        #  @var args_opts
        #  dictionary \n
        #  Dictionary of rules to be assumed during fitting
        #  Instance of constants.fitting_dict.args_opts
        #  @var params_vals
        #  dictionary \n
        #  Dictionary of variable values to be used during fitting.
        #  Instance of constants.fitting_dict.params_vals
        #  @var params
        #  lfmit.Parameters instance \n
        #  Instance of the lmfit.Parameters class
        #  All model parameters contained here.
        #  This can be created using cdm_params.halo.params (for CDM models) or alp_params.soliton.params (for ULDM models)
        #  @var halo_init
        #  Can be :
        #  - cdm_funcs.halo instance
        #  - alp_funcs.soliton instance \n
        #  Instance of cdm_funcs.halo or alp_funcs.soliton class depending on chosen model.
        self.model=model
        self.data=data
        self.fit_dict_in=fit_dict_in
        self.args_opts=self.fit_dict_in.args_opts()
        self.params_vals=self.fit_dict_in.params_vals()

        if (self.model=='Burkert' or self.model=='DC14' or self.model=='Einasto' or self.model=='NFW'):
            cdm_params_in=cdm_params.halo(self.model,self.data,fit_dict_in=self.fit_dict_in)
            self.params=cdm_params_in.params()
        elif (self.model=='psi_single' or self.model=='psi_multi'):
            alp_params_in=alp_params.soliton(self.model,data,fit_dict_in=self.fit_dict_in)
            self.params=alp_params_in.params()
        else:
            self.params=None
            print('Invalid halo model.')

        if (self.model=='Burkert' or self.model=='DC14' or self.model=='Einasto' or self.model=='NFW'):
            self.halo_init=cdm_funcs.halo(self.model,fit_dict_in=self.fit_dict_in)
        elif (self.model=='psi_single' or self.model=='psi_multi'):
            self.halo_init=alp_funcs.soliton(self.model,fit_dict_in=self.fit_dict_in)
            
    ## Define the DM halo mass within some radius assuming the chosen model.
    #  This defines the total DM halo mass within the given radius assuming the given model parameters.
    #  Particular equations for mass depends on chosen model.
    #  @param self
    #  object pointer
    #  @param params
    #  lmfit.Parameters instance \n
    #  Instance of the lmfit.Parameters class.
    #  All model parameters contained here.  
    #  This can be created using cdm_params.halo.params (for CDM models) or alp_params.soliton.params (for ULDM models)
    #  @param r
    #  ndarray[N] \n
    #  Numpy array of N radii values in units of kpc.
    #  @returns
    #  ndarray[N] \n
    #  Numpy array of N mass values in units of solar mass.
    def mass(self,params,r):
        Min=self.halo_init.mass(params,r)
        return np.asarray(Min)
        
    ## Define the total circular velocity of the DM halo at some radius assuming the chosen model.
    #  This calculates the total circular velocity of the DM halo at the given radius assuming the given model parameters.
    #  Total circular velocity is given by,
    #  \f[ V(r) = \sqrt{\frac{M(r)}{M_P^2 \, r}} \f]
    #  where \f$M(r)\f$ = halo.halo.mass, and \f$M_P\f$ is the Planck mass (see constants.standard)
    #  @param self
    #  object pointer
    #  @param params
    #  lmfit.Parameters instance \n
    #  Instance of the lmfit.Parameters class.
    #  All model parameters contained here.
    #  This can be created using cdm_params.halo.params (for CDM models) or alp_params.soliton.params (for ULDM models)
    #  @param r
    #  ndarray[N] \n
    #  Numpy array of N radii values in units of kpc.
    #  @returns
    #  ndarray[N] \n
    #  Numpy array of N circular velocity values in units of km/s.
    def velocity(self,params,r):
        Vin=np.sqrt(self.mass(params,r)*msun/(MP**2*r*kpcTOGeV))*sTOGeV/kmTOGeV
        return np.asarray(Vin)