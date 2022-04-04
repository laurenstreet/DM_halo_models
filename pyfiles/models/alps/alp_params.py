## @package alp_params
#  The package containing the parameters for the ULDM halo models.

import numpy as np
from scipy.special import lambertw
import lmfit
import pyfiles.data_models.constants as constants
import pyfiles.models.cdm.cdm_params as cdm_params

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
#  @var rhoc
#  float \n
#  Critical density of universe in \f$\mbox{GeV}^4\f$
#  @var fitting_dict_in
#  constants.fitting_dict instance \n
#  Instance of the constants.fitting_dict class using all default values
standard_consts_in=constants.standard()
kmTOGeV=standard_consts_in['kmTOGeV']
sTOGeV=standard_consts_in['sTOGeV']
kpcTOGeV=standard_consts_in['kpcTOGeV']
msun=standard_consts_in['msun']
MP=standard_consts_in['MP']
rhoc=standard_consts_in['rhocrit']
fitting_dict_in=constants.fitting_dict()

## Define the value of \f$ \alpha \f$ when using ULDM matched models.
#  This defines the value of \f$ \alpha \f$ in the ULDM matched models.  This value of is fixed from the condition
#  \f$ \rho_{\mbox{sol}}\left(3 \, r_{\mbox{c,sol}}\right) = \rho_{\mbox{Einasto}} \left(3 \, r_{\mbox{c,sol}}\right) \f$.
#  @param m22
#  mass of ULDM in units of \f$ 10^{-22} \, \mbox{eV} \f$ \n
#  @param msol
#  mass of soliton in units of solar masses \n
#  @param c200
#  Einasto profile variable \f$ c_{200} \f$ \n
#  @param V200
#  Einasto profile variable \f$ V_{200} \f$ \n
#  @returns
#  float
#  Value of \f$ \alpha \f$ to be used for ULDM matched models
def alphamatched(m22,msol,c200,V200):
    V200=V200*kmTOGeV/sTOGeV
    piece0=np.log((1.70907*1e32 * c200**3)/(m22**6 * msol**4 * (-1 + 1/(1 + c200) + np.log(1 + c200))))
    piece1=np.log(c200/(m22**2 * msol * V200))
    piece2=np.log((1.42593*1e30 * c200**3)/(m22**6 * msol**4 * (-1 + 1/(1 + c200) + np.log(1 + c200))))
    piece00=-5.62816 - 2 * piece1
    piece11=(np.exp(piece00/(4.7863 + piece2)) * piece00)/piece0
    piece22=-((lambertw(piece11))/(2.81408 + piece1))
    alphain=-(2/piece0) + piece22
    if np.isreal(alphain)==True:
        alphain=alphain.real
    else:
        pass
    return alphain

## The class containing all parameters for fitting the ULDM halo models.
#  This class contains all parameters necessary to perform the fitting procedures for all ULDM halo models.
class soliton:
    
    ## Define the constructor of the soliton class.
    #  This defines the constructor of the soliton class.
    #  @param self
    #  object pointer
    #  @param model
    #  str \n
    #  Model to assume for DM halo.
    #  Can be equal to :
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
        #  @var c200
        #  list \n
        #  Values for c200 in the form [starting value, min, max]
        #  Contained in constants.fitting_dict.params_vals
        #  @var v200
        #  list \n
        #  Values for V200 in the form [starting value, min, max]
        #  Contained in constants.fitting_dict.params_vals
        #  @var v200fac
        #  list \n
        #  Values for v200fac in the form [starting value, min, max]
        #  Contained in constants.fitting_dict.params_vals
        #  @var MLD
        #  list \n
        #  Values for MLd in the form [starting value, min, max]
        #  Contained in constants.fitting_dict.params_vals
        #  @var MLB
        #  list \n
        #  Values for MLb in the form [starting value, min, max]
        #  Contained in constants.fitting_dict.params_vals
        #  @var alpha
        #  list \n
        #  Values for alpha in the form [starting value, min, max]
        #  Contained in constants.fitting_dict.params_vals
        #  @var msol
        #  float \n
        #  Values for msol in the form [starting value, min, max]
        #  Contained in constants.fitting_dict.params_vals
        #  @var m22
        #  float \n
        #  Values for m22 in the form [starting value, min, max]
        #  Contained in constants.fitting_dict.params_vals
        #  @var msol_2
        #  float \n
        #  Values for msol two in the form [starting value, min, max]
        #  Contained in constants.fitting_dict.params_vals
        #  @var m22_2
        #  float \n
        #  Values for m22 two in the form [starting value, min, max]
        #  Contained in constants.fitting_dict.params_vals
        #  @var matched
        #  bool \n
        #  Denotes how to combine the soliton and outer halo.
        #  Equal constants.fitting_dict.sol_match
        #  @var mfree
        #  float \n
        #  Denotes how to treat soliton particle mass in fitting procedure.
        #  Equal to constants.fitting_dict.sol_mfree
        #  @var cdmhalo
        #  float \n
        #  Denotes which CDM profile to use for outer halo in ULDM galactic structure.
        #  Equal to constants.fitting_dict.sol_cdmhalo
        self.model=model
        self.data=data
        self.fit_dict_in=fit_dict_in
        self.args_opts=self.fit_dict_in.args_opts()
        self.params_vals=self.fit_dict_in.params_vals()
        self.c200=self.params_vals['CDM']['c200']
        self.v200=self.params_vals['CDM']['v200']
        self.v200fac=self.params_vals['CDM']['v200_fac']
        self.MLD=self.params_vals['CDM']['MLd']
        self.MLB=self.params_vals['CDM']['MLb']
        self.alpha=self.params_vals['CDM']['alpha']
        self.msol=self.params_vals['soliton']['Msol']
        self.m22=self.params_vals['soliton']['m22']
        self.msol_2=self.params_vals['soliton']['Msol_2']
        self.m22_2=self.params_vals['soliton']['m22_2']
        self.matched=self.args_opts['soliton']['matched']
        self.mfree=self.args_opts['soliton']['mfree']
        self.cdmhalo=self.args_opts['soliton']['cdm_halo']

    ## Define the parameters to be assumed.
    #  This defines the parameters to be assumed during the fitting procedures for all ULDM halos.
    #  @param self
    #  object pointer
    #  @returns
    #  lmfit.Parameters instance
    #  Instance of the lmfit.Parameters class
    def params(self):
        vbul=self.data['Vbulge']
        lum=self.data['Luminosity']
        params=lmfit.Parameters()
        params._asteval.symtable['v200min_dc14'] = cdm_params.base_funcs.v200min_dc14
        params._asteval.symtable['alphamatched'] = alphamatched

        params.add_many(('c200',self.c200[0],True,self.c200[1],self.c200[2]),
                        ('MLd',self.MLD[0],True,self.MLD[1],self.MLD[2]),
                        ('luminosity',lum,False))

        if vbul.all()==0:
            params.add(name='mstar',expr='MLd*luminosity')
        else:     
            params.add(name='MLb',value=self.MLB[0],vary=True,min=self.MLB[1],max=self.MLB[2])
            params.add(name='mstar',expr='(MLd+MLb)*luminosity')
        if (self.cdmhalo=='Burkert' or self.cdmhalo=='Einasto' or self.cdmhalo=='NFW'):
            params.add(name='v200',value=self.v200[0],vary=True,min=self.v200[1],max=self.v200[2])
        else:
            params.add(name='v200_factor',value=self.v200fac[0],vary=True,min=self.v200fac[1])
            params.add(name='v200',expr='v200_factor*v200min_dc14(mstar)',min=self.v200[1],max=self.v200[2])

        if (self.model=='psi_single'):
            if self.mfree==True:
                params.add(name='m22',value=self.m22[0],min=self.m22[1],max=self.m22[2])
            else:
                params.add(name='m22',value=self.fit_dict_in.sol_m22,vary=False)
            params.add(name='Msol',value=self.msol[0],vary=True,min=self.msol[1],max=self.msol[2])
            if (self.cdmhalo=='Einasto'):
                if self.matched==False:
                    params.add(name='alpha',value=self.alpha[0],vary=True,min=self.alpha[1],max=self.alpha[2])
                else:
                    params.add(name='alpha',expr='alphamatched(m22,Msol,c200,v200)')
            else:
                pass

        else:
            params.add_many(('c200_2',self.c200[0],True,self.c200[1],self.c200[2]),
                            ('v200_2',self.v200[0],True,self.v200[1],self.v200[2]))
            if self.mfree==True:
                params.add(name='m22',value=self.m22[0],min=self.m22[1],max=self.m22[2])
                params.add(name='m22_2',value=self.m22_2[0],min=self.m22_2[1],max=self.m22_2[2])
            else:
                params.add(name='m22',value=self.fit_dict_in.sol_m22,vary=False)
                params.add(name='m22_2',value=self.fit_dict_in.sol_m22_2,vary=False)
            params.add(name='Msol',value=self.msol[0],vary=True,min=self.msol[1],max=self.msol[2])
            params.add(name='Msol_2',value=self.msol_2[0],vary=True,min=self.msol_2[1],max=self.msol_2[2])
            if (self.cdmhalo=='Einasto'):
                if self.matched==False:
                    params.add(name='alpha',value=self.alpha[0],vary=True,min=self.alpha[1],max=self.alpha[2])
                    params.add(name='alpha_2',value=1,vary=True,min=self.alpha[1],max=self.alpha[2])
                else:
                    params.add(name='alpha',expr='alphamatched(m22,Msol,c200,v200)')
                    params.add(name='alpha_2',expr='alphamatched(m22_2,Msol_2,c200_2,v200_2)')
            else:
                pass

        return params