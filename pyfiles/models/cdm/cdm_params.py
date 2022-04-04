## @package cdm_params
#  The package containing the parameters for the CDM halo models.

import numpy as np
import lmfit
import pyfiles.data_models.constants as constants

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

## The class containing all base functions for the CDM halo parameters.
#  This class contains functions necessary to describe CDM halo parameters for various cases.
class base_funcs:

    ## Define the constructor of the base_funcs class.
    #  This defines the constructor of the base_funcs class.
    #  @param self
    #  object pointer
    def __init__(self):
        pass

    ## Define the expression for V200 to be assumed in particular cases.
    #  This defines the allowable values for V200 in \f$\log_{10}\f$ space when using the fit routine
    #  constants.fitting_dict.fit_routine = 'DC14_check'.
    #  The allowable values for V200 are given by,
    #  \f[ V_{200} \geq \left[\left(M_* + M_{\mbox{gas}}\right) 
    #  \sqrt{\frac{2 \pi \, \rho_{\mbox{crit}}}{3}} \frac{20 \times 10^9}{0.2 \, M_P^3}\right]^{1/3}, \f]
    #  where,
    #  \f[ M_* \approx \left( \tilde{\Upsilon}_{\mbox{disk}} + \tilde{\Upsilon}_{\mbox{bulge}} \right) L. \f]
    #  Here, \f$ \rho_{\mbox{crit}} = \f$ rhocrit and \f$ M_P \f$ = MP in constants.standard, 
    #  \f$ M_{\mbox{gas}} \f$ = Mgas and \f$ L = \f$ Luminosity in galaxy.galaxy.data, 
    #  \f$ \tilde{\Upsilon}_{\mbox{disk}} = \f$ MLD and \f$ \tilde{\Upsilon}_{\mbox{bulge}} = \f$ MLB in 
    #  cdm_params.halo.params.
    #  @param self
    #  object pointer
    #  @param mstar
    #  float \n
    #  Total stellar mass in units of \f$10^9\f$ solar masses.
    #  @param mgas
    #  float \n
    #  Total HI gas mass in units of \f$10^9\f$ solar masses.
    #  @param vfac
    #  float \n
    #  Factor to vary in fitting procedure.
    #  @returns
    #  float \n
    #  Value for V200 in \f$\log_{10}\f$ space 
    #  in units of \f$\log_{10}\left(\mbox{km} \, \mbox{s}^{-1}\right)\f$.
    def v200min_frac_dc14check(self,mstar,mgas,vfac):
        const=np.sqrt(3/(2*np.pi*rhoc))*MP**3/20/(10**9*msun)
        v200min=((mstar+mgas)/(0.2*const))**(1/3)
        return np.log10(vfac*v200min/kmTOGeV*sTOGeV)

    ## Define the minimum allowed V200 for the DC14 model.
    #  This defines the minimum allowed V200 for the DC14 model.
    #  This is assumed when using the fit routines :
    #   - constants.fitting_dict.fit_routine = 'uni_priors',
    #   - constants.fitting_dict.fit_routine = 'c200_priors_check',
    #   - constants.fitting_dict.fit_routine = 'v200_priors_check',
    #   - constants.fitting_dict.fit_routine = 'MLd_priors_check',
    #   - constants.fitting_dict.fit_routine = 'MLb_priors_check',
    #   - constants.fitting_dict.fit_routine = 'CDM_check'
    #  The allowed values for V200 are given by,
    #  \f[ V_{200} \geq \left[10^{1.3} M_* 
    #  \sqrt{\frac{2 \pi \, \rho_{\mbox{crit}}}{3}} \frac{20 \times 10^9}{M_P^3}\right]^{1/3}, \f]
    #  where,
    #  \f[ M_* \approx \left( \tilde{\Upsilon}_{\mbox{disk}} + \tilde{\Upsilon}_{\mbox{bulge}} \right) L. \f]
    #  Here, \f$ \rho_{\mbox{crit}} = \f$ rhocrit and \f$ M_P \f$ = MP in constants.standard, 
    #  \f$ M_{\mbox{gas}} \f$ = Mgas and \f$ L = \f$ Luminosity in galaxy.galaxy.data, 
    #  \f$ \tilde{\Upsilon}_{\mbox{disk}} = \f$ MLD and \f$ \tilde{\Upsilon}_{\mbox{bulge}} = \f$ MLB in 
    #  cdm_params.halo.params.
    #  @param self
    #  object pointer
    #  @param mstar
    #  float \n
    #  Total stellar mass in units of \f$10^9\f$ solar masses.
    #  @returns
    #  float \n
    #  Minimun allowed value for V200 for the DC14 model for various cases.
    def v200min_dc14(self,mstar):
        const=np.sqrt(3/(2*np.pi*rhoc))*MP**3/20/(10**9*msun)
        v200min=(10**(1.3)*mstar/const)**(1/3)
        return v200min/kmTOGeV*sTOGeV

## The class containing all parameters for fitting the CDM halo models.
#  This class contains all parameters necessary to perform the fitting procedures for all CDM halo models.
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

    ## Define the parameters to be assumed.
    #  This defines the parameters to be assumed during the fitting procedures for all CDM halos.
    #  @param self
    #  object pointer
    #  @returns
    #  lmfit.Parameters instance
    #  Instance of the lmfit.Parameters class
    def params(self):
        vbul=self.data['Vbulge']
        lum=self.data['Luminosity']
        params=lmfit.Parameters()
        params._asteval.symtable['v200min_dc14'] = base_funcs().v200min_dc14
        params._asteval.symtable['v200min_frac_dc14check'] = base_funcs().v200min_frac_dc14check

        params.add_many(('c200',self.c200[0],True,self.c200[1],self.c200[2]),
                        ('luminosity',lum,False),
                        ('MLd',self.MLD[0],True,self.MLD[1],self.MLD[2]))

        if vbul.all()==0:
            if self.args_opts['fitting_routine']['DC14_check']==True:
                params.add(name='mstar',expr='10**(MLd)*luminosity')
            else:
                params.add(name='mstar',expr='MLd*luminosity')
        else:     
            if (self.args_opts['fitting_routine']['DC14_check']==False and self.args_opts['fitting_routine']['Einasto_check']==False):
                params.add(name='MLb',value=self.MLB[0],vary=True,min=self.MLB[1],max=self.MLB[2])
                params.add(name='mstar',expr='(MLd+MLb)*luminosity')
            else:
                if (self.args_opts['fitting_routine']['Einasto_check']==True):
                    params.add(name='MLb',expr='1.4*MLd',min=self.MLB[1],max=self.MLB[2])
                    params.add(name='mstar',expr='(MLd+MLb)*luminosity')
                else:
                    params.add(name='mstar',expr='10**(MLd)*luminosity')

        if (self.args_opts['fitting_routine']['DC14_check']==False):
            if (self.model=='Burkert' or self.model=='Einasto' or self.model=='NFW'):
                params.add(name='v200',value=self.v200[0],vary=True,min=self.v200[1],max=self.v200[2])
            else:
                params.add(name='v200_factor',value=self.v200fac[0],vary=True,min=self.v200fac[1]),
                params.add(name='v200',expr='v200_factor*v200min_dc14(mstar)',min=self.v200[1],max=self.v200[2])
            if (self.model=='Einasto'):
                params.add(name='alpha',value=self.alpha[0],vary=True,min=self.alpha[1],max=self.alpha[2])
            else:
                pass
        else:
            mgas=self.data['Mgas']
            params.add(name='mgas',value=mgas,vary=False)
            params.add(name='v200_factor',value=self.v200fac[0],vary=True,min=self.v200fac[1]),
            params.add(name='v200',expr='v200min_frac_dc14check(mstar,mgas,v200_factor)',min=self.v200[1],max=self.v200[2])

        return params