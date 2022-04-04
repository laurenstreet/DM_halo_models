## @package alp_funcs
#  The package containing the functions for the ULDM halo models.

import numpy as np
import pyfiles.data_models.constants as constants
import pyfiles.models.cdm.cdm_funcs as cdm_funcs

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
#  @var rhocrit
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
rhocrit=standard_consts_in['rhocrit']
fitting_dict_in=constants.fitting_dict()

## The class containing base functions needed to describe the ULDM halo models.
class base_funcs:

    ## Define the constructor of the base_funcs class.
    #  This defines the constructor of the base_funcs class.
    #  @param self
    #  object pointer
    #  @param model
    #  str \n
    #  Model to assume for DM halo.
    #  Can be equal to :
    #  - psi_single
    #  - psi_multi
    #  @param fit_dict_in (optional)
    #  constants.fitting_dict instance \n
    #  Instance of the constants.fitting_dict class.
    #  Contains all the necessary rules and values to be used in fitting.
    def __init__(self,model,fit_dict_in=fitting_dict_in):

        ## @var model
        #  str \n
        #  Model to assume for DM halo.
        #  @var fit_dict_in
        #  constants.fitting_dict instance \n
        #  Instance of the constants.fitting_dict class.
        #  @var args_opts
        #  dictionary \n
        #  Dictionary of rules to be assumed during fitting
        #  Instance of constants.fitting_dict.args_opts
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
        #  @var m22
        #  float \n
        #  Soliton particle mass
        #  in units of \f$10^{-22} \, \mbox{eV}\f$.
        #  @var m22_2
        #  float \n
        #  Soliton particle mass two
        #  in units of \f$10^{-22} \, \mbox{eV}\f$.
        self.model=model
        self.fit_dict_in=fit_dict_in
        self.args_opts=self.fit_dict_in.args_opts()
        self.matched=self.args_opts['soliton']['matched']
        self.mfree=self.args_opts['soliton']['mfree']
        self.cdmhalo=self.args_opts['soliton']['cdm_halo']
        if self.mfree==False:
            self.m22=self.fit_dict_in.sol_m22
            if self.model=='psi_multi':
                self.m22_2=self.fit_dict_in.sol_m22_2
            else:
                pass
        else:
            pass
    
    ## Define the total soliton mass.
    #  This defines the total soliton mass for the given model parameters.
    #  @param self
    #  object pointer
    #  @param params
    #  lmfit.Parameters instance \n
    #  Instance of the lmfit.Parameters class.
    #  All model parameters contained here.  
    #  This can be created using alp_params.soliton.params.
    #  @returns 
    #  float (for single flavored models) or ndarray[2] (for double flavored models) \n
    #  The total soliton mass (for single flavored models) 
    #  or both total soliton masses, msol_1 and msol_2 (for double flavored models) 
    #  in units of solar mass.
    def msol(self,params):
        msol_1=params['Msol']
        if self.model=='psi_multi':
            msol_2=params['Msol_2']
            msol_fin=np.asarray([msol_1,msol_2])
        else:
            msol_fin=msol_1
        return msol_fin

    ## Define the soliton profile variable rc.
    #  This defines the soliton profile variable rc for the given model parameters.
    #  The soliton profile variable rc is given by,
    #  \f[ r_c \approx 0.228 \left(\frac{M_{\mbox{sol}}}{10^9}\right)^{-1} m^{-2}, \f]
    #  where \f$ M_{\mbox{sol}} = \f$ alp_funcs.base_funcs.msol 
    #  and \f$ m  = \f$ m22 in alp_params.soliton.params.
    #  @param self
    #  object pointer
    #  @param params
    #  lmfit.Parameters instance \n
    #  Instance of the lmfit.Parameters class.
    #  All model parameters contained here.  
    #  This can be created using alp_params.soliton.params.
    #  @returns 
    #  float (for single flavored models) or ndarray[2] (for double flavored models) \n
    #  The halo profile variable rc in units of kpc.
    #  The soliton profile variable rc (for single flavored models) 
    #  or both soliton profile variables, rc_1 and rc_2 (for double flavored models) 
    #  in units of kpc.
    def rc(self,params):
        if self.model=='psi_single':
            msol_1=self.msol(params)
        else:
            msol_1=self.msol(params)[0]
            msol_2=self.msol(params)[1]
        if self.mfree==False:
            m22_in=self.m22
            if self.model=='psi_multi':
                m22_2_in=self.m22_2
            else:
                pass
        else:
            m22_in=params['m22']
            if self.model=='psi_multi':
                m22_2_in=params['m22_2']
            else:
                pass
        rcin1=(2.28*1e8*(m22_in**-2)/msol_1)
        rc_fin=rcin1
        if self.model=='psi_multi':
            rcin2=(2.28*1e8*(m22_2_in**-2)/msol_2)
            rc_fin=np.asarray([rcin1,rcin2])
        else:
            pass
        return rc_fin

    ## Define the soliton profile variable rhoc.
    #  This defines the soliton profile variable rhoc for the given model parameters.
    #  The soliton profile variables rhoc is given by,
    #  \f[ \rho_c \approx 7 \times 10^9 \left( \frac{M_{\mbox{sol}}}{10^9} \right)^4 m^6, \f]
    #  where \f$ M_{\mbox{sol}} = \f$ alp_funcs.base_funcs.msol 
    #  and \f$ m  = \f$ m22 in alp_params.soliton.params. 
    #  @param self
    #  object pointer
    #  @param params
    #  lmfit.Parameters instance \n
    #  Instance of the lmfit.Parameters class.
    #  All model parameters contained here.  
    #  This can be created using alp_params.soliton.params.
    #  @returns 
    #  float (for single flavored models) or ndarray[2] (for double flavored models) \n
    #  The soliton profile variable rhoc (for single flavored models) 
    #  or both soliton profile variables, rhoc_1 and rhoc_2 (for double flavored models)
    #  in units of \f$M_{\odot}/\mbox{kpc}^3\f$.
    def rhoc(self,params):
        if self.model=='psi_single':
            msol_1=self.msol(params)
        else:
            msol_1=self.msol(params)[0]
            msol_2=self.msol(params)[1]
        if self.mfree==False:
            m22_in=self.m22
            if self.model=='psi_multi':
                m22_2_in=self.m22_2
            else:
                pass
        else:
            m22_in=params['m22']
            if self.model=='psi_multi':
                m22_2_in=params['m22_2']
            else:
                pass
        rhocin1=7.00283*1e-27*m22_in**6*msol_1**4
        rhoc_fin=rhocin1
        if self.model=='psi_multi':
            rhocin2=7.00283*1e-27*m22_2_in**6*msol_2**4
            rhoc_fin=np.asarray([rhocin1,rhocin2])
        else:
            pass
        return rhoc_fin

    ## Define the soliton profile variable xc.
    #  This defines the soliton profile variable xc = r/rc for the given model parameters.
    #  The soliton profile variable xc is given by,
    #  \f[ x_c = r/r_c, \f]
    #  where \f$ r_c = \f$ alp_funcs.base_funcs.rc.
    #  @param self
    #  object pointer
    #  @param params
    #  lmfit.Parameters instance \n
    #  Instance of the lmfit.Parameters class.
    #  All model parameters contained here.  
    #  This can be created using alp_params.soliton.params.
    #  @param r
    #  ndarray[N] \n
    #  Numpy array of N radii values in units of kpc.
    #  @returns 
    #  ndarray[N] (for single flavored models) or ndarray[2,N] (for double flavored models) \n
    #  Numpy array of N (for N given radii) soliton profile variables xc (for single flavored models)
    #  or [2,N] soliton profile variables, xc_1 and xc_2 (for double flavored models).
    def xc(self,params,r):
        if self.model=='psi_single':
            rc1=self.rc(params)
            xc1=r/rc1
            xc_fin=xc1
        else:
            rc1=self.rc(params)[0]
            rc2=self.rc(params)[1]
            xc1=r/rc1
            xc2=r/rc2
            xc_fin=np.asarray([xc1,xc2])
        return xc_fin

    ## Define the mass profile of the soliton.
    #  This defines the mass of the soliton at a given radius for the given model parameters.
    #  @param self
    #  object pointer
    #  @param params
    #  lmfit.Parameters instance \n
    #  Instance of the lmfit.Parameters class.
    #  All model parameters contained here.  
    #  This can be created using alp_params.soliton.params.
    #  @param r
    #  ndarray[N] \n
    #  Numpy array of N radii values in units of kpc.
    #  @returns 
    #  ndarray[N] (for single flavored models) or ndarray[2,N] (for double flavored models) \n
    #  Numpy array of N soliton mass values (for single flavored models) 
    #  or numpy array of [2,N] mass values (for double flavored models) 
    #  in units of solar mass.
    def mass_sol_init(self,params,r):
        rc_all=self.rc(params)*kpcTOGeV
        rhoc_all=self.rhoc(params)*msun/kpcTOGeV**3
        xc_all=self.xc(params,r)
        if self.model=='psi_single':
            rc_1=rc_all
            rhoc_1=rhoc_all
            x_1=xc_all
        else:
            rc_1=rc_all[0]
            rc_2=rc_all[1]
            rhoc_1=rhoc_all[0]
            rhoc_2=rhoc_all[1]
            x_1=xc_all[0]
            x_2=xc_all[1]
        def Minit(x,rhoc,rc):
            Min=(4*np.pi*rhoc*rc**3*(1/(10.989 + x**2)**7 * (-3.42652*1e6*x + 4.37168*1e6*x**3 
                + 56036*x**5 + 75545.6*x**7 + 4433.16*x**9 + 142.55*x**11 + 1.94581*x**13 
                + (1.13588*1e7 + 7.23555*1e6*x**2 + 1.97531*1e6*x**4 + 299588*x**6 + 27262.5*x**8 
                + 1488.53*x**10 + 45.1522*x**12 + 0.586978*x**14)*np.arctan(0.301662*x))))/msun
            return Min
        if self.matched==False:
            Min_1=Minit(x_1,rhoc_1,rc_1)
            if self.model=='psi_multi':
                Min_2=Minit(x_2,rhoc_2,rc_2)
            else:
                pass
        else:
            Minall_init_11=np.heaviside(3-x_1,0.5)
            Minall_init_21=np.heaviside(x_1-3,0.5)
            Min_1=Minall_init_11*Minit(x_1,rhoc_1,rc_1)+Minall_init_21*Minit(3,rhoc_1,rc_1)
            if self.model=='psi_multi':
                Minall_init_12=np.heaviside(3-x_2,0.5)
                Minall_init_22=np.heaviside(x_2-3,0.5)
                Min_2=Minall_init_12*Minit(x_2,rhoc_2,rc_2)+Minall_init_22*Minit(3,rhoc_2,rc_2)
            else:
                pass
        if self.model=='psi_single':
            Min_fin=Min_1
        else:
            Min_fin=np.asarray([Min_1,Min_2])
        return Min_fin

## The class containing mass functions for the ULDM halo models.
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
    #  @param fit_dict_in (optional)
    #  constants.fitting_dict instance \n
    #  Instance of the constants.fitting_dict class.
    #  Contains all the necessary rules and values to be used in fitting.
    def __init__(self,model,fit_dict_in=fitting_dict_in):

        ## @var model
        #  str \n
        #  Model to assume for DM halo.
        #  @var fit_dict_in
        #  constants.fitting_dict instance \n
        #  Instance of the constants.fitting_dict class.
        #  @var args_opts
        #  dictionary \n
        #  Dictionary of rules to be assumed during fitting
        #  Instance of constants.fitting_dict.args_opts
        #  @var matched
        #  bool \n
        #  Denotes how to combine the soliton and outer halo.
        #  Equal constants.fitting_dict.sol_match
        #  @var mfree
        #  float \n
        #  Denotes how to treat soliton particle mass in fitting procedure.
        #  Equal to constants.fitting_dict.sol_mfree
        #  @var m22
        #  float \n
        #  Soliton particle mass
        #  in units of \f$10^{-22} \, \mbox{eV}\f$.
        #  @var m22_2
        #  float \n
        #  Soliton particle mass two
        #  in units of \f$10^{-22} \, \mbox{eV}\f$.
        #  @var cdmhalo
        #  float \n
        #  Denotes which CDM profile to use for outer halo in ULDM galactic structure.
        #  Equal to constants.fitting_dict.sol_cdmhalo
        #  @var base_funcs_in
        #  alp_funcs.base_funcs instance \n
        #  Instance of the alp_funcs.base_funcs class
        #  @var halo_init
        #  cdm_funcs.halo instance \n
        #  Instance of the cdm_funcs.halo class
        self.model=model
        self.fit_dict_in=fit_dict_in
        self.args_opts=self.fit_dict_in.args_opts()
        self.matched=self.args_opts['soliton']['matched']
        self.mfree=self.args_opts['soliton']['mfree']
        self.m22=self.fit_dict_in.sol_m22
        if self.model=='psi_multi':
            self.m22_2=self.fit_dict_in.sol_m22_2
        else:
            pass
        self.cdmhalo=self.args_opts['soliton']['cdm_halo']
        self.base_funcs_in=base_funcs(self.model,fit_dict_in=self.fit_dict_in)        
        self.halo_init=cdm_funcs.halo(self.cdmhalo,fit_dict_in=self.fit_dict_in)

    ## Define the mass profile of the ULDM galactic structure.
    #  This defines the mass of the ULDM galactic structure (soliton and outer halo) at a given radius
    #  given the model parameters.
    #  @param self
    #  object pointer
    #  @param params
    #  lmfit.Parameters instance \n
    #  Instance of the lmfit.Parameters class.
    #  All model parameters contained here.  
    #  This can be created using cdm_params.halo.params.
    #  @param r
    #  ndarray[N] \n
    #  Numpy array of N radii values in units of kpc.
    #  @returns 
    #  ndarray[N] \n
    #  Numpy array of N mass values in units of solar mass.
    def mass(self,params,r):
        def MCDM_in(params,r,mass_num):
            self.fit_dict_in.sol_mass_ind=mass_num
            self.halo_init=cdm_funcs.halo(self.cdmhalo,fit_dict_in=self.fit_dict_in)
            MCDM=self.halo_init.mass(params,r)
            return MCDM
        if self.matched==False:
            MCDM_1=MCDM_in(params,r,0)
            if self.model=='psi_multi':
                MCDM_2=MCDM_in(params,r,1)
            else:
                pass
        else:
            xc_all=self.base_funcs_in.xc(params,r)
            if self.model=='psi_single':
                x_1=xc_all
                MCDM_1_init=np.heaviside(x_1-3,1)
                MCDM_1=MCDM_1_init*MCDM_in(params,r,0)
            else:
                x_1=xc_all[0]
                x_2=xc_all[1]
                MCDM_1_init=np.heaviside(x_1-3,1)
                MCDM_1=MCDM_1_init*MCDM_in(params,r,0)
                MCDM_2_init=np.heaviside(x_2-3,1)
                MCDM_2=MCDM_2_init*MCDM_in(params,r,1)
        mass_sol_all=self.base_funcs_in.mass_sol_init(params,r)
        if self.model=='psi_single':
            Minsol_1=mass_sol_all
            Minall=Minsol_1+MCDM_1
        else:
            Minsol_1=mass_sol_all[0]
            Minsol_2=mass_sol_all[1]
            Minall=Minsol_1+Minsol_2+MCDM_1+MCDM_2
        return np.asarray(Minall)

    ## Define the total mass of the outer halo.
    #  This defines the total mass of the outer halo assuming the model parameters.
    #  @param self
    #  object pointer
    #  @param params
    #  lmfit.Parameters instance \n
    #  Instance of the lmfit.Parameters class.
    #  All model parameters contained here.  
    #  This can be created using alp_params.soliton.params.
    #  @returns 
    #  float \n
    #  Total mass of the outer halo in units of solar mass.
    def Mhalo(self,params):
        rc_all=self.base_funcs_in.rc(params)
        if self.model=='psi_single':
            rc_1=rc_all
        else:
            rc_1=rc_all[0]
            rc_2=rc_all[1]
        def mhaloin(params,rc,mass_num):
            self.fit_dict_in.sol_mass_ind=mass_num
            Mmatchin=self.halo_init.mass(params,3*rc)
            Mhaloin=self.halo_init.Mvir(params)
            if self.matched==True:
                Minall=Mhaloin-Mmatchin
            else:
                Minall=Mhaloin
            return Minall
        if self.model=='psi_single':
            Min_fin=mhaloin(params,rc_1,0)
        else:
            Min_fin=np.asarray([mhaloin(params,rc_1,0),mhaloin(params,rc_2,1)])
        return Min_fin

    ## Define the total galactic DM halo mass.
    #  This defines the total mass of the galactic DM given the model parameters.
    #  @param self
    #  object pointer
    #  @param params
    #  lmfit.Parameters instance \n
    #  Instance of the lmfit.Parameters class.
    #  All model parameters contained here.  
    #  This can be created using cdm_params.halo.params.
    #  @returns 
    #  float \n
    #  Total mass of the galactic DM in units of solar mass.
    def Mvir(self,params):
        if self.model=='psi_single':
            msol_1=self.base_funcs_in.msol(params)
            Mhalo_1=self.Mhalo(params)
            Minall=msol_1+Mhalo_1
        else:
            msol_1=self.base_funcs_in.msol(params)[0]
            msol_2=self.base_funcs_in.msol(params)[1]
            Mhalo_1=self.Mhalo(params)[0]
            Mhalo_2=self.Mhalo(params)[1]
            Minall=msol_1+msol_2+Mhalo_1+Mhalo_2
        return Minall