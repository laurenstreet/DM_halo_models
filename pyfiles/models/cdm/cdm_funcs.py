## @package cdm_funcs
#  The package containing the functions for the CDM halo models.

import numpy as np
import uncertainties as un
import pyfiles.data_models.constants as constants
from scipy.special import hyp2f1
from scipy.special import gammainc
from scipy.special import gamma

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

## The class containing base functions needed to describe the CDM halo models.
class base_funcs:

    ## Define the constructor of the base_funcs class.
    #  This defines the constructor of the base_funcs class.
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
        #  @var mass_num
        #  int \n
        #  Used to differentiate between soliton 1 and soliton 2 in double flavored ULDM models.
        #  Outer CDM halo is halo 1 if mass_num = 0 and is halo 2 if mass_num = 1.
        self.model=model
        self.fit_dict_in=fit_dict_in
        self.args_opts=self.fit_dict_in.args_opts()
        self.mass_num=self.args_opts['soliton']['mass_ind']

    ## Define the mass fraction for the DC14 model.
    #  This defines the stellar to DM mass fraction for the DC14 model assuming the given model parameters.
    #  The stellar to DM mass fraction is given by,
    #  \f[ X = \log_{10}\left(M_*/M_{\mbox{halo}}\right), \f]
    #  where,
    #  \f[ M_* \approx \left( \tilde{\Upsilon}_{\mbox{disk}} + \tilde{\Upsilon}_{\mbox{bulge}} \right) L, \f]
    #  with \f$ L = \f$ Luminosity in galaxy.galaxy.data, \f$ M_{\mbox{halo}} = \f$ cdm_funcs.halo.Mvir, and 
    #  \f$ \tilde{\Upsilon}_{\mbox{disk}} = \f$ MLD and \f$ \tilde{\Upsilon}_{\mbox{bulge}} = \f$ MLB in 
    #  cdm_params.halo.params.
    #  @param self
    #  object pointer
    #  @param params
    #  lmfit.Parameters instance \n
    #  Instance of the lmfit.Parameters class.
    #  All model parameters contained here.  
    #  This can be created using cdm_params.halo.params.
    #  @returns 
    #  float \n
    #  The stellar to DM mass fraction in the DC14 model.
    def mass_frac_dc14(self,params):
        mstar=params['mstar']
        if self.mass_num==0:
            if self.args_opts['fitting_routine']['DC14_check']==True:
                V200=10**(params['v200'])*kmTOGeV/sTOGeV
            else:
                V200=params['v200']*kmTOGeV/sTOGeV
        else:
            if self.args_opts['fitting_routine']['DC14_check']==True:
                V200=10**(params['v200_2'])*kmTOGeV/sTOGeV
            else:
                V200=params['v200_2']*kmTOGeV/sTOGeV
        mhalo=np.sqrt(3/(2*np.pi*rhocrit))*MP**3*(V200)**3/20/msun/(10**9)
        Xin=np.log10(mstar/mhalo)
        return Xin
    
    ## Define the halo profile variable rc.
    #  This defines the halo profile variable rc for the given model parameters.
    #  The halo profile variable rc is given by,
    #  \f[ r_c = \sqrt{\frac{3}{2 \pi \, \rho_{\mbox{crit}}}} 
    #  \frac{M_P \, V_{200}}{20 \, c_{200}}, \f]
    #  where \f$ \rho_{\mbox{crit}} = \f$ rhocrit and \f$ M_P \f$ = MP in constants.standard,
    #  while \f$ V_{200} = \f$ v200 and \f$ c_{200} \f$ = c200 in cdm_params.halo.params.
    #  @param self
    #  object pointer
    #  @param params
    #  lmfit.Parameters instance \n
    #  Instance of the lmfit.Parameters class.
    #  All model parameters contained here.  
    #  This can be created using cdm_params.halo.params.
    #  @returns 
    #  float \n
    #  The halo profile variable rc in units of kpc.
    def rc(self,params):
        if self.mass_num==0:
            if self.args_opts['fitting_routine']['DC14_check']==True:
                V200=10**(params['v200'])*kmTOGeV/sTOGeV
                c200=10**(params['c200'])
            else:
                V200=params['v200']*kmTOGeV/sTOGeV
                c200=params['c200']
        else:
            if self.args_opts['fitting_routine']['DC14_check']==True:
                V200=10**(params['v200_2'])*kmTOGeV/sTOGeV
                c200=10**(params['c200_2'])
            else:
                V200=params['v200_2']*kmTOGeV/sTOGeV
                c200=params['c200_2']
        if self.model=='DC14':
            X=self.mass_frac_dc14(params)
            c200_fac=1+np.exp(0.0001*(3.4*(X+4.5)))
            c200=c200*c200_fac
        rcin=MP*np.sqrt(3/(2*np.pi*rhocrit))*V200/(20*c200)/kpcTOGeV
        return rcin
    
    ## Define the halo profile variable rhoc.
    #  This defines the halo profile variable rhoc for the given model parameters.
    #  The halo profile variables rhoc is given by,
    #  \f[ \rho_c = \frac{M_{200}}{4 \pi \, r_c^3 
    #  \left[ \ln\left( 1+c_{200} \right) - \frac{c_{200}}{1+c_{200}}\right]}, \f]
    #  where \f$ r_c = \f$ cdm_funcs.base_funcs.rc, \f$ M_{200} = \f$ cdm_funcs.halo.Mvir,
    #  and \f$ c_{200} = \f$ c200 in cdm_params.halo.params. 
    #  @param self
    #  object pointer
    #  @param params
    #  lmfit.Parameters instance \n
    #  Instance of the lmfit.Parameters class.
    #  All model parameters contained here.  
    #  This can be created using cdm_params.halo.params.
    #  @returns 
    #  float \n
    #  The halo profile variable rhoc 
    #  in units of \f$M_{\odot}/\mbox{kpc}^3\f$.
    def rhoc(self,params):
        if self.mass_num==0:
            if self.args_opts['fitting_routine']['DC14_check']==True:
                c200=10**(params['c200'])
            else:
                c200=params['c200']
        else:
            if self.args_opts['fitting_routine']['DC14_check']==True:
                c200=10**(params['c200_2'])
            else:
                c200=params['c200_2']
        if self.model=='DC14':
            X=self.mass_frac_dc14(params)
            c200_fac=1+np.exp(0.0001*(3.4*(X+4.5)))
            c200=c200*c200_fac
        rhocin=200*c200**3*rhocrit/(3*(un.unumpy.log(1+c200)-c200/(1+c200)))/msun*kpcTOGeV**3
        return rhocin

    ## Define the halo profile variable xc.
    #  This defines the halo profile variable xc for the given model parameters.
    #  The halo profile variable xc is given by,
    #  \f[ x_c = r/r_c, \f]
    #  where \f$ r_c = \f$ cdm_funcs.base_funcs.rc.
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
    #  Numpy array of N halo profile variables xc.
    def xc(self,params,r):
        rc=self.rc(params)
        x=r/rc
        return x

## The class containing mass functions for the CDM halo models.
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
        #  @var mass_num
        #  int \n
        #  Used to differentiate between soliton 1 and soliton 2 in double flavored ULDM models.
        #  Outer CDM halo is halo 1 if mass_num = 0 and is halo 2 if mass_num = 1.
        #  @var base_funcs_in
        #  cdm_funcs.base_funcs instance \n
        #  Instance of the cdm_funcs.base_funcs class
        self.model=model
        self.fit_dict_in=fit_dict_in
        self.args_opts=self.fit_dict_in.args_opts()
        self.mass_num=self.args_opts['soliton']['mass_ind']
        self.base_funcs_in=base_funcs(self.model,fit_dict_in=self.fit_dict_in)

    ## Define the mass profile of the CDM galactic structure.
    #  This defines the mass of the CDM galactic structure at a given radius
    #  given the model parameters.  The mass profile depends on the model assumed.
    #  The mass profile for each model is given by :
    #  - Burkert,
    #       \f[ M_{\mbox{Burkert}}(r) = \pi \, \rho_c \, r_c^3 
    #       \left\{ -2 \arctan(x_c(r)) + \log\left[ \left(1+x_c(r)\right)^2 \left(1+x_c(r)^2\right) \right] \right\}, \f]
    #  - DC14,
    #       \f[ M_{\mbox{DC}14}(r) = 4 \pi \, \rho_c \, r_c^3 \, x_c(r)^{3-\gamma} \, 
    #       {}_2\mbox{F}_1\left(\frac{3-\gamma}{\alpha}, \frac{\beta-\gamma}{\alpha}, \frac{3+\alpha-\gamma}{\alpha}, -x_c(r)^\alpha \right)
    #       (3-\gamma)^{-1},\f]
    #       where,
    #       \f[ \alpha = 2.94 - 
    #       \log_{10}\left[ \left(10^{X+2.33}\right)^{-1.08}
    #       + \left(10^{X+2.33}\right)^{2.99}\right], \f]
    #       \f[ \beta = 4.23 + 1.34 X + 0.26 X^2, \f]
    #       \f[ \gamma = -0.06 - 
    #       \log_{10}\left[ \left(10^{X+2.56}\right)^{-0.68}
    #       + 10^{X+2.56}\right]. \f]
    #       Here, \f$ X = \f$ cdm_funcs.base_funcs.mass_frac_dc14,
    #       and \f$ {}_2\mbox{F}_1 = \f$ Gaussian hypergeometric function.
    #  - Einasto,
    #       \f[ M_{\mbox{Einasto}}(r) = 4 \pi \, \rho_c \, r_c^3 \exp\left(2/\alpha\right) \left(2/\alpha\right)^{-3/\alpha} \,
    #       \Gamma\left(3/\alpha,2 x_c^{\alpha}/\alpha\right)/\alpha, \f]
    #       where \f$ \alpha = \f$ alpha in cdm_params.halo.params 
    #       and \f$ \Gamma\left(a,x\right) = \f$ incomplete Gamma function.
    #  - NFW,
    #       \f[ M_{\mbox{NFW}}(r) = 4 \pi \, \rho_c \, r_c^3 \left[\ln(1+x_c(r))-\frac{x_c(r)}{1+x_c(r)}\right]. \f]
    #  For each of the above masses, \f$ \rho_c = \f$ cdm_funcs.base_funcs.rhoc,
    #  \f$ r_c = \f$ cdm_funcs.base_funcs.rc, and \f$ x_c(r) = \f$ cdm_funcs.base_funcs.xc.
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
        rhoc=self.base_funcs_in.rhoc(params)
        rc=self.base_funcs_in.rc(params)
        x=self.base_funcs_in.xc(params,r)
        if self.model=='Burkert':
            Min=np.pi*rhoc*rc**3*(-2*np.arctan(x)+np.log((1+x)**2*(1+x**2)))
        elif self.model=='DC14':
            X=self.base_funcs_in.mass_frac_dc14(params)
            exa=X+2.33
            exa1=10**exa
            exg=X+2.56
            exg1=10**exg
            a=2.94-np.log10(exa1**(-1.08)+exa1**(2.99))
            b=4.23+1.34*X+0.26*X**2
            g=-0.06-np.log10(exg1**(-0.68)+exg1)
            hypgeo=hyp2f1((3-g)/a,(b-g)/a,(3+a-g)/a,-x**a)
            Min=4*np.pi*rhoc*rc**3*x**(3-g)*hypgeo/(3-g)
        elif self.model=='Einasto':
            if self.mass_num==0:
                alpha=params['alpha']    
            else:
                alpha=params['alpha_2']
            if ((type(alpha)==np.ndarray and np.isreal(alpha)==True)
                or type(alpha)!=np.ndarray and np.isreal(alpha.value)==True and np.isfinite((alpha.value).real)==True):
                if type(alpha)==np.ndarray:
                    alpha=alpha.real
                else:
                    alpha=(alpha.value).real
                gamma_ex=gamma(3/alpha)
                gammainc_ex=gammainc(3/alpha,2/alpha*x**alpha)
                piece0=1/alpha*np.exp(2/alpha)*(2/alpha)**(-3/alpha)*gamma_ex*gammainc_ex
                Min=4*np.pi*rhoc*rc**3*piece0
            else:
                Min=x*np.inf
        elif self.model=='NFW':
            Min=4*np.pi*rhoc*rc**3*(np.log(1+x)-x/(1+x))
        return np.asarray(Min)

    ## Define the total galactic DM halo mass.
    #  This defines the total mass of the galactic DM given the model parameters.
    #  The total galactic DM mass is given by,
    #  \f[ M_{200} = \sqrt{\frac{3}{2 \pi \rho_{\mbox{crit}}}} \frac{M_P^3 V_{200}^3}{20}, \f]
    #  where \f$ \rho_{\mbox{crit}} = \f$ rhocrit and \f$ M_P \f$ = MP in constants.standard,
    #  while \f$ V_{200} = \f$ v200 in cdm_params.halo.params. 
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
        if self.mass_num==0:
            if self.args_opts['fitting_routine']['DC14_check']==True:
                V200=10**(params['v200'])*kmTOGeV/sTOGeV
            else:
                V200=params['v200']*kmTOGeV/sTOGeV
        else:
            if self.args_opts['fitting_routine']['DC14_check']==True:
                V200=10**(params['v200_2'])*kmTOGeV/sTOGeV
            else:
                V200=params['v200_2']*kmTOGeV/sTOGeV
        Mvirin=np.sqrt(3/(2*np.pi*rhocrit))*MP**3*(V200)**3/20/msun
        return Mvirin