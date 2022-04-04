## @package constants
#  The package containing constants, fitting routine definitions, and fitting routine parameter values.

import numpy as np

## The dictionary containing standard constants.
#  This defines the dictionary containing standard constants needed during fitting procedure.
#  @returns
#  dictionary \n 
#  Dictionary of standard constants:
#  - kmTOGeV in units \f$[\mbox{km}^{-1} \mbox{GeV}^{-1}] \f$ = conversion from kilometer to GeV
#  - sTOGeV in units \f$[\mbox{s}^{-1} \, \mbox{GeV}^{-1}]\f$ = conversion from second to GeV
#  - kpcTOGeV in units \f$[\mbox{kpc}^{-1} \, \mbox{GeV}^{-1}]\f$ = conversion from kpc to GeV
#  - msun in units \f$[\mbox{GeV}]\f$ = mass of sun in GeV
#  - MP in units \f$[\mbox{GeV}]\f$ = Planck mass in GeV
#  - rhocrit in units \f$[\mbox{GeV}^4]\f$ = critical density of universe in \f$\mbox{GeV}^4\f$
#  - MOND_gdag in units \f$[\mbox{m} \, \mbox{s}^{-2}]\f$ = gravitational acceleration constant
#    for gravitational acceleration relation in MOND theory
def standard():
        standard_dict={
                'kmTOGeV':5.076142131979695e+18,
                'sTOGeV':1.5197568389057748e+24,
                'kpcTOGeV':1.5663334542856025e+35,
                'msun':1.115747188908623e+57,
                'MP':1.22*10**19,
                'rhocrit':4.3052429461571337e-47,
                'MOND_gdag':1.2e-10
        }
        return standard_dict

## The class containing dictionaries of fitting variables.
class fitting_dict:

        ## Define the constructor of the fitting_dict class.
        #  This defines the constructor of the fitting_dict class.
        #  @param self
        #  object pointer
        #  @param fit_routine (optional)
        #  str \n
        #  Denotes the fitting routine to use.  Can be equal to:
        #  - 'uni_priors' : uniform priors on all parameters (used for main results of paper)
        #  - 'c200_priors_check' : used to check affect of changing prior ranges for c200
        #  - 'v200_priors_check' : used to check affect of changing prior ranges for v200
        #  - 'MLd_priors_check' : used to check affect of changing prior ranges for MLd
        #  - 'MLb_priors_check' : used to check affect of changing prior ranges for MLb
        #  - 'CDM_check' : used to obtain results to be compared with 
        #    Pengfei Li et al 2020 ApJS 247 31 : 
        #    https://doi.org/10.3847/1538-4365/ab700e
        #  - 'DC14_check' : used to obtain results to be compared with 
        #    Monthly Notices of the Royal Astronomical Society, Volume 466, Issue 2, April 2017, Pages 1648–1668 :  
        #    https://doi.org/10.1093/mnras/stw3101
        #  - 'Einasto_check' : used to obtain results to be compared with 
        #    Nicolas Loizeau and Glennys R. Farrar 2021 ApJL 920 L10 : 
        #    https://doi.org/10.3847/2041-8213/ac1bb7
        #  @param sol_match (optional)
        #  bool \n
        #  Denotes how to combine the soliton and outer halo.  If equal to:
        #  - False : soliton and CDM halo to be summed
        #  - True : soliton and CDM halo to be matched
        #  @param sol_mfree (optional)
        #  bool \n
        #  Denotes how to treat soliton particle mass in fitting procedure.  
        #  If equal to:
        #  - False : soliton particle mass set to be fixed
        #  - True : soliton particle mass to be free
        #  @param sol_cdm_halo (optional)
        #  str \n
        #  Denotes which CDM profile to use for outer halo in ULDM galactic structure.  
        #  Can be equal to:
        #  - 'Burkert'
        #  - 'DC14'
        #  - 'Einasto'
        #  - 'NFW'
        #  @param sol_m22 (optional)
        #  float \n
        #  Soliton particle mass (for single flavored models) 
        #  and soliton 1 particle mass (for double flavored models) in units of \f$10^{-22} \, \mbox{eV}\f$.  
        #  Only used for case in which soliton particle mass is fixed (i.e. constants.fitting_dict.sol_mfree = False)
        #  @param sol_m22_2 (optional)
        #  float \n
        #  Soliton 2 particle mass (for double flavored models) in units of \f$10^{-22} \, \mbox{eV}\f$.  
        #  Only used for case in which soliton particle mass is fixed (i.e. constants.fitting_dict.sol_mfree = False)
        #  @param sol_m22_tab (optional)
        #  ndarray[N] \n
        #  Numpy array of soliton particle masses (for single flavored, matched models) 
        #  and soliton 1 particle masses (for double flavored, matched models) in units of \f$10^{-22} \, \mbox{eV}\f$.  
        #  Only used for case in which soliton particle mass is fixed (i.e. constants.fitting_dict.sol_mfree = False)
        #  @param sol_m22_tab_prime (optional)
        #  ndarray[N] \n
        #  Numpy array of soliton particle masses (for single flavored, summed models) 
        #  and soliton 1 particle masses (for double flavored, summed models) in units of \f$10^{-22} \, \mbox{eV}\f$.  
        #  Only used for case in which soliton particle mass is fixed (i.e. constants.fitting_dict.sol_mfree = False)
        #  @param sol_m22_2_tab (optional)
        #  ndarray[N] \n 
        #  Numpy array of soliton 2 particle masses (for double flavored, matched models) in units of \f$10^{-22} \, \mbox{eV}\f$.  
        #  Only used for case in which soliton particle mass is fixed (i.e. constants.fitting_dict.sol_mfree = False)   
        #  @param sol_m22_2_tab_prime (optional)
        #  ndarray[N] \n 
        #  Numpy array of soliton 2 particle masses (for double flavored, summed models) in units of \f$10^{-22} \, \mbox{eV}\f$.  
        #  Only used for case in which soliton particle mass is fixed (i.e. constants.fitting_dict.sol_mfree = False) 
        #  @param sol_mass_ind (optional)
        #  int \n
        #  Used to differentiate between soliton 1 and soliton 2 in double flavored models.  
        #  Can be equal to 0 (to denote soliton 1) or 1 (to denote soliton 2)
        def __init__(self, fit_routine = 'uni_priors', 
                        sol_match = False, 
                        sol_mfree = False, 
                        sol_cdm_halo = 'Einasto', 
                        sol_m22 = 10**(1.5), 
                        sol_m22_2 = 1, 
                        sol_m22_tab = np.float_power(10, (np.arange(0, 2, 0.15))),
                        sol_m22_tab_prime = np.float_power(10, (np.arange(-3, 4, 0.5))),
                        sol_m22_2_tab = np.float_power(10, (np.arange(0, 2, 0.15))), 
                        sol_m22_2_tab_prime = np.float_power(10, (np.arange(-3, 4, 0.5))),
                        sol_mass_ind = 0):

                ## @var fit_routine
                #  str \n
                #  Denotes the fitting routine to use.
                #  @var sol_match
                #  bool \n
                #  Denotes how to combine the soliton and outer halo.
                #  @var sol_mfree
                #  bool \n
                #  Denotes how to treat soliton particle mass in fitting procedure.
                #  @var sol_cdm_halo
                #  str \n
                #  Denotes which CDM profile to use for outer halo in ULDM galactic structure.
                #  @var sol_m22
                #  float \n
                #  Soliton particle mass (for single flavored models) 
                #  and soliton 1 particle mass (for double flavored models) in units of \f$10^{-22} \, \mbox{eV}\f$.
                #  @var sol_m22_2
                #  float \n
                #  Soliton 2 particle mass (for double flavored models) in units of \f$10^{-22} \, \mbox{eV}\f$.
                #  @var sol_m22_tab
                #  ndarray[N] \n
                #  Numpy array of soliton particle masses (for single flavored, matched models) 
                #  and soliton 1 particle masses (for double flavored, matched models) in units of \f$10^{-22} \, \mbox{eV}\f$.
                #  @var sol_m22_tab_prime
                #  ndarray[N] \n
                #  Numpy array of soliton particle masses (for single flavored, summed models) 
                #  and soliton 1 particle masses (for double flavored, summed models) in units of \f$10^{-22} \, \mbox{eV}\f$.
                #  @var sol_m22_2_tab
                #  ndarray[N] \n 
                #  Numpy array of soliton 2 particle masses (for double flavored, matched models) in units of \f$10^{-22} \, \mbox{eV}\f$.
                #  @var sol_m22_2_tab_prime
                #  ndarray[N] \n 
                #  Numpy array of soliton 2 particle masses (for double flavored, summed models) in units of \f$10^{-22} \, \mbox{eV}\f$.
                #  @var sol_mass_ind
                #  int \n
                #  Used to differentiate between soliton 1 and soliton 2 in double flavored models.
                self.fit_routine=fit_routine
                self.sol_match=sol_match  
                self.sol_mfree=sol_mfree                
                self.sol_cdm_halo=sol_cdm_halo
                self.sol_m22=sol_m22
                self.sol_m22_2=sol_m22_2
                self.sol_m22_tab=sol_m22_tab
                self.sol_m22_tab_prime=sol_m22_tab_prime
                self.sol_m22_2_tab=sol_m22_2_tab  
                self.sol_m22_2_tab_prime=sol_m22_2_tab_prime  
                self.sol_mass_ind=sol_mass_ind

        ## Define the fitting rules.
        #  This defines the rules to be assumed during fitting. \n 
        #  Many prior cases and fitting routines to compare to previous studies are taken into account.
        #  @param self
        #  object pointer
        #  @returns
        #  dictionary \n
        #  Dictionary of rules to be assumed during fitting:
        #  - 'fitting_routine' : Dictionary of possible fitting routines \n 
        #    Each of the following fitting routines can be equal to true or false.
        #    By default, the only fitting routine that is set to true is the one equal to fit_routine.
        #       - 'uni_priors' : bool
        #       - 'c200_priors_check' : bool
        #       - 'v200_priors_check' : bool
        #       - 'MLd_priors_check' : bool
        #       - 'MLb_priors_check' : bool
        #       - 'CDM_check' : bool
        #       - 'DC14_check' : bool
        #       - 'Einasto_check' : bool
        #  - 'soliton' : Dictionary of soliton variables
        #       - 'matched' : bool (equal to constants.fitting_dict.sol_match)
        #       - 'mfree' : bool (equal to constants.fitting_dict.sol_mfree)
        #       - 'cdm_halo' : str (equal to constants.fitting_dict.sol_cdm_halo)
        #       - 'm22' : float (equal to constants.fitting_dict.sol_m22)
        #       - 'm22_2' : float (equal to constants.fitting_dict.sol_m22_2)
        #       - 'm22_tab' : ndarray[N] (equal to constants.fitting_dict.sol_m22_tab)
        #       - 'm22_tab_prime' : ndarray[N] (equal to constants.fitting_dict.sol_m22_tab_prime)
        #       - 'm22_2_tab' : ndarray[N] (equal to constants.fitting_dict.sol_m22_2_tab)
        #       - 'm22_2_tab_prime' : ndarray[N] (equal to constants.fitting_dict.sol_m22_2_tab_prime)
        #       - 'mass_ind' : int (equal to constants.fitting_dict.sol_mass_ind)
        def args_opts(self):
                args_opts_in={
                        'fitting_routine':{},
                        'soliton':{}}
                fit_key=[
                        'uni_priors',
                        'c200_priors_check',
                        'v200_priors_check',
                        'MLd_priors_check',
                        'MLb_priors_check',
                        'CDM_check',
                        'DC14_check',
                        'Einasto_check'        
                ]
                for key in fit_key:
                        args_opts_in['fitting_routine'][key]=False
                args_opts_in['fitting_routine'][self.fit_routine]=True

                args_opts_in['soliton']['matched']=self.sol_match
                args_opts_in['soliton']['mfree']=self.sol_mfree
                args_opts_in['soliton']['cdm_halo']=self.sol_cdm_halo
                args_opts_in['soliton']['m22']=self.sol_m22
                args_opts_in['soliton']['m22_2']=self.sol_m22_2
                args_opts_in['soliton']['m22_tab']=self.sol_m22_tab
                args_opts_in['soliton']['m22_tab_prime']=self.sol_m22_tab_prime
                args_opts_in['soliton']['m22_2_tab']=self.sol_m22_2_tab
                args_opts_in['soliton']['m22_2_tab_prime']=self.sol_m22_2_tab_prime
                args_opts_in['soliton']['mass_ind']=self.sol_mass_ind
                return args_opts_in

        ## Define the fitting values.
        #  This defines the values for various variables used during the fitting procedure. \n 
        #  Many prior cases and fitting routines to compare to previous studies are taken into account.
        #  Values are setup to be utilized in lmfit.Parameters class.
        #  @param self
        #  object pointer
        #  @returns
        #  dictionary \n
        #  Possible dictionary of variables values to be used during fitting. \n
        #  Variables values are setup in lists with format [starting value,min,max]  
        #  Dictionary to be used depends on value of fit_routine. \n 
        #  If:
        #  - fit_routine = 'uni_priors', resulting dictionary is:
        #       - 'CDM' : values for variables describing CDM halos
        #               - 'c200' : \f$[3,\, 1,\, 100]\f$
        #               - 'v200' : \f$[100,\, 1,\, 1000]\f$ in units of \f$\mbox{km} \, \mbox{s}^{-1}\f$
        #               - 'v200_fac' : \f$[1.5,\, 1,\, \infty]\f$
        #               - 'MLd' : \f$[0.5,\, 0.01,\, 5]\f$ in units of \f$M_{\odot}/L_{\odot}\f$
        #               - 'MLb' : \f$[0.7,\, 0.01,\, 5]\f$ in units of \f$M_{\odot}/L_{\odot}\f$
        #               - 'alpha' : \f$[0.16,\, -\infty,\, \infty]\f$ (only used for Einasto profile)
        #       - 'soliton' : values for variables describing soliton
        #               - 'Msol' : \f$[10^9,\, 10^{4.5},\, 10^{12}]\f$ in units of \f$M_{\odot}\f$
        #               - 'Msol_2' : \f$[10^9,\, 10^{4.5},\, 10^{12}]\f$ in units of \f$M_{\odot}\f$
        #               - 'm22' : \f$[1,\, 10^{-3},\, 10^3]\f$ in units of \f$10^{-22} \, \mbox{eV}\f$
        #               - 'm22_2' : \f$[1,\, 10^{-3},\, 10^3]\f$ in units of \f$10^{-22} \, \mbox{eV}\f$
        #  - fit_routine = 'c200_priors_check', resulting dictionary is same as 'uni_priors' with following changes:
        #       - 'CDM' :
        #               - 'c200' : \f$[3,\, 0,\, \infty]\f$
        #  - fit_routine = 'v200_priors_check', resulting dictionary is same as 'uni_priors' with following changes:
        #       - 'CDM' :
        #               - 'v200' : \f$[100,\, 0,\, \infty]\f$ in units of \f$\mbox{km} \, \mbox{s}^{-1}\f$
        #  - fit_routine = 'MLd_priors_check', resulting dictionary is same as 'uni_priors' with following changes:
        #       - 'CDM' :
        #               - 'MLd' : \f$[0.5,\, 0,\, \infty]\f$ in units of \f$M_{\odot}/L_{\odot}\f$
        #  - fit_routine = 'MLb_priors_check', resulting dictionary is same as 'uni_priors' with following changes:
        #       - 'CDM' :
        #               - 'MLb' : \f$[0.7,\, 0,\, \infty]\f$ in units of \f$M_{\odot}/L_{\odot}\f$
        #  - fit_routine = 'CDM_check', resulting dictionary is same as 'uni_priors' with following changes:
        #       - 'CDM' :
        #               - 'c200' : \f$[3,\, 10^{-1},\, 10^3]\f$
        #               - 'v200' : \f$[100,\, 10,\, 500]\f$ in units of \f$\mbox{km} \, \mbox{s}^{-1}\f$
        #  - fit_routine = 'DC14_check', resulting dictionary is same as 'uni_priors' with following changes:
        #       - 'CDM' :
        #               - 'c200' : \f$[\log_{10}5,\, \log_{10}1,\, \log_{10}100]\f$
        #               - 'v200' : \f$[\log_{10}100,\, \log_{10}10,\, \log_{10}500]\f$ 
        #                 in units of \f$\log_{10}\left(\mbox{km} \, \mbox{s}^{-1}\right)\f$
        #               - 'v200_fac' : \f$[2,\, 1,\, \infty]\f$
        #               - 'MLd' : \f$[\log_{10}0.5,\, \log_{10}0.3,\, \log_{10}0.8]\f$ 
        #                 in units of \f$\log_{10}\left(M_{\odot}/L_{\odot}\right)\f$
        #  - fit_routine = 'Einasto_check', resulting dictionary is same as 'uni_priors' with following changes:
        #       - 'CDM' :
        #               - 'v200' : \f$[100,\, 1,\, 500]\f$ in units of \f$\mbox{km} \, \mbox{s}^{-1}\f$
        #               - 'alpha' : \f$[0.16,\, 10^{-3},\, 10]\f$
        def params_vals(self):
                params_vals_in={}
                fit_key=[
                        'uni_priors',
                        'c200_priors_check',
                        'v200_priors_check',
                        'MLd_priors_check',
                        'MLb_priors_check',
                        'CDM_check',
                        'DC14_check',
                        'Einasto_check'        
                ]
                for key in fit_key:
                        params_vals_in[key]={}
                        params_vals_in[key]['CDM']={}
                        params_vals_in[key]['soliton']={}
                        params_vals_in[key]['CDM']['c200']=[3,1,100]
                        params_vals_in[key]['CDM']['v200']=[100,1,1000]
                        params_vals_in[key]['CDM']['v200_fac']=[1.5,1,np.inf]
                        params_vals_in[key]['CDM']['MLd']=[0.5,0.01,5]
                        params_vals_in[key]['CDM']['MLb']=[0.7,0.01,5]
                        params_vals_in[key]['CDM']['alpha']=[0.16,-np.inf,np.inf]
                        params_vals_in[key]['soliton']['Msol']=[1e9,10**(4.5),1e12]
                        params_vals_in[key]['soliton']['Msol_2']=[1e9,10**(4.5),1e12]
                        params_vals_in[key]['soliton']['m22']=[1,1e-3,1e3]
                        params_vals_in[key]['soliton']['m22_2']=[1,1e-3,1e3]

                priors_check_key=[
                        'c200_priors_check',
                        'v200_priors_check',
                        'MLd_priors_check',
                        'MLb_priors_check'
                ]
                vars_check_key=[
                        'c200',
                        'v200',
                        'MLd',
                        'MLb'
                ]
                for i in range(len(priors_check_key)):
                        params_vals_in[priors_check_key[i]]['CDM'][vars_check_key[i]][1]=0
                        params_vals_in[priors_check_key[i]]['CDM'][vars_check_key[i]][2]=np.inf

                params_vals_in['CDM_check']['CDM']['c200']=[3,1e-1,1e3]
                params_vals_in['CDM_check']['CDM']['v200']=[100,10,500]

                params_vals_in['DC14_check']['CDM']['c200']=[np.log10(5),np.log10(1),np.log10(100)]
                params_vals_in['DC14_check']['CDM']['v200']=[np.log10(100),np.log10(10),np.log10(500)]
                params_vals_in['DC14_check']['CDM']['v200_fac']=[2,1,np.inf]   
                params_vals_in['DC14_check']['CDM']['MLd']=[np.log10(0.5),np.log10(0.3),np.log10(0.8)]

                params_vals_in['Einasto_check']['CDM']['v200']=[100,1,500]
                params_vals_in['Einasto_check']['CDM']['alpha']=[0.16,1e-3,10]

                params_vals_in=params_vals_in[self.fit_routine]
                return params_vals_in