## @package results
#  The package containing the procedures to obtain results for various cases.

import numpy as np
from uncertainties import unumpy
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import pyfiles.data_models.constants as constants
import pyfiles.data_models.galaxy as galaxy
import pyfiles.data_models.halo as halo
import pyfiles.fitting.model_fit as model_fit
import pyfiles.models.cdm.cdm_funcs as cdm_funcs
import pyfiles.models.alps.alp_funcs as alp_funcs
plt.style.use('pyfiles/data/presentation.mplstyle')

## @var fitting_dict_in
#  constants.fitting_dict instance \n
#  Instance of the constants.fitting_dict class using all default values
fitting_dict_in=constants.fitting_dict()

## The class to obtain fit results for all CDM halos analyzed.
#  This class can be used to obtain results for all CDM halos analyzed.  
#  120 galaxies in the SPARC catalog are used for CDM only fits,
#  93 galaxies used for ULDM fits with Einasto halo.
class results_CDM_all:

    ## Define the constructor of the results_CDM_all class.
    #  This defines the constructor of the results_CDM_all class.
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
    #  @param ULDM_fits (optional)
    #  bool \n
    #  True if performing fits for the ULDM models.
    #  @param fit_dict_in (optional)
    #  constants.fitting_dict instance \n
    #  Instance of the constants.fitting_dict class.
    #  Contains all the necessary rules and values to be used in fitting.
    def __init__(self,model,ULDM_fits=False,fit_dict_in=fitting_dict_in):

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
        #  True if performing ULDM fits and ULDM matched models
        #  @var ULDM_fits
        #  bool \n
        #  True if performing ULDM fits
        self.model=model
        self.fit_dict_in=fit_dict_in
        self.args_opts=self.fit_dict_in.args_opts()
        self.matched=self.args_opts['soliton']['matched']
        self.ULDM_fits=ULDM_fits

    ## Define the fit results.
    #  This defines the fit results for the assumed model for 120 galaxies in the SPARC catalog.
    #  @param self
    #  object pointer
    #  @returns
    #  dictionary \n
    #  Dictionary of resulting fit parameters.  Results are as follows :
    #  fit = {'Vbulge_none' : {...}, 'Vbulge' : {...}} \n
    #  where 'Vbulge_none'/'Vbulge' corresponds to galaxies without/with a bulge component.
    #  - fit['Vbulge_none'] = { \n
    #    'Name' : ndarray[120], \n 
    #    'fit' : ndarray[120], \n
    #    'Chi_sq' : ndarray[120], \n
    #    'BIC' : ndarray[120], \n 
    #    'Mvir' : ndarray[120] (unumpy.uarray), \n 
    #    'params' : dictionary, \n
    #    } \n
    #  - fit['Vbulge'] = same as fit['Vbulge_none']
    #  Each of the components of fit['Vbulge_none'] and fit['Vbulge'] are :
    #  - 'Name' = name of galaxy
    #  - 'fit' = model_fit.model_fit.fit object
    #  - 'Chi_sq' = reduced chi-squared
    #  - 'BIC' = Bayesian information criterion
    #  - 'Mvir' = M200 (mass enclosed within radius R200) in units of solar masses
    #  - 'params' = { \n
    #     'c200' : ndarray[120] (unumpy.uarray) = halo concentration, \n
    #     'v200' : ndarray[120] (unumpy.uarray) = halo virial velocity, \n
    #     'MLd' : ndarray[120] (unumpy.uarray) = mass-to-light ratio of disk, \n
    #     'MLb' : ndarray[120] (unumpy.uarray) = mass-to-light ratio of bulge, \n
    #     'alpha' : ndarray[120] (unumpy.uarray) = Einasto halo parameter, \n
    #     'mstar' : ndarray[120] (unumpy.uarray) = mass of stellar component, \n
    #     'Vf' : ndarray[120] = maximum circular velocity, \n
    #     'mgas' : ndarray[120] = mass of gas component, \n
    #     } \n
    def fit(self):
        lvl1=['Vbulge_none','Vbulge']
        lvl2=['Name','fit','Chi_sq','BIC','Mvir']
        lvl3=['c200','v200','MLd','MLb','alpha','mstar','Vf','mgas']
        mod_funcs=cdm_funcs.halo(self.model,fit_dict_in=self.fit_dict_in)
        CDM_dict_in={}
        for key in lvl1:
            CDM_dict_in[key]={}
            for key1 in lvl2:
                CDM_dict_in[key][key1]=np.asarray([])
            CDM_dict_in[key]['params']={}
            for key2 in lvl3:
                CDM_dict_in[key]['params'][key2]=np.asarray([])

        def results_in(dict_in,data_in,name_in,bulge=False):
            params_now={}
            params_now['params']={}
            for key in lvl3:
                params_now['params'][key]=np.asarray([])
            dict_in['Name']=np.append(dict_in['Name'],name_in)
            halo_init=halo.halo(self.model,data_in,fit_dict_in=self.fit_dict_in)
            params=halo_init.params
            fit_init=model_fit.model_fit(halo_init,ULDM_fits=self.ULDM_fits)
            fit=fit_init.fit(params,data_in)
            if (fit.params['c200'].stderr!=None and np.isnan(fit.params['c200'].stderr)==False 
                    and fit.params['c200'].stderr<fit.params['c200'].value):
                c200err=fit.params['c200'].stderr
            else:
                c200err=np.inf
            params_arr_in=unumpy.uarray([fit.params['c200'].value],[c200err])
            dict_in['params']['c200']=np.append(dict_in['params']['c200'],params_arr_in)
            params_now['params']['c200']=np.append(params_now['params']['c200'],params_arr_in)
            if (fit.params['v200'].stderr!=None and np.isnan(fit.params['v200'].stderr)==False 
                    and fit.params['v200'].stderr<fit.params['v200'].value):
                v200err=fit.params['v200'].stderr
            else:
                v200err=np.inf
            params_arr_in=unumpy.uarray([fit.params['v200'].value],[v200err])
            dict_in['params']['v200']=np.append(dict_in['params']['v200'],params_arr_in)
            params_now['params']['v200']=np.append(params_now['params']['v200'],params_arr_in)
            if (fit.params['MLd'].stderr!=None and np.isnan(fit.params['MLd'].stderr)==False 
                    and fit.params['MLd'].stderr<fit.params['MLd'].value):
                MLderr=fit.params['MLd'].stderr
            else:
                MLderr=np.inf
            params_arr_in=unumpy.uarray([fit.params['MLd'].value],[MLderr])
            dict_in['params']['MLd']=np.append(dict_in['params']['MLd'],params_arr_in)
            params_now['params']['MLd']=np.append(params_now['params']['MLd'],params_arr_in)
            if bulge==True:
                if (fit.params['MLb'].stderr!=None and np.isnan(fit.params['MLb'].stderr)==False 
                        and fit.params['MLb'].stderr<fit.params['MLb'].value):
                    MLberr=fit.params['MLb'].stderr
                else:
                    MLberr=np.inf
                params_arr_in=unumpy.uarray([fit.params['MLb'].value],[MLberr])
                dict_in['params']['MLb']=np.append(dict_in['params']['MLb'],params_arr_in)
                params_now['params']['MLb']=np.append(params_now['params']['MLb'],params_arr_in)
            if self.model=='Einasto':
                if (fit.params['alpha'].stderr!=None and np.isnan(fit.params['alpha'].stderr)==False 
                        and fit.params['alpha'].stderr<fit.params['alpha'].value):
                    alphaerr=fit.params['alpha'].stderr
                else:
                    alphaerr=np.inf
                params_arr_in=unumpy.uarray([fit.params['alpha'].value],[alphaerr])
                dict_in['params']['alpha']=np.append(dict_in['params']['alpha'],params_arr_in)
                params_now['params']['alpha']=np.append(params_now['params']['alpha'],params_arr_in)
            if (fit.params['mstar'].stderr!=None and np.isnan(fit.params['mstar'].stderr)==False 
                    and fit.params['mstar'].stderr<fit.params['mstar'].value):
                mstarerr=fit.params['mstar'].stderr
            else:
                mstarerr=np.inf
            params_arr_in=unumpy.uarray([fit.params['mstar'].value],[mstarerr])
            dict_in['params']['mstar']=np.append(dict_in['params']['mstar'],params_arr_in)
            params_now['params']['mstar']=np.append(params_now['params']['mstar'],params_arr_in)
            dict_in['params']['Vf']=np.append(dict_in['params']['Vf'],data_in['Vf'])
            params_now['params']['Vf']=np.append(params_now['params']['Vf'],data_in['Vf'])
            dict_in['params']['mgas']=np.append(dict_in['params']['mgas'],data_in['Mgas'])
            params_now['params']['mgas']=np.append(params_now['params']['mgas'],data_in['Mgas'])
            dict_in['fit']=np.append(dict_in['fit'],fit)
            dict_in['Chi_sq']=np.append(dict_in['Chi_sq'],fit.redchi)
            dict_in['BIC']=np.append(dict_in['BIC'],fit_init.bic(fit))
            dict_in['Mvir']=np.append(dict_in['Mvir'],mod_funcs.Mvir(params_now['params']))
            return dict_in

        gal=galaxy.galaxy('random')
        for key in gal.df1['galaxy']:
            gal=galaxy.galaxy(key)
            data=gal.data
            if (data['Quality']!=3 and data['Inclination']>30 and data['Vf']!=0):
                if data['Vbulge'].all()==0:
                    if self.ULDM_fits==False:
                        CDM_dict_in['Vbulge_none']=results_in(CDM_dict_in['Vbulge_none'],data,key,bulge=False)
                    else:
                        if (len(data['Vobs'])-11)>0:
                            CDM_dict_in['Vbulge_none']=results_in(CDM_dict_in['Vbulge_none'],data,key,bulge=False)
                elif data['Vbulge'].any()!=0:
                    if self.ULDM_fits==False:
                        CDM_dict_in['Vbulge']=results_in(CDM_dict_in['Vbulge'],data,key,bulge=True)
                    else:
                        if (len(data['Vobs'])-12)>0:
                            CDM_dict_in['Vbulge']=results_in(CDM_dict_in['Vbulge'],data,key,bulge=True)
        return CDM_dict_in

## The class to obtain fit results for all single flavored ULDM models analyzed.
#  This class can be used to obtain results for all single flavored ULDM models analyzed.  
#  93 galaxies in the SPARC catalog are used.
class results_psi_single_all:

    ## Define the constructor of the results_psi_single_all class.
    #  This defines the constructor of the results_psi_single_all class.
    #  @param self
    #  object pointer
    #  @param fit_dict_in (optional)
    #  constants.fitting_dict instance \n
    #  Instance of the constants.fitting_dict class.
    #  Contains all the necessary rules and values to be used in fitting.
    def __init__(self,fit_dict_in=fitting_dict_in):

        ## @var fit_dict_in
        #  constants.fitting_dict instance \n
        #  Instance of the constants.fitting_dict class.
        #  @var args_opts
        #  dictionary \n
        #  Dictionary of rules to be assumed during fitting
        #  Instance of constants.fitting_dict.args_opts
        #  @var mfree
        #  bool \n
        #  Equal to true/false if soliton particle mass free/fixed in fitting procedure
        #  @var cdmhalo
        #  str \n
        #  Can be equal to :
        #  - 'Burkert'
        #  - 'DC14'
        #  - 'Einasto'
        #  - 'NFW'
        #  @var matched
        #  bool \n
        #  True if using ULDM matched models
        self.fit_dict_in=fit_dict_in
        self.args_opts=self.fit_dict_in.args_opts()
        self.mfree=self.args_opts['soliton']['mfree']
        self.cdmhalo=self.args_opts['soliton']['cdm_halo']
        self.matched=self.args_opts['soliton']['matched']

    ## Define the fit results.
    #  This defines the fit results for the single flavor ULDM models for 93 galaxies in the SPARC catalog.
    #  @param self
    #  object pointer
    #  @returns
    #  dictionary \n
    #  Dictionary of resulting fit parameters.  Results are as follows :
    #  fit = {'Vbulge_none' : {...}, 'Vbulge' : {...}} \n
    #  where 'Vbulge_none'/'Vbulge' corresponds to galaxies without/with a bulge component.
    #  - fit['Vbulge_none'] = { \n
    #    'Name' : ndarray[93], \n 
    #    'fit' : ndarray[93], \n
    #    'Chi_sq' : ndarray[93], \n
    #    'BIC' : ndarray[93], \n 
    #    'Mhalo' : ndarray[93], \n
    #    'Mvir' : ndarray[93], \n 
    #    'params' : dictionary, \n
    #    } \n
    #  - fit['Vbulge'] = same as fit['Vbulge_none']
    #  Each of the components of fit['Vbulge_none'] and fit['Vbulge'] are :
    #  - 'Name' = name of galaxy
    #  - 'fit' = model_fit.model_fit.fit object
    #  - 'Chi_sq' = reduced chi-squared
    #  - 'BIC' = Bayesian information criterion
    #  - 'Mhalo' = mass of ULDM halo
    #  - 'Mvir' = M200 (mass enclosed within radius R200) in units of solar masses
    #  - 'params' = { \n
    #     'c200' : ndarray[93] (unumpy.uarray) = halo concentration, \n
    #     'v200' : ndarray[93] (unumpy.uarray) = halo virial velocity, \n
    #     'MLd' : ndarray[93] (unumpy.uarray) = mass-to-light ratio of disk, \n
    #     'MLb' : ndarray[93] (unumpy.uarray) = mass-to-light ratio of bulge, \n
    #     'alpha' : ndarray[93] (unumpy.uarray) = Einasto halo parameter, \n
    #     'm22' : ndarray[93] (unumpy.uarray) = ULDM particle mass, \n
    #     'Msol' : ndarray[93] (unumpy.uarray) = soliton mass, \n
    #     'mstar' : ndarray[93] (unumpy.uarray) = mass of stellar component, \n
    #     'Vf' : ndarray[93] = maximum circular velocity, \n
    #     'mgas' : ndarray[93] = mass of gas component, \n
    #     } \n
    def fit(self):
        if self.matched==True:
            m_tab=self.args_opts['soliton']['m22_tab']
        else:
            m_tab=self.args_opts['soliton']['m22_tab_prime']
        lvl1=['Vbulge_none','Vbulge']
        lvl2=['Name','fit','Chi_sq','BIC','Mhalo','Mvir']
        lvl3=['c200','v200','MLd','MLb','alpha','m22','Msol','mstar','Vf','mgas']
        psis_dict_in={}
        for key in lvl1:
            psis_dict_in[key]={}
            if self.mfree==False:
                for i in range(len(m_tab)):
                    psis_dict_in[key]['m='+str(np.log10(m_tab[i]))]={}
                    for key1 in lvl2:
                        psis_dict_in[key]['m='+str(np.log10(m_tab[i]))][key1]=np.asarray([])
                    psis_dict_in[key]['m='+str(np.log10(m_tab[i]))]['params']={}
                    for key2 in lvl3:
                        psis_dict_in[key]['m='+str(np.log10(m_tab[i]))]['params'][key2]=np.asarray([])
            else:
                for key1 in lvl2:
                    psis_dict_in[key][key1]=np.asarray([])
                psis_dict_in[key]['params']={}
                for key2 in lvl3:
                        psis_dict_in[key]['params'][key2]=np.asarray([])

        def results_in(dict_in,data_in,bulge=False,fit_dict_in=self.fit_dict_in):
            halo_init=halo.halo('psi_single',data_in,fit_dict_in=fit_dict_in)
            params=halo_init.params
            fit_init=model_fit.model_fit(halo_init,ULDM_fits=True)
            fit=fit_init.fit(params,data_in)
            dict_in['Name']=np.append(dict_in['Name'],key)
            if (fit.params['c200'].stderr!=None and np.isnan(fit.params['c200'].stderr)==False 
                    and fit.params['c200'].stderr<fit.params['c200'].value):
                c200err=fit.params['c200'].stderr
            else:
                c200err=np.inf
            params_arr_in=unumpy.uarray([fit.params['c200'].value],[c200err])
            dict_in['params']['c200']=np.append(dict_in['params']['c200'],params_arr_in)
            if (fit.params['v200'].stderr!=None and np.isnan(fit.params['v200'].stderr)==False 
                    and fit.params['v200'].stderr<fit.params['v200'].value):
                v200err=fit.params['v200'].stderr
            else:
                v200err=np.inf
            params_arr_in=unumpy.uarray([fit.params['v200'].value],[v200err])
            dict_in['params']['v200']=np.append(dict_in['params']['v200'],params_arr_in)
            if (fit.params['MLd'].stderr!=None and np.isnan(fit.params['MLd'].stderr)==False 
                    and fit.params['MLd'].stderr<fit.params['MLd'].value):
                MLderr=fit.params['MLd'].stderr
            else:
                MLderr=np.inf
            params_arr_in=unumpy.uarray([fit.params['MLd'].value],[MLderr])
            dict_in['params']['MLd']=np.append(dict_in['params']['MLd'],params_arr_in)
            if bulge==True:
                if (fit.params['MLb'].stderr!=None and np.isnan(fit.params['MLb'].stderr)==False 
                        and fit.params['MLb'].stderr<fit.params['MLb'].value):
                    MLberr=fit.params['MLb'].stderr
                else:
                    MLberr=np.inf
                params_arr_in=unumpy.uarray([fit.params['MLb'].value],[MLberr])
                dict_in['params']['MLb']=np.append(dict_in['params']['MLb'],params_arr_in)
            if self.cdmhalo=='Einasto':
                if (fit.params['alpha'].stderr!=None and np.isnan(fit.params['alpha'].stderr)==False 
                        and fit.params['alpha'].stderr<fit.params['alpha'].value):
                    alphaerr=fit.params['alpha'].stderr
                else:
                    alphaerr=np.inf
                params_arr_in=unumpy.uarray([fit.params['alpha'].value],[alphaerr])
                dict_in['params']['alpha']=np.append(dict_in['params']['alpha'],params_arr_in)
            if (fit.params['mstar'].stderr!=None and np.isnan(fit.params['mstar'].stderr)==False 
                    and fit.params['mstar'].stderr<fit.params['mstar'].value):
                mstarerr=fit.params['mstar'].stderr
            else:
                mstarerr=np.inf
            params_arr_in=unumpy.uarray([fit.params['mstar'].value],[mstarerr])
            dict_in['params']['mstar']=np.append(dict_in['params']['mstar'],params_arr_in)
            dict_in['params']['Vf']=np.append(dict_in['params']['Vf'],data_in['Vf'])
            dict_in['params']['mgas']=np.append(dict_in['params']['mgas'],data_in['Mgas'])
            if (fit.params['Msol'].stderr!=None and np.isnan(fit.params['Msol'].stderr)==False 
                    and fit.params['Msol'].stderr<fit.params['Msol'].value):
                msolerr=fit.params['Msol'].stderr
            else:
                msolerr=np.inf
            params_arr_in=unumpy.uarray([fit.params['Msol']],[msolerr])
            dict_in['params']['Msol']=np.append(dict_in['params']['Msol'],params_arr_in)
            if self.mfree==False:
                dict_in['params']['m22']=np.append(dict_in['params']['m22'],fit.params['m22'])
            else:
                if (fit.params['m22'].stderr!=None and np.isnan(fit.params['m22'].stderr)==False 
                        and fit.params['m22'].stderr<fit.params['m22'].value):
                    m22err=fit.params['m22'].stderr
                else:
                    m22err=np.inf
                params_arr_in=unumpy.uarray([fit.params['m22']],[m22err])
                dict_in['params']['m22']=np.append(dict_in['params']['m22'],params_arr_in)
            dict_in['fit']=np.append(dict_in['fit'],fit)
            dict_in['Chi_sq']=np.append(dict_in['Chi_sq'],fit.redchi)
            dict_in['BIC']=np.append(dict_in['BIC'],fit_init.bic(fit))
            psi_in=alp_funcs.soliton('psi_single',fit_dict_in=fit_dict_in)
            dict_in['Mhalo']=np.append(dict_in['Mhalo'],psi_in.Mhalo(fit.params))
            dict_in['Mvir']=np.append(dict_in['Mvir'],psi_in.Mvir(fit.params))
            return dict_in

        if self.mfree==False:
            loop_len=len(m_tab)
        else:
            loop_len=1
        for i in range(loop_len):
            self.fit_dict_in.sol_m22=m_tab[i]
            gal=galaxy.galaxy('random')
            for key in gal.df1['galaxy']:
                gal=galaxy.galaxy(key)
                data=gal.data
                if (data['Quality']!=3 and data['Inclination']>30 and data['Vf']!=0):
                    if data['Vbulge'].all()==0: 
                        if (len(data['Vobs'])-11)>0:
                            if self.mfree==False:
                                psis_dict_in['Vbulge_none']['m='+str(np.log10(m_tab[i]))]=results_in(psis_dict_in['Vbulge_none']['m='+str(np.log10(m_tab[i]))],
                                    data,bulge=False,fit_dict_in=self.fit_dict_in)
                            else:
                                psis_dict_in['Vbulge_none']=results_in(psis_dict_in['Vbulge_none'],
                                    data,bulge=False,fit_dict_in=self.fit_dict_in)
                    elif data['Vbulge'].any()!=0:
                        if (len(data['Vobs'])-12)>0:
                            if self.mfree==False:
                                psis_dict_in['Vbulge']['m='+str(np.log10(m_tab[i]))]=results_in(psis_dict_in['Vbulge']['m='+str(np.log10(m_tab[i]))],
                                    data,bulge=True,fit_dict_in=self.fit_dict_in)
                            else:
                                psis_dict_in['Vbulge']=results_in(psis_dict_in['Vbulge'],
                                    data,bulge=True,fit_dict_in=self.fit_dict_in)
        return psis_dict_in

## The class to obtain fit results for all double flavored ULDM models analyzed.
#  This class can be used to obtain results for all double flavored ULDM models analyzed.  
#  93 galaxies in the SPARC catalog are used.
class results_psi_multi_all:

    ## Define the constructor of the results_psi_multi_all class.
    #  This defines the constructor of the results_psi_multi_all class.
    #  @param self
    #  object pointer
    #  @param fit_dict_in (optional)
    #  constants.fitting_dict instance \n
    #  Instance of the constants.fitting_dict class.
    #  Contains all the necessary rules and values to be used in fitting.
    def __init__(self,fit_dict_in=fitting_dict_in):

        ## @var fit_dict_in
        #  constants.fitting_dict instance \n
        #  Instance of the constants.fitting_dict class.
        #  @var args_opts
        #  dictionary \n
        #  Dictionary of rules to be assumed during fitting
        #  Instance of constants.fitting_dict.args_opts
        #  @var mfree
        #  bool \n
        #  Equal to true/false if soliton particle mass free/fixed in fitting procedure
        #  @var cdmhalo
        #  str \n
        #  Can be equal to :
        #  - 'Burkert'
        #  - 'DC14'
        #  - 'Einasto'
        #  - 'NFW'
        #  @var matched
        #  bool \n
        #  True if using ULDM matched models
        self.fit_dict_in=fit_dict_in
        self.args_opts=self.fit_dict_in.args_opts()
        self.mfree=self.args_opts['soliton']['mfree']
        self.cdmhalo=self.args_opts['soliton']['cdm_halo']
        self.matched=self.args_opts['soliton']['matched']

    ## Define the fit results.
    #  This defines the fit results for the double flavor ULDM models for 93 galaxies in the SPARC catalog.
    #  @param self
    #  object pointer
    #  @returns
    #  dictionary \n
    #  Dictionary of resulting fit parameters.  Results are as follows :
    #  fit = {'Vbulge_none' : {...}, 'Vbulge' : {...}} \n
    #  where 'Vbulge_none'/'Vbulge' corresponds to galaxies without/with a bulge component.
    #  - fit['Vbulge_none'] = { \n
    #    'Name' : ndarray[93], \n 
    #    'fit' : ndarray[93], \n
    #    'Chi_sq' : ndarray[93], \n
    #    'BIC' : ndarray[93], \n 
    #    'Mhalo_1' : ndarray[93], \n
    #    'Mhalo_2' : ndarray[93], \n
    #    'Mvir' : ndarray[93], \n 
    #    'params' : dictionary, \n
    #    } \n
    #  - fit['Vbulge'] = same as fit['Vbulge_none']
    #  Each of the components of fit['Vbulge_none'] and fit['Vbulge'] are :
    #  - 'Name' = name of galaxy
    #  - 'fit' = model_fit.model_fit.fit object
    #  - 'Chi_sq' = reduced chi-squared
    #  - 'BIC' = Bayesian information criterion
    #  - 'Mhalo_1' = mass of ULDM halo one
    #  - 'Mhalo_2' = mass of ULDM halo two
    #  - 'Mvir' = M200 (mass enclosed within radius R200) in units of solar masses
    #  - 'params' = { \n
    #     'c200' : ndarray[93] (unumpy.uarray) = halo one concentration, \n
    #     'c200_2' : ndarray[93] (unumpy.uarray) = halo two concentration, \n
    #     'v200' : ndarray[93] (unumpy.uarray) = halo one virial velocity, \n
    #     'v200_2' : ndarray[93] (unumpy.uarray) = halo two virial velocity, \n
    #     'MLd' : ndarray[93] (unumpy.uarray) = mass-to-light ratio of disk, \n
    #     'MLb' : ndarray[93] (unumpy.uarray) = mass-to-light ratio of bulge, \n
    #     'alpha' : ndarray[93] (unumpy.uarray) = Einasto halo parameter, \n
    #     'm22' : ndarray[93] (unumpy.uarray) = ULDM particle one mass, \n
    #     'm22_2' : ndarray[93] (unumpy.uarray) = ULDM particle two mass, \n
    #     'Msol' : ndarray[93] (unumpy.uarray) = soliton one mass, \n
    #     'Msol_2' : ndarray[93] (unumpy.uarray) = soliton two mass, \n
    #     'mstar' : ndarray[93] (unumpy.uarray) = mass of stellar component, \n
    #     'Vf' : ndarray[93] = maximum circular velocity, \n
    #     'mgas' : ndarray[93] = mass of gas component, \n
    #     } \n
    def fit(self):
        if self.matched==True:
            m_tab_2=self.args_opts['soliton']['m22_2_tab']
        else:
            m_tab_2=self.args_opts['soliton']['m22_2_tab_prime']
        lvl1=['Vbulge_none','Vbulge']
        lvl2=['Name','fit','Chi_sq','BIC','Mhalo_1','Mhalo_2','Mvir']
        lvl3=['c200','c200_2','v200','v200_2','MLd','MLb','alpha','alpha_2','m22','m22_2','Msol','Msol_2','mstar','Vf','mgas']
        psim_dict_in={}
        for key in lvl1:
            psim_dict_in[key]={}
            if self.mfree==False:
                for i in range(len(m_tab_2)):
                    psim_dict_in[key]['m='+str(np.log10(m_tab_2[i]))]={}
                    for key1 in lvl2:
                        psim_dict_in[key]['m='+str(np.log10(m_tab_2[i]))][key1]=np.asarray([])
                    psim_dict_in[key]['m='+str(np.log10(m_tab_2[i]))]['params']={}
                    for key2 in lvl3:
                        psim_dict_in[key]['m='+str(np.log10(m_tab_2[i]))]['params'][key2]=np.asarray([])
            else:
                for key1 in lvl2:
                    psim_dict_in[key][key1]=np.asarray([])
                psim_dict_in[key]['params']={}
                for key2 in lvl3:
                        psim_dict_in[key]['params'][key2]=np.asarray([])
        
        def results_in(dict_in,data_in,bulge=False,fit_dict_in=self.fit_dict_in):
            halo_init=halo.halo('psi_multi',data_in,fit_dict_in=fit_dict_in)
            params=halo_init.params
            fit_init=model_fit.model_fit(halo_init,ULDM_fits=True)
            fit=fit_init.fit(params,data_in)
            dict_in['Name']=np.append(dict_in['Name'],key)
            if (fit.params['c200'].stderr!=None and np.isnan(fit.params['c200'].stderr)==False 
                    and fit.params['c200'].stderr<fit.params['c200'].value):
                c200err=fit.params['c200'].stderr
            else:
                c200err=np.inf
            params_arr_in=unumpy.uarray([fit.params['c200'].value],[c200err])
            dict_in['params']['c200']=np.append(dict_in['params']['c200'],params_arr_in)
            if (fit.params['c200_2'].stderr!=None and np.isnan(fit.params['c200_2'].stderr)==False 
                    and fit.params['c200_2'].stderr<fit.params['c200_2'].value):
                c200err=fit.params['c200_2'].stderr
            else:
                c200err=np.inf
            params_arr_in=unumpy.uarray([fit.params['c200_2'].value],[c200err])
            dict_in['params']['c200_2']=np.append(dict_in['params']['c200_2'],params_arr_in)
            if (fit.params['v200'].stderr!=None and np.isnan(fit.params['v200'].stderr)==False 
                    and fit.params['v200'].stderr<fit.params['v200'].value):
                v200err=fit.params['v200'].stderr
            else:
                v200err=np.inf
            params_arr_in=unumpy.uarray([fit.params['v200'].value],[v200err])
            dict_in['params']['v200']=np.append(dict_in['params']['v200'],params_arr_in)
            if (fit.params['v200_2'].stderr!=None and np.isnan(fit.params['v200_2'].stderr)==False 
                    and fit.params['v200_2'].stderr<fit.params['v200_2'].value):
                v200err=fit.params['v200_2'].stderr
            else:
                v200err=np.inf
            params_arr_in=unumpy.uarray([fit.params['v200_2'].value],[v200err])
            dict_in['params']['v200_2']=np.append(dict_in['params']['v200_2'],params_arr_in)
            if (fit.params['MLd'].stderr!=None and np.isnan(fit.params['MLd'].stderr)==False 
                    and fit.params['MLd'].stderr<fit.params['MLd'].value):
                MLderr=fit.params['MLd'].stderr
            else:
                MLderr=np.inf
            params_arr_in=unumpy.uarray([fit.params['MLd'].value],[MLderr])
            dict_in['params']['MLd']=np.append(dict_in['params']['MLd'],params_arr_in)
            if bulge==True:
                if (fit.params['MLb'].stderr!=None and np.isnan(fit.params['MLb'].stderr)==False 
                        and fit.params['MLb'].stderr<fit.params['MLb'].value):
                    MLberr=fit.params['MLb'].stderr
                else:
                    MLberr=np.inf
                params_arr_in=unumpy.uarray([fit.params['MLb'].value],[MLberr])
                dict_in['params']['MLb']=np.append(dict_in['params']['MLb'],params_arr_in)
            if self.cdmhalo=='Einasto':
                if (fit.params['alpha'].stderr!=None and np.isnan(fit.params['alpha'].stderr)==False 
                        and fit.params['alpha'].stderr<fit.params['alpha'].value):
                    alphaerr=fit.params['alpha'].stderr
                else:
                    alphaerr=np.inf
                params_arr_in=unumpy.uarray([fit.params['alpha'].value],[alphaerr])
                dict_in['params']['alpha']=np.append(dict_in['params']['alpha'],params_arr_in)
                if (fit.params['alpha_2'].stderr!=None and np.isnan(fit.params['alpha_2'].stderr)==False):
                    alphaerr=fit.params['alpha_2'].stderr
                else:
                    alphaerr=np.inf
                params_arr_in=unumpy.uarray([fit.params['alpha_2'].value],[alphaerr])
                dict_in['params']['alpha_2']=np.append(dict_in['params']['alpha_2'],params_arr_in)
            if (fit.params['mstar'].stderr!=None and np.isnan(fit.params['mstar'].stderr)==False 
                    and fit.params['mstar'].stderr<fit.params['mstar'].value):
                mstarerr=fit.params['mstar'].stderr
            else:
                mstarerr=np.inf
            params_arr_in=unumpy.uarray([fit.params['mstar'].value],[mstarerr])
            dict_in['params']['mstar']=np.append(dict_in['params']['mstar'],params_arr_in)
            dict_in['params']['Vf']=np.append(dict_in['params']['Vf'],data_in['Vf'])
            dict_in['params']['mgas']=np.append(dict_in['params']['mgas'],data_in['Mgas'])
            if (fit.params['Msol'].stderr!=None and np.isnan(fit.params['Msol'].stderr)==False 
                    and fit.params['Msol'].stderr<fit.params['Msol'].value):
                msolerr=fit.params['Msol'].stderr
            else:
                msolerr=np.inf
            params_arr_in=unumpy.uarray([fit.params['Msol']],[msolerr])
            dict_in['params']['Msol']=np.append(dict_in['params']['Msol'],params_arr_in)
            if (fit.params['Msol_2'].stderr!=None and np.isnan(fit.params['Msol_2'].stderr)==False 
                    and fit.params['Msol_2'].stderr<fit.params['Msol_2'].value):
                msolerr=fit.params['Msol_2'].stderr
            else:
                msolerr=np.inf
            params_arr_in=unumpy.uarray([fit.params['Msol_2']],[msolerr])
            dict_in['params']['Msol_2']=np.append(dict_in['params']['Msol_2'],params_arr_in)
            if self.mfree==False:
                dict_in['params']['m22']=np.append(dict_in['params']['m22'],fit.params['m22'])
                dict_in['params']['m22_2']=np.append(dict_in['params']['m22_2'],fit.params['m22_2'])
            else:
                if (fit.params['m22'].stderr!=None and np.isnan(fit.params['m22'].stderr)==False 
                        and fit.params['m22'].stderr<fit.params['m22'].value):
                    m22err=fit.params['m22'].stderr
                else:
                    m22err=np.inf
                params_arr_in=unumpy.uarray([fit.params['m22']],[m22err])
                dict_in['params']['m22']=np.append(dict_in['params']['m22'],params_arr_in)
                if (fit.params['m22_2'].stderr!=None and np.isnan(fit.params['m22_2'].stderr)==False 
                        and fit.params['m22_2'].stderr<fit.params['m22_2'].value):
                    m22err=fit.params['m22_2'].stderr
                else:
                    m22err=np.inf
                params_arr_in=unumpy.uarray([fit.params['m22_2']],[m22err])
                dict_in['params']['m22_2']=np.append(dict_in['params']['m22_2'],params_arr_in)
            dict_in['fit']=np.append(dict_in['fit'],fit)
            dict_in['Chi_sq']=np.append(dict_in['Chi_sq'],fit.redchi)
            dict_in['BIC']=np.append(dict_in['BIC'],fit_init.bic(fit))
            psi_in=alp_funcs.soliton('psi_multi',fit_dict_in=fit_dict_in)
            dict_in['Mhalo_1']=np.append(dict_in['Mhalo_1'],psi_in.Mhalo(fit.params)[0])
            dict_in['Mhalo_2']=np.append(dict_in['Mhalo_2'],psi_in.Mhalo(fit.params)[1])
            dict_in['Mvir']=np.append(dict_in['Mvir'],psi_in.Mvir(fit.params))
            return dict_in

        if self.mfree==False:
            loop_len=len(m_tab_2)
        else:
            loop_len=1
        for i in range(loop_len):
            self.fit_dict_in.sol_m22_2=m_tab_2[i]
            gal=galaxy.galaxy('random')
            for key in gal.df1['galaxy']:
                gal=galaxy.galaxy(key)
                data=gal.data
                if (data['Quality']!=3 and data['Inclination']>30 and data['Vf']!=0):
                    if data['Vbulge'].all()==0:
                        if (len(data['Vobs'])-11)>0:
                            if self.mfree==False:
                                psim_dict_in['Vbulge_none']['m='+str(np.log10(m_tab_2[i]))]=results_in(psim_dict_in['Vbulge_none']['m='+str(np.log10(m_tab_2[i]))],
                                    data,bulge=False,fit_dict_in=self.fit_dict_in)
                            else:
                                psim_dict_in['Vbulge_none']=results_in(psim_dict_in['Vbulge_none'],
                                    data,bulge=False,fit_dict_in=self.fit_dict_in)
                    elif data['Vbulge'].any()!=0:
                        if (len(data['Vobs'])-12)>0:
                            if self.mfree==False:
                                psim_dict_in['Vbulge']['m='+str(np.log10(m_tab_2[i]))]=results_in(psim_dict_in['Vbulge']['m='+str(np.log10(m_tab_2[i]))],
                                    data,bulge=True,fit_dict_in=self.fit_dict_in)
                            else:
                                psim_dict_in['Vbulge']=results_in(psim_dict_in['Vbulge'],
                                    data,bulge=True,fit_dict_in=self.fit_dict_in)
        return psim_dict_in

## The class to obtain fit results to check all CDM model implementation.
#  The class to obtain fit results for comparison with Pengfei Li et al 2020 ApJS 247 31 :
#  https://doi.org/10.3847/1538-4365/ab700e
class results_CDM_check:

    ## Define the constructor of the results_CDM_check class.
    #  This defines the constructor of the results_CDM_check class.
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
        self.model=model
        self.fit_dict_in=fit_dict_in

    ## Define the fit results.
    #  This defines the fit results for the assumed model for all galaxies in the SPARC catalog.
    #  @param self
    #  object pointer \n
    #  @returns
    #  dictionary \n
    #  Dictionary of resulting fit parameters.  Results are as follows :
    #  fit = {'Vbulge_none' : {...}, 'Vbulge' : {...}} \n
    #  where 'Vbulge_none'/'Vbulge' corresponds to galaxies without/with a bulge component.
    #  - fit['Vbulge_none'] = { \n
    #    'Name' : str[175], \n 
    #    'Chi_sq' : ndarray[175], \n
    #    'params' : ndarray[175, 3 or 4], \n
    #    'Mvir' : float, \n 
    #    'fit' : model_fit.model_fit.fit object, \n
    #    } \n
    #  - fit['Vbulge'] = same as 'Vbulge_none' with the change : \n
    #    'params' : ndarray[175, 4 or 5] \n
    #  Each of the components of fit['Vbulge_none'] and fit['Vbulge'] are :
    #  - 'Name' = name of galaxy
    #  - 'Chi_sq' = reduced chi-squared
    #  - 'params' = fit parameters
    #     - for Burkert, DC14, and NFW without bulge component: [c200, v200, MLd]
    #     - for Burkert, DC14, and NFW with bulge component:    [c200, v200, MLd, MLb]
    #     - for Einasto without bulge component:                [c200, v200, MLd, alpha]
    #     - for Einasto with bulge component:                   [c200, v200, MLd, MLb, alpha]
    #  - 'Mvir' = M200 (mass enclosed within radius R200) in units of solar masses
    #  - 'fit' = model_fit.model_fit.fit object
    def fit(self):
        lvl1=['Vbulge_none','Vbulge']
        lvl2=['Name','Chi_sq','params','Mvir','fit']
        mod_funcs=cdm_funcs.halo(self.model,fit_dict_in=self.fit_dict_in)
        CDM_dict_in={}
        for key in lvl1:
            CDM_dict_in[key]={}
            for key1 in lvl2:
                CDM_dict_in[key][key1]=np.asarray([])

        def results_in(dict_in,data_in,bulge=False):
            dict_in['Name']=np.append(dict_in['Name'],key)
            halo_init=halo.halo(self.model,data_in,fit_dict_in=self.fit_dict_in)
            params=halo_init.params
            fit_init=model_fit.model_fit(halo_init)
            fit=fit_init.fit(params,data_in)
            if (self.model=='Burkert' or self.model=='DC14' or self.model=='NFW'):
                if bulge==False:
                    dict_in['params']=np.append(dict_in['params'],[fit.params['c200'].value,fit.params['v200'].value,
                        fit.params['MLd'].value])
                    dict_in['params']=np.reshape(dict_in['params'],(int(len(dict_in['params'])/3),3))
                else:
                    dict_in['params']=np.append(dict_in['params'],[fit.params['c200'].value,fit.params['v200'].value,
                        fit.params['MLd'].value,fit.params['MLb'].value])
                    dict_in['params']=np.reshape(dict_in['params'],(int(len(dict_in['params'])/4),4))
            else:
                if bulge==False:
                    dict_in['params']=np.append(dict_in['params'],[fit.params['c200'].value,fit.params['v200'].value,
                        fit.params['MLd'].value,fit.params['alpha'].value])
                    dict_in['params']=np.reshape(dict_in['params'],(int(len(dict_in['params'])/4),4))
                else:
                    dict_in['params']=np.append(dict_in['params'],[fit.params['c200'].value,fit.params['v200'].value,
                        fit.params['MLd'].value,fit.params['MLb'].value,fit.params['alpha'].value])
                    dict_in['params']=np.reshape(dict_in['params'],(int(len(dict_in['params'])/5),5))
            dict_in['Chi_sq']=np.append(dict_in['Chi_sq'],fit.redchi)
            dict_in['Mvir']=np.append(dict_in['Mvir'],mod_funcs.Mvir(fit.params))
            dict_in['fit']=np.append(dict_in['fit'],fit)
            return dict_in

        gal=galaxy.galaxy('random')
        for key in gal.df1['galaxy']:
            gal=galaxy.galaxy(key)
            data=gal.data
            if data['Vbulge'].all()==0:
                CDM_dict_in['Vbulge_none']=results_in(CDM_dict_in['Vbulge_none'],data,bulge=False)
            elif data['Vbulge'].any()!=0:
                CDM_dict_in['Vbulge']=results_in(CDM_dict_in['Vbulge'],data,bulge=True)
        return CDM_dict_in

## The class to obtain fit results to check the DC14 and NFW model implementation.
#  The class to obtain fit results for comparison with 
#  Monthly Notices of the Royal Astronomical Society, Volume 466, Issue 2, April 2017, Pages 1648–1668 :
#  https://doi.org/10.1093/mnras/stw3101
class results_DC14_check:

    ## Define the constructor of the results_DC14_check class.
    #  This defines the constructor of the results_DC14_check class.
    #  @param self
    #  object pointer
    #  @param model
    #  str \n
    #  Model to assume for DM halo.
    #  Can be equal to :
    #  - DC14
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
        self.model=model
        self.fit_dict_in=fit_dict_in

    ## Define the fit results.
    #  This defines the fit results for the assumed model for 120 galaxies in the SPARC catalog.
    #  @param self
    #  object pointer \n
    #  @returns
    #  dictionary \n
    #  Dictionary of resulting fit parameters.  Results are as follows :
    #  fit = {'Vbulge_none' : {...}, 'Vbulge' : {...}} \n
    #  where 'Vbulge_none'/'Vbulge' corresponds to galaxies without/with a bulge component.
    #  - fit['Vbulge_none'] = { \n
    #    'Name' : str[149], \n 
    #    'Chi_sq' : ndarray[149], \n
    #    'params' : ndarray[149, 3], \n
    #    'Mvir' : float, \n 
    #    'BIC' : float, \n 
    #    'fit' : model_fit.model_fit.fit object, \n
    #    } \n
    #  - fit['Vbulge'] = same as 'Vbulge_none' with the change : \n
    #    'params' : ndarray[149, 4] \n
    #  Each of the components of fit['Vbulge_none'] and fit['Vbulge'] are :
    #  - 'Name' = name of galaxy
    #  - 'Chi_sq' = reduced chi-squared
    #  - 'params' = fit parameters
    #     - for DC14, and NFW without/with bulge component: [c200, v200, MLd]
    #  - 'Mvir' = M200 (mass enclosed within radius R200) in units of solar masses
    #  - 'BIC' = Bayesian information criterion (using lmfit definition)
    #  - 'fit' = model_fit.model_fit.fit object
    def fit(self):
        lvl1=['Vbulge_none','Vbulge']
        lvl2=['Name','Chi_sq','params','Mvir','BIC','fit']
        mod_funcs=cdm_funcs.halo(self.model,fit_dict_in=self.fit_dict_in)
        CDM_dict_in={}
        for key in lvl1:
            CDM_dict_in[key]={}
            for key1 in lvl2:
                CDM_dict_in[key][key1]=np.asarray([])

        def results_in(dict_in,data_in):
            dict_in['Name']=np.append(dict_in['Name'],key)
            halo_init=halo.halo(self.model,data_in,fit_dict_in=self.fit_dict_in)
            params=halo_init.params
            fit_init=model_fit.model_fit(halo_init)
            fit=fit_init.fit(params,data_in)
            dict_in['params']=np.append(dict_in['params'],[10**(fit.params['c200'].value),10**(fit.params['v200'].value),
                10**(fit.params['MLd'].value)])
            dict_in['params']=np.reshape(dict_in['params'],(int(len(dict_in['params'])/3),3))
            dict_in['Chi_sq']=np.append(dict_in['Chi_sq'],fit.redchi)
            dict_in['Mvir']=np.append(dict_in['Mvir'],mod_funcs.Mvir(fit.params))
            dict_in['fit']=np.append(dict_in['fit'],fit)
            dict_in['BIC']=np.append(dict_in['BIC'],fit_init.bic(fit))
            return dict_in

        gal=galaxy.galaxy('random')
        for key in gal.df1['galaxy']:
            gal=galaxy.galaxy(key)
            data=gal.data
            if (data['Quality']!=3 and data['Inclination']>30):
                if data['Vbulge'].all()==0:
                    CDM_dict_in['Vbulge_none']=results_in(CDM_dict_in['Vbulge_none'],data)
                elif data['Vbulge'].any()!=0:
                    CDM_dict_in['Vbulge']=results_in(CDM_dict_in['Vbulge'],data)
        return CDM_dict_in

## The class to obtain fit results to check the Einasto and NFW model implementation.
#  The class to obtain fit results for comparison with 
#  Nicolas Loizeau and Glennys R. Farrar 2021 ApJL 920 L10 :
#  https://doi.org/10.3847/2041-8213/ac1bb7
class results_Einasto_check:

    ## Define the constructor of the results_Einasto_check class.
    #  This defines the constructor of the results_Einasto_check class.
    #  @param self
    #  object pointer
    #  @param model
    #  str \n
    #  Model to assume for DM halo.
    #  Can be equal to :
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
        self.model=model
        self.fit_dict_in=fit_dict_in

    ## Define the fit results.
    #  This defines the fit results for the assumed model for 120 galaxies in the SPARC catalog.
    #  @param self
    #  object pointer \n
    #  @returns
    #  dictionary \n
    #  Dictionary of resulting fit parameters.  Results are as follows :
    #  fit = {'Vbulge_none' : {...}, 'Vbulge' : {...}} \n
    #  where 'Vbulge_none'/'Vbulge' corresponds to galaxies without/with a bulge component.
    #  - fit['Vbulge_none'] = { \n
    #    'Name' : str[121], \n 
    #    'Chi_sq' : ndarray[121], \n
    #    'params' : ndarray[121, 3 or 4], \n
    #    'Mvir' : float, \n 
    #    'fit' : model_fit.model_fit.fit object, \n
    #    } \n
    #  - fit['Vbulge'] = same as 'Vbulge_none' with the change : \n
    #    'params' : ndarray[121, 4 or 5] \n
    #  Each of the components of fit['Vbulge_none'] and fit['Vbulge'] are :
    #  - 'Name' = name of galaxy
    #  - 'Chi_sq' = reduced chi-squared
    #  - 'params' = fit parameters
    #     - for NFW without bulge component:     [c200, v200, MLd]
    #     - for NFW with bulge component:        [c200, v200, MLd, MLb]
    #     - for Einasto without bulge component: [c200, v200, MLd, alpha]
    #     - for Einasto with bulge component:    [c200, v200, MLd, MLb, alpha]
    #  - 'Mvir' = M200 (mass enclosed within radius R200) in units of solar masses
    #  - 'fit' = model_fit.model_fit.fit object
    def fit(self):
        lvl1=['Vbulge_none','Vbulge']
        lvl2=['Name','Chi_sq','params','Mvir','fit']
        mod_funcs=cdm_funcs.halo(self.model,fit_dict_in=self.fit_dict_in)
        CDM_dict_in={}
        for key in lvl1:
            CDM_dict_in[key]={}
            for key1 in lvl2:
                CDM_dict_in[key][key1]=np.asarray([])

        def results_in(dict_in,data_in,bulge=False):
            dict_in['Name']=np.append(dict_in['Name'],key)
            halo_init=halo.halo(self.model,data_in,fit_dict_in=self.fit_dict_in)
            params=halo_init.params
            fit_init=model_fit.model_fit(halo_init)
            fit=fit_init.fit(params,data_in)
            if (self.model=='NFW'):
                if bulge==False:
                    dict_in['params']=np.append(dict_in['params'],[fit.params['c200'].value,fit.params['v200'].value,
                        fit.params['MLd'].value])
                    dict_in['params']=np.reshape(dict_in['params'],(int(len(dict_in['params'])/3),3))
                else:
                    dict_in['params']=np.append(dict_in['params'],[fit.params['c200'].value,fit.params['v200'].value,
                        fit.params['MLd'].value,fit.params['MLb'].value])
                    dict_in['params']=np.reshape(dict_in['params'],(int(len(dict_in['params'])/4),4))
            else:
                if bulge==False:
                    dict_in['params']=np.append(dict_in['params'],[fit.params['c200'].value,fit.params['v200'].value,
                        fit.params['MLd'].value,fit.params['alpha'].value])
                    dict_in['params']=np.reshape(dict_in['params'],(int(len(dict_in['params'])/4),4))
                else:
                    dict_in['params']=np.append(dict_in['params'],[fit.params['c200'].value,fit.params['v200'].value,
                        fit.params['MLd'].value,fit.params['MLb'].value,fit.params['alpha'].value])
                    dict_in['params']=np.reshape(dict_in['params'],(int(len(dict_in['params'])/5),5))
            dict_in['Chi_sq']=np.append(dict_in['Chi_sq'],fit.redchi)
            dict_in['Mvir']=np.append(dict_in['Mvir'],mod_funcs.Mvir(fit.params))
            dict_in['fit']=np.append(dict_in['fit'],fit)
            return dict_in

        gal=galaxy.galaxy('random')
        for key in gal.df1['galaxy']:
            gal=galaxy.galaxy(key)
            data=gal.data
            if (len(data['Vobs'])>=10 and data['Quality']!=3):
                if data['Vbulge'].all()==0:
                    CDM_dict_in['Vbulge_none']=results_in(CDM_dict_in['Vbulge_none'],data,bulge=False)
                elif data['Vbulge'].any()!=0:
                    CDM_dict_in['Vbulge']=results_in(CDM_dict_in['Vbulge'],data,bulge=True)
        return CDM_dict_in

##  The class to obtain rotation curve plots for given galaxies.
class plots:

    ## Define the constructor of the plots class.
    #  This defines the constructor of the plots class.
    #  @param self
    #  object pointer
    #  @param galaxies
    #  str[N] \n
    #  List of galaxy names (or 'random') to plot the given galaxies (or random galaxies) in the SPARC catalog.
    def __init__(self):
        return
        
    ## Define the plot of the rotation curves for given galaxies and for all CDM models.
    #  This defines the plot of the rotation curves for all given galaxies and for all CDM models.
    #  @param self
    #  object pointer
    #  @param fit_dict_in (optional)
    #  constants.fitting_dict instance \n
    #  Instance of the constants.fitting_dict class.
    #  Contains all the necessary rules and values to be used in fitting.
    #  @param size (optional)
    #  tuple \n
    #  Size of figure.
    #  @param save_file (optional)
    #  str \n
    #  File path to save plot.  Default is None, in which case file is not saved.
    #  @returns
    #  matplotlib.pyplot.figure[N,4] \n
    #  Figure of rotation curves for all given galaxies (N galaxies in total) and for all CDM halos (4 in total).
    def rotcurves_CDM_all(self,galaxies,fit_dict_in=fitting_dict_in,size=(20,20),save_file=None):

        ## @var fit_dict_in
        #  constants.fitting_dict instance \n
        #  Instance of the constants.fitting_dict class.
        #  @var size
        #  tuple \n
        #  Size of figure.
        self.fit_dict_in=fit_dict_in
        self.size=size
        N=len(galaxies)
        _,axs=plt.subplots(N,4,figsize=self.size)
        for i in range(N):
            gal=galaxy.galaxy(galaxies[i])
            data=gal.data
            halo_init=halo.halo('Burkert',data,fit_dict_in=self.fit_dict_in)
            fit_init=model_fit.model_fit(halo_init)
            fit_init.plot(halo_init,gal,axs[i,0])
            halo_init=halo.halo('DC14',data,fit_dict_in=self.fit_dict_in)
            fit_init=model_fit.model_fit(halo_init)
            fit_init.plot(halo_init,gal,axs[i,1])
            halo_init=halo.halo('Einasto',data,fit_dict_in=self.fit_dict_in)
            fit_init=model_fit.model_fit(halo_init)
            fit_init.plot(halo_init,gal,axs[i,2])
            halo_init=halo.halo('NFW',data,fit_dict_in=self.fit_dict_in)
            fit_init=model_fit.model_fit(halo_init)
            fit_init.plot(halo_init,gal,axs[i,3])
        if save_file!=None:
            plt.savefig(save_file)
        return

    ## Define the plot of the rotation curves for given galaxies and for the single ULDM and double ULDM models.
    #  This defines the plot of the rotation curves for all given galaxies and for the single ULDM and double ULDM models.
    #  @param self
    #  object pointer
    #  @param fit_dict_in (optional)
    #  constants.fitting_dict instance \n
    #  Instance of the constants.fitting_dict class.
    #  Contains all the necessary rules and values to be used in fitting.
    #  @param size (optional)
    #  tuple \n
    #  Size of figure.
    #  @param save_file (optional)
    #  str \n
    #  File path to save plot.  Default is None, in which case file is not saved.
    #  @returns
    #  matplotlib.pyplot.figure[N,2] \n
    #  Figure of rotation curves for all given galaxies (N galaxies in total) 
    #  and for the single and double ULDM 
    #  (will correspond to either summed or matched model depending on value for constants.fitting_dict.sol_match).
    def rotcurves_psi_all(self,galaxies,fit_dict_in=fitting_dict_in,size=(20,20),save_file=None):

        ## @var fit_dict_in
        #  constants.fitting_dict instance \n
        #  Instance of the constants.fitting_dict class.
        #  @var size
        #  tuple \n
        #  Size of figure.
        self.fit_dict_in=fit_dict_in
        self.size=size
        N=len(galaxies)
        _,axs=plt.subplots(N,2,figsize=self.size)
        for i in range(N):
            gal=galaxy.galaxy(galaxies[i])
            data=gal.data
            halo_init=halo.halo('psi_single',data,fit_dict_in=self.fit_dict_in)
            fit_init=model_fit.model_fit(halo_init,ULDM_fits=True)
            fit_init.plot(halo_init,gal,axs[i,0])
            halo_init=halo.halo('psi_multi',data,fit_dict_in=self.fit_dict_in)
            fit_init=model_fit.model_fit(halo_init,ULDM_fits=True)
            fit_init.plot(halo_init,gal,axs[i,1])
        if save_file!=None:
            plt.savefig(save_file)
        return

    ## Define the plot of the rotation curves for given galaxies and for all CDM models.
    #  This defines the plot of the rotation curves for all given galaxies and for all CDM models.
    #  For comparison with Pengfei Li et al 2020 ApJS 247 31 : 
    #  https://doi.org/10.3847/1538-4365/ab700e
    #  @param self
    #  object pointer
    #  @param fit_dict_in (optional)
    #  constants.fitting_dict instance \n
    #  Instance of the constants.fitting_dict class.
    #  Contains all the necessary rules and values to be used in fitting.
    #  @param size (optional)
    #  tuple \n
    #  Size of figure.
    #  @returns
    #  matplotlib.pyplot.figure[N,4] \n
    #  Figure of rotation curves for all given galaxies (N galaxies in total) and for all CDM halos (4 in total).
    def rotcurves_CDM_check(self,galaxies,fit_dict_in=fitting_dict_in,size=(20,20)):
        self.fit_dict_in=fit_dict_in
        self.size=size
        N=len(galaxies)
        _,axs=plt.subplots(N,4,figsize=self.size)
        for i in range(N):
            gal=galaxy.galaxy(galaxies[i])
            data=gal.data
            halo_init=halo.halo('Burkert',data,fit_dict_in=self.fit_dict_in)
            fit_init=model_fit.model_fit(halo_init)
            fit_init.plot(halo_init,gal,axs[i,0])
            halo_init=halo.halo('DC14',data,fit_dict_in=self.fit_dict_in)
            fit_init=model_fit.model_fit(halo_init)
            fit_init.plot(halo_init,gal,axs[i,1])
            halo_init=halo.halo('Einasto',data,fit_dict_in=self.fit_dict_in)
            fit_init=model_fit.model_fit(halo_init)
            fit_init.plot(halo_init,gal,axs[i,2])
            halo_init=halo.halo('NFW',data,fit_dict_in=self.fit_dict_in)
            fit_init=model_fit.model_fit(halo_init)
            fit_init.plot(halo_init,gal,axs[i,3])
        return

    ## Define the plot of the rotation curves for given galaxies and for the DC14 and NFW models.
    #  This defines the plot of the rotation curves for all given galaxies and for the DC14 and NFW models.
    #  For comparison with 
    #  Monthly Notices of the Royal Astronomical Society, Volume 466, Issue 2, April 2017, Pages 1648–1668 :
    #  https://doi.org/10.1093/mnras/stw3101
    #  @param self
    #  object pointer
    #  @param fit_dict_in (optional)
    #  constants.fitting_dict instance \n
    #  Instance of the constants.fitting_dict class.
    #  Contains all the necessary rules and values to be used in fitting.
    #  @param size (optional)
    #  tuple \n
    #  Size of figure.
    #  @returns
    #  matplotlib.pyplot.figure[N,2] \n
    #  Figure of rotation curves for all given galaxies (N galaxies in total) and for the DC14 and NFW models.
    def rotcurves_DC14_check(self,galaxies,fit_dict_in=fitting_dict_in,size=(20,20)):
        self.fit_dict_in=fit_dict_in
        self.size=size
        N=len(galaxies)
        _,axs=plt.subplots(N,2,figsize=self.size)
        for i in range(N):
            gal=galaxy.galaxy(galaxies[i])
            data=gal.data
            halo_init=halo.halo('DC14',data,fit_dict_in=self.fit_dict_in)
            fit_init=model_fit.model_fit(halo_init)
            fit_init.plot(halo_init,gal,axs[i,0])
            halo_init=halo.halo('NFW',data,fit_dict_in=self.fit_dict_in)
            fit_init=model_fit.model_fit(halo_init)
            fit_init.plot(halo_init,gal,axs[i,1])
        return

    ## Define the plot of the rotation curves for given galaxies and for the Einasto and NFW models.
    #  This defines the plot of the rotation curves for all given galaxies and for the Einasto and NFW models.
    #  For comparison with Nicolas Loizeau and Glennys R. Farrar 2021 ApJL 920 L10 :
    #  https://doi.org/10.3847/2041-8213/ac1bb7
    #  @param self
    #  object pointer
    #  @param fit_dict_in (optional)
    #  constants.fitting_dict instance \n
    #  Instance of the constants.fitting_dict class.
    #  Contains all the necessary rules and values to be used in fitting.
    #  @param size (optional)
    #  tuple \n
    #  Size of figure.
    #  @returns
    #  matplotlib.pyplot.figure[N,2] \n
    #  Figure of rotation curves for all given galaxies (N galaxies in total) and for the Einasto and NFW models.
    def rotcurves_Einasto_check(self,galaxies,fit_dict_in=fitting_dict_in,size=(20,20)):
        self.fit_dict_in=fit_dict_in
        self.size=size
        N=len(galaxies)
        _,axs=plt.subplots(N,2,figsize=self.size)
        for i in range(N):
            gal=galaxy.galaxy(galaxies[i])
            data=gal.data
            halo_init=halo.halo('Einasto',data,fit_dict_in=self.fit_dict_in)
            fit_init=model_fit.model_fit(halo_init)
            fit_init.plot(halo_init,gal,axs[i,0])
            halo_init=halo.halo('NFW',data,fit_dict_in=self.fit_dict_in)
            fit_init=model_fit.model_fit(halo_init)
            fit_init.plot(halo_init,gal,axs[i,1])
        return

    ## Define the plot of the BIC differences for different priors using CDM models.
    #  This defines the plot of the differences in BIC for different prior cases for each CDM model analyzed.
    #  @param self
    #  object pointer
    #  @param dict_in
    #  dictionary \n
    #  See fits_CDM_all.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be of the form: \n
    #  dict_in = {
    #  - 'uni_priors' : {...}
    #  - 'c200_priors_check' : {...}
    #  - 'v200_priors_check' : {...}
    #  - 'MLd_priors_check' : {...} 
    #  - 'MLb_priors_check' : {...}
    #  },
    #  where each {...} must be of the form: \n
    #  {...} = 
    #  - 'Burkert' : {....}
    #  - 'DC14' : {....}
    #  - 'Einasto' : {....}
    #  - 'NFW' : {....}
    #  Finally, each {....} must be an instance of results.results_CDM_all.fit.
    #  @param bins (optional)
    #  int \n
    #  Number of bins for histogram plot.
    #  @param lwin (optional)
    #  int \n
    #  Width of lines for plot.
    #  @param save_file (optional)
    #  str \n
    #  File path to save plot.  Default is None, in which case file is not saved.
    #  @returns
    #  matplotlib.pyplot.figure[4,4] \n
    #  Figure of the BIC differences for different priors using all CDM models.
    def BIC_diffs_CDM(self,dict_in,bins=250,lwin=3,save_file=None):
        _,axs=plt.subplots(4,4,figsize=(50,50))
        BIC_dict_in={}
        for key in dict_in:
            BIC_dict_in[key]={}
            for key1 in dict_in[key]:
                BIC_dict_in[key][key1]=np.concatenate((dict_in[key][key1]['Vbulge_none']['BIC'],
                                                    dict_in[key][key1]['Vbulge']['BIC']))
        title_tab=[r'$c_{200}$',r'$V_{200}$',r'$\tilde{\Upsilon}_d$',r'$\tilde{\Upsilon}_b$']
        prior_tab=['uni_priors','c200_priors_check','v200_priors_check','MLd_priors_check','MLb_priors_check']
        model_tab=['Burkert','DC14','Einasto','NFW']
        for i in range(len(model_tab)):
            for j in range(len(prior_tab)-1):
                diffin=BIC_dict_in['uni_priors'][model_tab[i]]-BIC_dict_in[prior_tab[j+1]][model_tab[i]]
                diffin=np.ma.masked_invalid(diffin)
                axs[j,i].hist(diffin,bins=bins,color='black')
                axs[j,i].set_xlabel(r'$\Delta \mathrm{BIC}$')
                axs[j,i].set_ylabel('Number of galaxies')
                axs[j,i].set_yscale('log')
                axs[j,i].set_title(str(model_tab[i]) + r': ' + str(title_tab[j]) + r' priors check')
                axs[i,j].axvline(0,linestyle='--',c='k',linewidth=lwin)
                axs[i,j].axvline(2,c='b',linewidth=lwin)
                axs[i,j].axvline(-2,c='b',linewidth=lwin)
                axs[i,j].axvline(6,c='r',linewidth=lwin)
                axs[i,j].axvline(-6,c='r',linewidth=lwin)
                axs[i,j].axvline(10,c='g',linewidth=lwin)
                axs[i,j].axvline(-10,c='g',linewidth=lwin)
                axs[i,j].set_xlim(-100,100)
                axs[i,j].tick_params(axis='both', which='major', labelsize=20, width=2.5, length=10)
                axs[i,j].tick_params(axis='both', which='minor', labelsize=15, width=2, length=5)
        if save_file!=None:
            plt.savefig(save_file)
        return
    
    ## Define the plot of the BIC differences for the CDM models.
    #  This defines the plot of the differences in BIC between each CDM model analyzed.
    #  @param self
    #  object pointer
    #  @param dict_in
    #  dictionary \n
    #  See fits_CDM_all.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be of the form: \n
    #  dict_in = { 
    #  - 'Burkert' : {...}
    #  - 'DC14' : {...}
    #  - 'Einasto' : {...}
    #  - 'NFW' : {...}
    #  },
    #  where each {...} must be an instance of results.results_CDM_all.fit.
    #  @param bins (optional)
    #  int \n
    #  Number of bins for histogram plot.
    #  @param lwin (optional)
    #  int \n
    #  Width of lines for plot.
    #  @param save_file (optional)
    #  str \n
    #  File path to save plot.  Default is None, in which case file is not saved.
    #  @returns
    #  matplotlib.pyplot.figure[2,3] \n
    #  Figure of the differences in BIC between each CDM model.
    def BIC_CDM(self,dict_in,bins=250,lwin=3,save_file=None):
        _,axs=plt.subplots(2,3,figsize=(35,20))
        model_tab=['Burkert','DC14','Einasto','NFW']
        BIC_dict={}
        for key in dict_in:
            BIC_dict[key]=np.concatenate((dict_in[key]['Vbulge_none']['BIC'],
                                            dict_in[key]['Vbulge']['BIC']))
        BIC_dict_diffs={}
        for i in range(len(model_tab)-1):
            BIC_dict_diffs[model_tab[i]]={}
        for i in range(len(model_tab)-1):
            BIC_dict_diffs[model_tab[0]][model_tab[i+1]]=BIC_dict[model_tab[0]]-BIC_dict[model_tab[i+1]]
        for i in range(len(model_tab)-2):
            BIC_dict_diffs[model_tab[1]][model_tab[i+2]]=BIC_dict[model_tab[1]]-BIC_dict[model_tab[i+2]]
        BIC_dict_diffs['Einasto']['NFW']=BIC_dict['Einasto']-BIC_dict['NFW']
        textstr_in={}
        interv_tab=np.asarray([[-np.inf,-10],[-10,-6],[-6,-2],[-2,2],[2,6],[6,10],[10,np.inf]])
        for key in BIC_dict_diffs:
            textstr_in[key]={}
            for key1 in BIC_dict_diffs[key]:
                textstr_in[key][key1]={}
                for i in range(len(interv_tab)):
                    sum_in=0
                    for j in range(len(BIC_dict_diffs[key][key1])):
                        if interv_tab[i,0]<BIC_dict_diffs[key][key1][j]<=interv_tab[i,1]:
                            sum_in+=1
                    if (interv_tab[i,0]!=-np.inf and interv_tab[i,1]!=np.inf):
                        textstr_in[key][key1][str(i)]=str(interv_tab[i,0])+r'$<\Delta\mathrm{BIC}\leq$'+str(interv_tab[i,1])+r': '+str(round(sum_in/len(BIC_dict_diffs[key][key1]),2))
                    elif interv_tab[i,0]==-np.inf:
                        textstr_in[key][key1][str(i)]=r'$\Delta\mathrm{BIC}\leq$'+str(interv_tab[i,1])+r': '+str(round(sum_in/len(BIC_dict_diffs[key][key1]),2))
                    else:
                        textstr_in[key][key1][str(i)]=str(interv_tab[i,0])+r'$<\Delta\mathrm{BIC}$'+r': '+str(round(sum_in/len(BIC_dict_diffs[key][key1]),2))
        for i in range(len(model_tab)-1):
            axs[0,i].hist(BIC_dict_diffs['Burkert'][model_tab[i+1]],bins=bins,color='black')
            textstr=textstr_in['Burkert'][model_tab[i+1]]['0']
            for j in range(len(interv_tab)-1):
                textstr = '\n'.join((
                    textstr,
                    textstr_in['Burkert'][model_tab[i+1]][str(j+1)]
                    ))
            axs[0,i].text(0.05,0.95,textstr,transform=axs[0,i].transAxes,fontsize=22,verticalalignment='top',
                bbox=dict(facecolor='yellow',alpha=0.8))
        for i in range(len(model_tab)-2):
            axs[1,i].hist(BIC_dict_diffs['DC14'][model_tab[i+2]],bins=bins,color='black')
            textstr=textstr_in['DC14'][model_tab[i+2]]['0']
            for j in range(len(interv_tab)-1):
                textstr = '\n'.join((
                    textstr,
                    textstr_in['DC14'][model_tab[i+2]][str(j+1)]
                    ))
            axs[1,i].text(0.05,0.95,textstr,transform=axs[1,i].transAxes,fontsize=22,verticalalignment='top',
                bbox=dict(facecolor='yellow',alpha=0.8))
        axs[1,2].hist(BIC_dict_diffs['Einasto']['NFW'],bins=bins,color='black')
        textstr=textstr_in['Einasto']['NFW']['0']
        for j in range(len(interv_tab)-1):
            textstr = '\n'.join((
                textstr,
                textstr_in['Einasto']['NFW'][str(j+1)]
                ))
        axs[1,2].text(0.05,0.95,textstr,transform=axs[1,2].transAxes,fontsize=22,verticalalignment='top',
            bbox=dict(facecolor='yellow',alpha=0.8))
        for i in range(2):
            for j in range(3):
                axs[i,j].axvline(0,linestyle='--',c='k',linewidth=lwin)
                axs[i,j].axvline(2,c='b',linewidth=lwin)
                axs[i,j].axvline(-2,c='b',linewidth=lwin)
                axs[i,j].axvline(6,c='r',linewidth=lwin)
                axs[i,j].axvline(-6,c='r',linewidth=lwin)
                axs[i,j].axvline(10,c='g',linewidth=lwin)
                axs[i,j].axvline(-10,c='g',linewidth=lwin)
                axs[i,j].set_ylabel('Number of galaxies')
                axs[i,j].set_xlim(-100,100)
                axs[i,j].tick_params(axis='both', which='major', labelsize=20, width=2.5, length=10)
                axs[i,j].tick_params(axis='both', which='minor', labelsize=15, width=2, length=5)
        axs[0,0].set_xlabel(r'$\mathrm{BIC}_{\mathrm{Burkert}} - \mathrm{BIC}_{\mathrm{DC14}}$')
        axs[0,1].set_xlabel(r'$\mathrm{BIC}_{\mathrm{Burkert}} - \mathrm{BIC}_{\mathrm{Einasto}}$')
        axs[0,2].set_xlabel(r'$\mathrm{BIC}_{\mathrm{Burkert}} - \mathrm{BIC}_{\mathrm{NFW}}$')
        axs[1,0].set_xlabel(r'$\mathrm{BIC}_{\mathrm{DC14}} - \mathrm{BIC}_{\mathrm{Einasto}}$')
        axs[1,1].set_xlabel(r'$\mathrm{BIC}_{\mathrm{DC14}} - \mathrm{BIC}_{\mathrm{NFW}}$')
        axs[1,2].set_xlabel(r'$\mathrm{BIC}_{\mathrm{Einasto}} - \mathrm{BIC}_{\mathrm{NFW}}$');
        if save_file!=None:
            plt.savefig(save_file)
        return

    ## Define the plot of the reduced chi-square comparisons for the CDM models.
    #  This defines the plot of the reduced chi-square comparisons between each CDM model analyzed.
    #  @param self
    #  object pointer
    #  @param dict_in
    #  dictionary \n
    #  See fits_CDM_all.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be of the form: \n
    #  dict_in = { 
    #  - 'Burkert' : {...}
    #  - 'DC14' : {...}
    #  - 'Einasto' : {...}
    #  - 'NFW' : {...}
    #  },
    #  where each {...} must be an instance of results.results_CDM_all.fit.
    #  @param splt (optional)
    #  int \n
    #  Point size for plot.
    #  @param lwin (optional)
    #  int \n
    #  Width of lines for plot.
    #  @param save_file (optional)
    #  str \n
    #  File path to save plot.  Default is None, in which case file is not saved.
    #  @returns
    #  matplotlib.pyplot.figure[2,3] \n
    #  Figure of the reduced chi-square comparisons between each CDM model.
    def chi_CDM(self,dict_in,splt=50,lwin=3,save_file=None):
        _,axs=plt.subplots(2,3,figsize=(35,20))
        model_tab=['Burkert','DC14','Einasto','NFW']
        label_tab=[r'$\chi^2_{\nu,\mathrm{Burkert}}$',r'$\chi^2_{\nu,\mathrm{DC14}}$',
                    r'$\chi^2_{\nu,\mathrm{Einasto}}$',r'$\chi^2_{\nu,\mathrm{NFW}}$']
        chisq_dict={}
        for key in dict_in:
            chisq_dict[key]=np.concatenate((dict_in[key]['Vbulge_none']['Chi_sq'],
                                            dict_in[key]['Vbulge']['Chi_sq']))
        for i in range(len(model_tab)-1):
            mask=np.isfinite(chisq_dict['Burkert'])
            mod_new=chisq_dict['Burkert'][mask]
            mod_new_1=chisq_dict[model_tab[i+1]][mask]
            xy = np.vstack([mod_new,mod_new_1])
            z = gaussian_kde(xy)(xy)
            axs[0,i].scatter(mod_new,mod_new_1,c=z,s=splt,cmap='Greys')
            axs[0,i].set_xlabel(label_tab[0])
            axs[0,i].set_ylabel(label_tab[i+1])
        for i in range(len(model_tab)-2):
            mask=np.isfinite(chisq_dict['DC14'])
            mod_new=chisq_dict['DC14'][mask]
            mod_new_1=chisq_dict[model_tab[i+2]][mask]
            xy = np.vstack([mod_new,mod_new_1])
            z = gaussian_kde(xy)(xy)
            axs[1,i].scatter(mod_new,mod_new_1,c=z,s=splt,cmap='Greys')
            axs[1,i].set_xlabel(label_tab[1])
            axs[1,i].set_ylabel(label_tab[i+2])
        mask=np.isfinite(chisq_dict['Einasto'])
        mod_new=chisq_dict['Einasto'][mask]
        mod_new_1=chisq_dict['NFW'][mask]
        xy = np.vstack([mod_new,mod_new_1])
        z = gaussian_kde(xy)(xy)
        axs[1,2].scatter(mod_new,mod_new_1,c=z,s=splt,cmap='Greys')
        axs[1,2].set_xlabel(label_tab[2])
        axs[1,2].set_ylabel(label_tab[3])
        for i in range(2):
            for j in range(3):
                axs[i,j].axline([0,0],[1,1],linestyle='--',c='k')
                axs[i,j].axhline(1,c='b',linewidth=lwin)
                axs[i,j].axvline(1,c='b',linewidth=lwin)
                axs[i,j].set_xscale('log')
                axs[i,j].set_yscale('log')
                axs[i,j].grid(visible=True,axis='both',c='k',alpha=0.4)
                axs[i,j].tick_params(axis='both', which='major', labelsize=20, width=2.5, length=10)
                axs[i,j].tick_params(axis='both', which='minor', labelsize=15, width=2, length=5)
        if save_file!=None:
            plt.savefig(save_file)
        return
    
    ## Define the plot of the reduced chi-square distributions for the CDM models.
    #  This defines the plot of the reduced chi-square distribution for each CDM model analyzed.
    #  @param self
    #  object pointer
    #  @param dict_in
    #  dictionary \n
    #  See fits_CDM_all.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be of the form: \n
    #  dict_in = { 
    #  - 'Burkert' : {...}
    #  - 'DC14' : {...}
    #  - 'Einasto' : {...}
    #  - 'NFW' : {...}
    #  },
    #  where each {...} must be an instance of results.results_CDM_all.fit.
    #  @param bins (optional)
    #  int \n
    #  Number of bins for histogram plot.
    #  @param save_file (optional)
    #  str \n
    #  File path to save plot.  Default is None, in which case file is not saved.
    #  @returns
    #  matplotlib.pyplot.figure[2,2] \n
    #  Figure of the reduced chi-square distributions for each CDM model.
    def chi_dist_CDM(self,dict_in,bins=50,save_file=None):
        _,axs=plt.subplots(2,2,figsize=(20,21))
        in_tab=[]
        in_tabb=[]
        for key in dict_in:
            in_tab.append(dict_in[key]['Vbulge_none'])
            in_tabb.append(dict_in[key]['Vbulge'])
        title_tab=['Burkert','DC14','Einasto','NFW']
        index_in=0
        for i in range(2):
            for j in range(2):
                axs[i,j].hist(in_tab[index_in]['Chi_sq'],bins=bins,label=r'No $V_\mathrm{bulge}$');
                axs[i,j].hist(in_tabb[index_in]['Chi_sq'],bins=bins,label=r'With $V_\mathrm{bulge}$');
                axs[i,j].set_xlabel(r'$\chi^2_\nu$')
                axs[i,j].legend()
                axs[i,j].set_ylabel('Number of galaxies')
                axs[i,j].set_title(title_tab[index_in])
                axs[i,j].tick_params(axis='both', which='major', labelsize=20, width=2.5, length=10)
                axs[i,j].tick_params(axis='both', which='minor', labelsize=15, width=2, length=5)
                index_in+=1
        if save_file!=None:
            plt.savefig(save_file)
        return

    ## Define the plot of the parameter distributions for the CDM models.
    #  This defines the plot of the parameter distributions for each CDM model analyzed.
    #  @param self
    #  object pointer
    #  @param dict_in
    #  dictionary \n
    #  See fits_CDM_all.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be of the form: \n
    #  dict_in = { 
    #  - 'Burkert' : {...}
    #  - 'DC14' : {...}
    #  - 'Einasto' : {...}
    #  - 'NFW' : {...}
    #  },
    #  where each {...} must be an instance of results.results_CDM_all.fit.
    #  @param bins (optional)
    #  int \n
    #  Number of bins for histogram plot.
    #  @param save_file (optional)
    #  str \n
    #  File path to save plot.  Default is None, in which case file is not saved.
    #  @returns
    #  matplotlib.pyplot.figure[4,4] \n
    #  Figure of the parameter distributions for each CDM model.
    def params_dist_CDM(self,dict_in,bins=50,save_file=None):
        _,axs=plt.subplots(4,4,figsize=(50,50))
        in_tab=[]
        in_tabb=[]
        for key in dict_in:
            in_tab.append(dict_in[key]['Vbulge_none']['params'])
            in_tabb.append(dict_in[key]['Vbulge']['params'])
        title_tab=['Burkert','DC14','Einasto','NFW']
        label_tab=[r'$c_{200}$',r'$V_{200} \, [\mathrm{km/s}]$',r'$\tilde{\Upsilon}_d \, [M_\odot/L_\odot]$']
        params_tab=['c200','v200','MLd','MLb']
        for i in range(len(in_tab)):
            input_arr=in_tab[i]
            input_arrb=in_tabb[i]
            for j in range(3):
                input_arr_in=np.asarray([])
                for k in range(len(input_arr[params_tab[j]])):
                        input_arr_in=np.append(input_arr_in,input_arr[params_tab[j]][k].nominal_value)
                input_arr_inb=np.asarray([])
                for k in range(len(input_arrb[params_tab[j]])):
                        input_arr_inb=np.append(input_arr_inb,input_arrb[params_tab[j]][k].nominal_value)
                axs[j,i].hist(input_arr_in,bins=bins,label=r'No $V_\mathrm{bulge}$');
                axs[j,i].hist(input_arr_inb,bins=bins,label=r'With $V_\mathrm{bulge}$');
                axs[j,i].set_xlabel(label_tab[j]) 
            input_arr_inb=np.asarray([])
            for k in range(len(input_arrb[params_tab[3]])):
                input_arr_inb=np.append(input_arr_inb,input_arrb[params_tab[3]][k].nominal_value)
            axs[3,i].hist(input_arr_inb,bins=bins,label=r'With $V_\mathrm{bulge}$');
            axs[3,i].set_xlabel(r'$\tilde{\Upsilon}_b \, [M_\odot/L_\odot]$')     
            for j in range(4):
                axs[j,i].legend()
                axs[j,i].set_ylabel('Number of galaxies')
                axs[j,i].set_title(title_tab[i]) 
                axs[j,i].tick_params(axis='both', which='major', labelsize=20, width=2.5, length=10)
                axs[j,i].tick_params(axis='both', which='minor', labelsize=15, width=2, length=5)
        if save_file!=None:
            plt.savefig(save_file)
        return

    ## Define the plot of the reduced chi-square contours for the NFW model.
    #  This defines the plot of the reduced chi-square contours in the 
    #  $\tilde{\Upsilon}_d-c_{200}$ plane for the NFW model.
    #  @param self
    #  object pointer
    #  @param save_file (optional)
    #  str \n
    #  File path to save plot.  Default is None, in which case file is not saved.
    #  @returns
    #  matplotlib.pyplot.figure[2,2] \n
    #  Figure of the reduced chi-square contours in the 
    #  $\tilde{\Upsilon}_d-c_{200}$ plane for the NFW model.
    def MLd_degen_CDM(self,save_file=None):
        fig,axs=plt.subplots(2,2,figsize=(20,20))
        name_tab=['UGC00731','UGC05764','NGC3917','UGC05253']
        c200_tab=np.asarray([np.arange(5,13,1),np.arange(1,30,1),np.arange(1,5,1),np.arange(1,20,1)])
        MLD_tab=np.asarray([np.arange(0.01,6,0.05),np.arange(0.01,6,0.05),np.arange(0.01,6,0.05),np.arange(0.005,6,0.05)])
        plt_0=0
        plt_1=0
        for i in range(len(name_tab)):
            gal=galaxy.galaxy(name_tab[i])
            halo_init=halo.halo('NFW',gal.data)
            mod_fit_in=model_fit.model_fit(halo_init)
            fit_in=mod_fit_in.fit(halo_init.params,gal.data)
            v200_in=fit_in.params['v200'].value
            params=halo_init.params
            c200tab=c200_tab[i]
            MLDtab=MLD_tab[i]
            all_tab=np.asarray([])
            for j in range(len(c200tab)):
                for k in range(len(MLDtab)):
                    params['c200'].value=c200tab[j]
                    params['MLd'].value=MLDtab[k]
                    params['v200'].value=v200_in
                    mod_fit_in=model_fit.model_fit(halo_init)
                    res=np.sum(mod_fit_in.residual(params,gal.data['Radius'],gal.data))
                    all_tab=np.append(all_tab,[c200tab[j],MLDtab[k],res**2])
                    all_tab=np.reshape(all_tab,(int(len(all_tab)/3),3))
            pcm=axs[plt_1,plt_0].tricontourf(all_tab[:,0],all_tab[:,1],np.log10(all_tab[:,2]),50)
            fig.colorbar(pcm,ax=axs[plt_1,plt_0])
            axs[plt_1,plt_0].set_xlabel(r'$c_{200}$')
            axs[plt_1,plt_0].set_ylabel(r'$\tilde{\Upsilon}_d \, [M_{\odot}/L_{\odot}]$');
            axs[plt_1,plt_0].set_title(str(name_tab[i])+r': $V_{200} = $'+str(round(v200_in)));
            if plt_0<1:
                plt_0+=1
            else:
                plt_0=0
                plt_1+=1
        if save_file!=None:
            plt.savefig(save_file)
        return

    ## Define the plot of the empirical relations for the CDM models.
    #  This defines the plot of the empirical relations for each CDM models analyzed.
    #  @param self
    #  object pointer
    #  @param save_file (optional)
    #  str \n
    #  File path to save plot.  Default is None, in which case file is not saved.
    #  @returns
    #  matplotlib.pyplot.figure[4,4] \n
    #  Figure of the empirical relations for each CDM model.
    def relations_CDM(self,dict_in,save_file=None):
        model_tab=['Burkert','DC14','Einasto','NFW']
        gbar={}
        gtot={}
        gMOND={}
        fit_results={}
        for mod in model_tab:
            gbar[mod]=np.asarray([])
            gtot[mod]=np.asarray([])
            gMOND[mod]=np.asarray([])
            fit_results[mod]=0
        for key in gbar:
            fit_init=model_fit.grar_fit(key)
            fit=fit_init.fit()
            gbar[key]=fit['gbar']
            gtot[key]=fit['gtot']
            gMOND[key]=fit['gMOND']
            fit_results[key]=fit['fit']
        params_arr=unumpy.uarray([fit_results['Burkert'].params['gdag'].value,fit_results['DC14'].params['gdag'].value,
            fit_results['Einasto'].params['gdag'].value,fit_results['NFW'].params['gdag'].value],
            [fit_results['Burkert'].params['gdag'].stderr,fit_results['DC14'].params['gdag'].stderr,
            fit_results['Einasto'].params['gdag'].stderr,fit_results['NFW'].params['gdag'].stderr])
        params_arr_log=unumpy.log10(params_arr)
        model_tab=np.asarray(['Burkert','DC14','Einasto','NFW'])
        label_tab=np.asarray([r'Model : $\log_{10} \, g^{\dagger} = $' 
            + str(round(params_arr_log[0].nominal_value,2))
            + r'$\pm$' + str(round(params_arr_log[0].std_dev,4)) 
            + r'$\, [\log_{10}\left(\mathrm{m} \, s^{-2}\right)]$',
            r'Model : $\log_{10} \, g^{\dagger} = $' 
            + str(round(params_arr_log[1].nominal_value,2))
            + r'$\pm$' + str(round(params_arr_log[1].std_dev,4)) 
            + r'$\, [\log_{10}\left(\mathrm{m} \, s^{-2}\right)]$',
            r'Model : $\log_{10} \, g^{\dagger} = $' 
            + str(round(params_arr_log[2].nominal_value,2))
            + r'$\pm$' + str(round(params_arr_log[2].std_dev,4)) 
            + r'$\, [\log_{10}\left(\mathrm{m} \, s^{-2}\right)]$',
            r'Model : $\log_{10} \, g^{\dagger} = $' 
            + str(round(params_arr_log[3].nominal_value,2))
            + r'$\pm$' + str(round(params_arr_log[3].std_dev,4)) 
            + r'$\, [\log_{10}\left(\mathrm{m} \, s^{-2}\right)]$'])
        _,axs=plt.subplots(4,4,figsize=(50,50))
        for i in range(4):
            axs[0,i].scatter(gbar[model_tab[i]],gtot[model_tab[i]],label=r'Data points',c='k')
            axs[0,i].scatter(gbar[model_tab[i]],gMOND[model_tab[i]],
                label=r'MOND : $\log_{10} \, g^{\dagger} = $' 
                    + str(round(np.log10(1.2*1e-10),2))
                    + r'$\, [\log_{10}\left(\mathrm{m} \, s^{-2}\right)]$')
            fit_init=model_fit.grar_fit(model_tab[i])
            axs[0,i].scatter(gbar[model_tab[i]],fit_init.grar_model(gbar[model_tab[i]],fit_results[model_tab[i]].params),
                label=label_tab[i])
            axs[0,i].axline([0,0],[10**(-8.5),10**(-8.5)],linestyle='-',c='k')
            axs[0,i].set_xlabel(r'$g_\mathrm{bar} \, [\mathrm{m}/\mathrm{s}^2]$')
            axs[0,i].set_ylabel(r'$g_\mathrm{tot} \, [\mathrm{m}/\mathrm{s}^2]$')
            axs[0,i].set_xscale('log')
            axs[0,i].set_yscale('log')
            axs[0,i].set_title(str(model_tab[i]))
            axs[0,i].grid(visible=True,axis='both',c='k',alpha=0.4)
            axs[0,i].legend()
        model_tab=['Burkert','DC14','Einasto','NFW']
        plt_0=0
        for mod in model_tab:
            in0=unumpy.log10(dict_in[mod]['Vbulge_none']['params']['c200'])
            in1=unumpy.log10(dict_in[mod]['Vbulge']['params']['c200'])
            c200_in=np.concatenate((in0,in1))
            in0=unumpy.log10(dict_in[mod]['Vbulge_none']['Mvir'])
            in1=unumpy.log10(dict_in[mod]['Vbulge']['Mvir'])
            m200_in=np.concatenate((in0,in1))
            args_sort=np.argsort(m200_in)
            m200_in=m200_in[args_sort]
            c200_in=c200_in[args_sort]
            c200CMR_Wa=np.asarray([])
            for i in range(len(m200_in)):
                c200CMR_Wa=np.append(c200CMR_Wa,model_fit.conc_mass_rel_Wa(10**(m200_in[i])))
            c200CMR_Wa_val=np.asarray([])
            for i in range(len(c200CMR_Wa)):
                c200CMR_Wa_val=np.append(c200CMR_Wa_val,c200CMR_Wa[i].nominal_value)
            c200CMR_Wa_val=np.log10(c200CMR_Wa_val)
            c200CMR_Du=model_fit.conc_mass_rel_Du(10**(m200_in))
            c200CMR_Du_val=np.asarray([])
            for i in range(len(c200CMR_Du)):
                c200CMR_Du_val=np.append(c200CMR_Du_val,c200CMR_Du[i].nominal_value)
            m200val=np.asarray([])
            for i in range(len(m200_in)):
                m200val=np.append(m200val,m200_in[i].nominal_value)
            axs[1,plt_0].plot(m200val,c200CMR_Wa_val,label='CMR_W',c='k',linestyle='solid')
            axs[1,plt_0].plot(m200val,c200CMR_Du_val,label='CMR_D',c='k',linestyle='dashed')
            m200val=np.asarray([])
            c200val=np.asarray([])
            m200err=np.asarray([])
            c200err=np.asarray([])
            for i in range(len(m200_in)):
                if (m200_in[i].std_dev!=np.inf and c200_in[i].std_dev!=np.inf and m200_in[i].std_dev<m200_in[i].nominal_value):
                    m200val=np.append(m200val,m200_in[i].nominal_value)
                    m200err=np.append(m200err,m200_in[i].std_dev)
                    c200val=np.append(c200val,c200_in[i].nominal_value)
                    c200err=np.append(c200err,c200_in[i].std_dev)    
            axs[1,plt_0].errorbar(m200val,c200val,xerr=m200err,yerr=c200err,c='k',fmt='o')
            m200val=np.asarray([])
            c200val=np.asarray([])
            m200err=np.asarray([])
            for i in range(len(m200_in)):
                if (m200_in[i].std_dev!=np.inf and c200_in[i].std_dev==np.inf and m200_in[i].std_dev<m200_in[i].nominal_value):
                    m200val=np.append(m200val,m200_in[i].nominal_value)
                    m200err=np.append(m200err,m200_in[i].std_dev)
                    c200val=np.append(c200val,c200_in[i].nominal_value)  
            axs[1,plt_0].errorbar(m200val,c200val,xerr=m200err,yerr=None,c='b',fmt='v')
            m200val=np.asarray([])
            c200val=np.asarray([])
            c200err=np.asarray([])
            for i in range(len(m200_in)):
                if ((m200_in[i].std_dev==np.inf and c200_in[i].std_dev!=np.inf) 
                        or (m200_in[i].std_dev!=np.inf and m200_in[i].std_dev>m200_in[i].nominal_value and c200_in[i].std_dev!=np.inf)):
                    m200val=np.append(m200val,m200_in[i].nominal_value)
                    c200val=np.append(c200val,c200_in[i].nominal_value)
                    c200err=np.append(c200err,c200_in[i].std_dev)    
            axs[1,plt_0].errorbar(m200val,c200val,xerr=None,yerr=c200err,c='r',fmt='s')
            m200val=np.asarray([])
            c200val=np.asarray([])
            for i in range(len(m200_in)):
                if ((m200_in[i].std_dev==np.inf and c200_in[i].std_dev==np.inf)
                        or (m200_in[i].std_dev!=np.inf and m200_in[i].std_dev>m200_in[i].nominal_value and c200_in[i].std_dev==np.inf)):
                    m200val=np.append(m200val,m200_in[i].nominal_value)
                    c200val=np.append(c200val,c200_in[i].nominal_value)  
            axs[1,plt_0].scatter(m200val,c200val,c='g',marker='x')
            axs[1,plt_0].set_ylabel(r'$\log_{10}c_{200}$')
            axs[1,plt_0].set_xlabel(r'$\log_{10}\left(M_{200} \, [M_{\odot}]\right)$')
            axs[1,plt_0].set_title(mod)
            axs[1,plt_0].legend()
            axs[1,plt_0].set_xlim(8.5,15.5)
            axs[1,plt_0].set_ylim(-0.25,2.25)
            axs[1,plt_0].grid(visible=True,axis='both',c='k',alpha=0.4)
            plt_0+=1
        plt_0=0
        for mod in model_tab:
            in0=np.log10(dict_in[mod]['Vbulge_none']['params']['Vf'])
            in1=np.log10(dict_in[mod]['Vbulge']['params']['Vf'])
            Vf_in=np.concatenate((in0,in1))
            axs[2,plt_0].plot(Vf_in,model_fit.BTFR(10**(Vf_in)),label='BTFR',c='k',linestyle='solid')
            mass_in=dict_in[mod]['Vbulge_none']['params']['mstar']+dict_in[mod]['Vbulge_none']['params']['mgas']
            in0=unumpy.log10((mass_in)*1e9)
            mass_in=dict_in[mod]['Vbulge']['params']['mstar']+dict_in[mod]['Vbulge']['params']['mgas']
            in1=unumpy.log10((mass_in)*1e9)
            mstar_in=np.concatenate((in0,in1))
            mstar_in_val=np.asarray([])
            mstar_in_err=np.asarray([])
            Vf_in_val=np.asarray([])
            for i in range(len(mstar_in)):
                if (mstar_in[i].std_dev!=np.inf):
                    mstar_in_val=np.append(mstar_in_val,mstar_in[i].nominal_value)
                    mstar_in_err=np.append(mstar_in_err,mstar_in[i].std_dev)
                    Vf_in_val=np.append(Vf_in_val,Vf_in[i])
            axs[2,plt_0].errorbar(Vf_in_val,mstar_in_val,xerr=None,yerr=mstar_in_err,c='k',fmt='o')
            mstar_in_val=np.asarray([])
            Vf_in_val=np.asarray([])
            for i in range(len(mstar_in)):
                if (mstar_in[i].std_dev==np.inf):
                    mstar_in_val=np.append(mstar_in_val,mstar_in[i].nominal_value)
                    Vf_in_val=np.append(Vf_in_val,Vf_in[i])
            axs[2,plt_0].scatter(Vf_in_val,mstar_in_val,c='g',marker='x')
            axs[2,plt_0].set_ylabel(r'$\log_{10} \left(M_{\mathrm{b}} \, [M_\odot]\right)$')
            axs[2,plt_0].set_xlabel(r'$\log_{10} \left(V_f \, [km/s]\right)$')
            axs[2,plt_0].set_title(mod)
            axs[2,plt_0].legend()
            axs[2,plt_0].set_ylim(7.5,12.5)
            axs[2,plt_0].grid(visible=True,axis='both',c='k',alpha=0.4)
            plt_0+=1
        plt_0=0
        for mod in model_tab:
            in0=dict_in[mod]['Vbulge_none']['params']['mstar']
            in1=dict_in[mod]['Vbulge']['params']['mstar']
            mstar_in=np.concatenate((in0,in1))
            in0=dict_in[mod]['Vbulge_none']['Mvir']
            in1=dict_in[mod]['Vbulge']['Mvir']
            m200_in=np.concatenate((in0,in1))
            args_sort=np.argsort(m200_in)
            m200_in=m200_in[args_sort]
            mstarAMR=unumpy.log10(model_fit.abund_match_rel(m200_in))
            m200_in_log=unumpy.log10(m200_in)
            mstar_in_log=unumpy.log10(mstar_in*1e9)
            m200_in_val=np.asarray([])
            mstarAMR_val=np.asarray([])
            for i in range(len(m200_in)):
                m200_in_val=np.append(m200_in_val,m200_in_log[i].nominal_value)
                mstarAMR_val=np.append(mstarAMR_val,mstarAMR[i].nominal_value)
            axs[3,plt_0].plot(m200_in_val,mstarAMR_val,label='AMR',c='k',linestyle='solid')
            m200_in_val=np.asarray([])
            mstar_in_val=np.asarray([])
            m200_in_err=np.asarray([])
            mstar_in_err=np.asarray([])
            for i in range(len(m200_in)):
                if (m200_in[i].std_dev!=np.inf and mstar_in[i].std_dev!=np.inf and m200_in[i].std_dev<m200_in[i].nominal_value):
                    m200_in_val=np.append(m200_in_val,m200_in_log[i].nominal_value)
                    mstar_in_val=np.append(mstar_in_val,mstar_in_log[i].nominal_value)
                    m200_in_err=np.append(m200_in_err,m200_in_log[i].std_dev)
                    mstar_in_err=np.append(mstar_in_err,mstar_in_log[i].std_dev)
            axs[3,plt_0].errorbar(m200_in_val,mstar_in_val,xerr=m200_in_err,yerr=mstar_in_err,c='k',fmt='o')  
            m200_in_val=np.asarray([])
            mstar_in_val=np.asarray([])
            m200_in_err=np.asarray([])
            for i in range(len(m200_in)):
                if (m200_in[i].std_dev!=np.inf and mstar_in[i].std_dev==np.inf and m200_in[i].std_dev<m200_in[i].nominal_value):
                    m200_in_val=np.append(m200_in_val,m200_in_log[i].nominal_value)
                    mstar_in_val=np.append(mstar_in_val,mstar_in_log[i].nominal_value)
                    m200_in_err=np.append(m200_in_err,m200_in_log[i].std_dev)
            axs[3,plt_0].errorbar(m200_in_val,mstar_in_val,xerr=m200_in_err,yerr=None,c='b',fmt='v')  
            m200_in_val=np.asarray([])
            mstar_in_val=np.asarray([])
            mstar_in_err=np.asarray([])
            for i in range(len(m200_in)):
                if ((m200_in[i].std_dev==np.inf and mstar_in[i].std_dev!=np.inf) 
                        or (m200_in[i].std_dev!=np.inf and m200_in[i].std_dev>m200_in[i].nominal_value and mstar_in[i].std_dev!=np.inf)):
                    m200_in_val=np.append(m200_in_val,m200_in_log[i].nominal_value)
                    mstar_in_val=np.append(mstar_in_val,mstar_in_log[i].nominal_value)
                    mstar_in_err=np.append(mstar_in_err,mstar_in_log[i].std_dev)
            axs[3,plt_0].errorbar(m200_in_val,mstar_in_val,xerr=None,yerr=mstar_in_err,c='r',fmt='s') 
            m200_in_val=np.asarray([])
            mstar_in_val=np.asarray([])
            for i in range(len(m200_in)):
                if ((m200_in[i].std_dev==np.inf and mstar_in[i].std_dev==np.inf)
                        or (m200_in[i].std_dev!=np.inf and m200_in[i].std_dev>m200_in[i].nominal_value and mstar_in[i].std_dev==np.inf)):
                    m200_in_val=np.append(m200_in_val,m200_in_log[i].nominal_value)
                    mstar_in_val=np.append(mstar_in_val,mstar_in_log[i].nominal_value)
            axs[3,plt_0].scatter(m200_in_val,mstar_in_val,c='g',marker='x')
            axs[3,plt_0].set_ylabel(r'$\log_{10}\left(M_* \, [M_{\odot}]\right)$')
            axs[3,plt_0].set_xlabel(r'$\log_{10}\left(M_{200} \, [M_{\odot}]\right)$')
            axs[3,plt_0].set_title(mod)
            axs[3,plt_0].legend()
            axs[3,plt_0].set_xlim(8.5,15.5)
            axs[3,plt_0].set_ylim(4.5,12.5)
            axs[3,plt_0].grid(visible=True,axis='both',c='k',alpha=0.4)
            plt_0+=1
        if save_file!=None:
            plt.savefig(save_file)
        return

    ## Define the plot of the BIC differences between the ULDM (particle mass free) and Einasto models.
    #  This defines the plot of the differences in BIC vs. ULDM particle mass between
    #  each of the ULDM (particle mass free) models and the Einasto model.
    #  @param self
    #  object pointer
    #  @param psis_dict_in
    #  dictionary \n
    #  The dictionary for the single flavor models.
    #  See fits_Einasto_all.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be of the form: \n
    #  psis_dict_in = { 
    #  - 'Summed' : {...}
    #  - 'Matched' : {...}
    #  },
    #  where each {...} must be an instance of results.results_psi_single_all.fit.
    #  @param psim_dict_in
    #  dictionary \n
    #  The dictionary for the double flavor models.
    #  See fits_Einasto_all.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be of the form: \n
    #  psim_dict_in = { 
    #  - 'Summed' : {...}
    #  - 'Matched' : {...}
    #  },
    #  where each {...} must be an instance of results.results_psi_multi_all.fit.
    #  @param CDM_dict_in
    #  dictionary \n
    #  The dictionary for the Einasto.
    #  See fits_Einasto_all.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be an instance of results.results_CDM_all.fit.
    #  @param splt (optional)
    #  int \n
    #  Point size for plot.
    #  @param lwin (optional)
    #  int \n
    #  Width of lines for plot.
    #  @param save_file (optional)
    #  str \n
    #  File path to save plot.  Default is None, in which case file is not saved.
    #  @returns
    #  matplotlib.pyplot.figure[3,2] \n
    #  Figure of the differences in BIC vs. particle mass between each ULDM (particle mass free) model and the Einasto model.
    def BIC_psi_mfree(self,psis_dict_in,psim_dict_in,CDM_dict_in,splt=50,lwin=3,save_file=None):
        _,axs=plt.subplots(3,2,figsize=(25,35))
        interv_tab=np.asarray([[-np.inf,-10],[-10,-6],[-6,-2],[-2,2],[2,6],[6,10],[10,np.inf]])
        CDM_in=np.concatenate((CDM_dict_in['Vbulge_none']['BIC'],CDM_dict_in['Vbulge']['BIC']))
        plt_0=0
        for key in psis_dict_in:
            psi_in=np.concatenate((psis_dict_in[key]['Vbulge_none']['BIC'],
                psis_dict_in[key]['Vbulge']['BIC']))
            mtab_in=np.concatenate((psis_dict_in[key]['Vbulge_none']['params']['m22'],
                psis_dict_in[key]['Vbulge']['params']['m22']))
            mtab_vals=np.asarray([])
            for i in range(len(mtab_in)):
                mtab_vals=np.append(mtab_vals,mtab_in[i].nominal_value)
            diff_in=CDM_in-psi_in
            mask=np.isfinite(diff_in)
            mtab_new=mtab_vals[mask]
            diff_new=diff_in[mask]
            xy = np.vstack([mtab_new,diff_new])
            z = gaussian_kde(xy)(xy)
            axs[0,plt_0].scatter(mtab_new,diff_new,c=z,s=splt,cmap='Greys')
            axs[0,plt_0].set_xlabel(r'$m \,[10^{-22}\,\mathrm{eV}]$')
            axs[0,plt_0].set_title(r'Single, ' + str(key))
            textstr_in={}
            for i in range(len(interv_tab)):
                sum_in=0
                for j in range(len(diff_in)):
                    if interv_tab[i,0]<diff_in[j]<=interv_tab[i,1]:
                        sum_in+=1
                if (interv_tab[i,0]!=-np.inf and interv_tab[i,1]!=np.inf):
                    textstr_in[str(i)]=str(interv_tab[i,0])+r'$<\Delta\mathrm{BIC}\leq$'+str(interv_tab[i,1])+r': '+str(round(sum_in/len(diff_in),2))
                elif interv_tab[i,0]==-np.inf:
                    textstr_in[str(i)]=r'$\Delta\mathrm{BIC}\leq$'+str(interv_tab[i,1])+r': '+str(round(sum_in/len(diff_in),2))
                else:
                    textstr_in[str(i)]=str(interv_tab[i,0])+r'$<\Delta\mathrm{BIC}$'+r': '+str(round(sum_in/len(diff_in),2))
            textstr=textstr_in['0']
            for i in range(len(interv_tab)-1):
                textstr = '\n'.join((
                    textstr,
                    textstr_in[str(i+1)],
                    ))
            axs[0,plt_0].text(0.05,0.95,textstr,transform=axs[0,plt_0].transAxes,fontsize=22,verticalalignment='top',
                bbox=dict(facecolor='yellow',alpha=0.8))
            plt_0+=1
        plt_0=0
        for key in psim_dict_in:
            psi_in=np.concatenate((psim_dict_in[key]['Vbulge_none']['BIC'],
                psim_dict_in[key]['Vbulge']['BIC']))
            mtab_1_in=np.concatenate((psim_dict_in[key]['Vbulge_none']['params']['m22'],
                psim_dict_in[key]['Vbulge']['params']['m22']))
            mtab_2_in=np.concatenate((psim_dict_in[key]['Vbulge_none']['params']['m22_2'],
                psim_dict_in[key]['Vbulge']['params']['m22_2']))
            mtab_vals=np.asarray([])
            for i in range(len(mtab_1_in)):
                mtab_vals=np.append(mtab_vals,mtab_1_in[i].nominal_value)
            diff_in=CDM_in-psi_in
            mask=np.isfinite(diff_in)
            mtab_new=mtab_vals[mask]
            diff_new=diff_in[mask]
            xy = np.vstack([mtab_new,diff_new])
            z = gaussian_kde(xy)(xy)
            axs[1,plt_0].scatter(mtab_new,diff_new,c=z,s=splt,cmap='Greys')
            axs[1,plt_0].set_xlabel(r'$m_1 \,[10^{-22}\,\mathrm{eV}]$')
            axs[1,plt_0].set_title(r'Double, ' + str(key))
            textstr_in={}
            for i in range(len(interv_tab)):
                sum_in=0
                for j in range(len(diff_in)):
                    if interv_tab[i,0]<diff_in[j]<=interv_tab[i,1]:
                        sum_in+=1
                if (interv_tab[i,0]!=-np.inf and interv_tab[i,1]!=np.inf):
                    textstr_in[str(i)]=str(interv_tab[i,0])+r'$<\Delta\mathrm{BIC}\leq$'+str(interv_tab[i,1])+r': '+str(round(sum_in/len(diff_in),2))
                elif interv_tab[i,0]==-np.inf:
                    textstr_in[str(i)]=r'$\Delta\mathrm{BIC}\leq$'+str(interv_tab[i,1])+r': '+str(round(sum_in/len(diff_in),2))
                else:
                    textstr_in[str(i)]=str(interv_tab[i,0])+r'$<\Delta\mathrm{BIC}$'+r': '+str(round(sum_in/len(diff_in),2))
            textstr=textstr_in['0']
            for i in range(len(interv_tab)-1):
                textstr = '\n'.join((
                    textstr,
                    textstr_in[str(i+1)],
                    ))
            axs[1,plt_0].text(0.05,0.95,textstr,transform=axs[1,plt_0].transAxes,fontsize=22,verticalalignment='top',
                bbox=dict(facecolor='yellow',alpha=0.8))
            mtab_in=mtab_1_in/mtab_2_in
            mtab_vals=np.asarray([])
            for i in range(len(mtab_in)):
                mtab_vals=np.append(mtab_vals,mtab_in[i].nominal_value)
            diff_in=CDM_in-psi_in
            mask=np.isfinite(diff_in)
            mtab_new=mtab_vals[mask]
            diff_new=diff_in[mask]
            xy = np.vstack([mtab_new,diff_new])
            z = gaussian_kde(xy)(xy)
            axs[2,plt_0].scatter(mtab_new,diff_new,c=z,s=splt,cmap='Greys')
            axs[2,plt_0].set_xlabel(r'$m_1/m_2$')
            axs[2,plt_0].set_title(r'Double, ' + str(key))
            textstr=textstr_in['0']
            for i in range(len(interv_tab)-1):
                textstr = '\n'.join((
                    textstr,
                    textstr_in[str(i+1)],
                    ))
            axs[2,plt_0].text(0.05,0.95,textstr,transform=axs[2,plt_0].transAxes,fontsize=22,verticalalignment='top',
                bbox=dict(facecolor='yellow',alpha=0.8))
            plt_0+=1
        for i in range(3):
            for j in range(2):
                axs[i,j].axhline(0,linestyle='--',c='k',linewidth=lwin)
                axs[i,j].axhline(2,c='b',linewidth=lwin)
                axs[i,j].axhline(-2,c='b',linewidth=lwin)
                axs[i,j].axhline(6,c='r',linewidth=lwin)
                axs[i,j].axhline(-6,c='r',linewidth=lwin)
                axs[i,j].axhline(10,c='g',linewidth=lwin)
                axs[i,j].axhline(-10,c='g',linewidth=lwin)
                axs[i,j].set_ylabel(r'BIC$_{\mathrm{Einasto}}$-BIC$_{\mathrm{ULDM}}$')
                axs[i,j].set_xscale('log')
                axs[i,j].set_ylim(-100,100)
                axs[i,j].grid(visible=True,axis='both',c='k',alpha=0.4)
        if save_file!=None:
            plt.savefig(save_file)
        return

    ## Define the plot of the reduced chi-square for the ULDM (particle mass free) and Einasto models.
    #  This defines the plot of the reduced chi-square comparisons between 
    #  each of the ULDM (particle mass free) models and the Einasto model.
    #  @param self
    #  object pointer
    #  @param psis_dict_in
    #  dictionary \n
    #  The dictionary for the single flavor models.
    #  See fits_Einasto_all.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be of the form: \n
    #  psis_dict_in = { 
    #  - 'Summed' : {...}
    #  - 'Matched' : {...}
    #  },
    #  where each {...} must be an instance of results.results_psi_single_all.fit.
    #  @param psim_dict_in
    #  dictionary \n
    #  The dictionary for the double flavor models.
    #  See fits_Einasto_all.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be of the form: \n
    #  psim_dict_in = { 
    #  - 'Summed' : {...}
    #  - 'Matched' : {...}
    #  },
    #  where each {...} must be an instance of results.results_psi_multi_all.fit.
    #  @param CDM_dict_in
    #  dictionary \n
    #  The dictionary for the Einasto.
    #  See fits_Einasto_all.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be an instance of results.results_CDM_all.fit.
    #  @param splt (optional)
    #  int \n
    #  Point size for plot.
    #  @param lwin (optional)
    #  int \n
    #  Width of lines for plot.
    #  @param save_file (optional)
    #  str \n
    #  File path to save plot.  Default is None, in which case file is not saved.
    #  @returns
    #  matplotlib.pyplot.figure[2,2] \n
    #  Figure of the reduced chi-square comparisons between each of the ULDM (particle mass free) models and the Einasto model. 
    def chi_psi_mfree(self,psis_dict_in,psim_dict_in,CDM_dict_in,splt=50,lwin=3,save_file=None):
        _,axs=plt.subplots(2,2,figsize=(30,25))
        model_tab=['psis','psim']
        chisq_in={'CDM':{},'psis':{},'psim':{}}
        chisq_in['CDM']=np.concatenate((CDM_dict_in['Vbulge_none']['Chi_sq'],CDM_dict_in['Vbulge']['Chi_sq']))
        for key in psis_dict_in:
            chisq_in['psis'][key]=np.concatenate((psis_dict_in[key]['Vbulge_none']['Chi_sq'],
                psis_dict_in[key]['Vbulge']['Chi_sq']))
            chisq_in['psim'][key]=np.concatenate((psim_dict_in[key]['Vbulge_none']['Chi_sq'],
                psim_dict_in[key]['Vbulge']['Chi_sq']))
        plt_0=0
        plt_1=0
        lab_ind=0
        for i in range(len(model_tab)):
            for key in chisq_in['psis']:
                mask=np.isfinite(chisq_in[model_tab[i]][key])
                mod_new=chisq_in[model_tab[i]][key][mask]
                mod_new_1=chisq_in['CDM'][mask]
                mask=np.isfinite(mod_new_1)
                mod_new=mod_new[mask]
                mod_new_1=mod_new_1[mask]
                xy = np.vstack([mod_new,mod_new_1])
                z = gaussian_kde(xy)(xy)
                axs[plt_1,plt_0].scatter(mod_new,mod_new_1,c=z,s=splt,cmap='Greys')
                axs[plt_1,plt_0].set_ylabel(r'$\chi^2_{\nu,\mathrm{Einasto}}$')
                axs[plt_1,plt_0].set_xlabel(r'$\chi^2_{\nu,\mathrm{ULDM}}$')
                if plt_0<1:
                    plt_0+=1
                else:
                    plt_0=0
                    plt_1+=1
                lab_ind+=1
        for i in range(2):
            for j in range(2):
                axs[i,j].axline([0,0],[1,1],linestyle='--',c='k',linewidth=lwin)
                axs[i,j].axhline(1,c='b',linewidth=lwin)
                axs[i,j].axvline(1,c='b',linewidth=lwin)
                axs[i,j].set_xscale('log')
                axs[i,j].set_yscale('log')
                axs[i,j].grid(visible=True,axis='both',c='k',alpha=0.4)
        axs[0,0].set_title(r'Single, Summed')
        axs[0,1].set_title(r'Single, Matched')
        axs[1,0].set_title(r'Double, Summed')
        axs[1,1].set_title(r'Double, Matched');
        if save_file!=None:
            plt.savefig(save_file)
        return

    ## Define the plot of the soliton-halo (SH) relation for the ULDM (particle mass free) models.
    #  This defines the plot of the SH relation vs. particle mass for each of the ULDM (particle mass free) models.
    #  @param self
    #  object pointer
    #  @param psis_dict_in
    #  dictionary \n
    #  The dictionary for the single flavor models.
    #  See fits_Einasto_all.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be of the form: \n
    #  psis_dict_in = { 
    #  - 'Summed' : {...}
    #  - 'Matched' : {...}
    #  },
    #  where each {...} must be an instance of results.results_psi_single_all.fit.
    #  @param psim_dict_in
    #  dictionary \n
    #  The dictionary for the double flavor models.
    #  See fits_Einasto_all.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be of the form: \n
    #  psim_dict_in = { 
    #  - 'Summed' : {...}
    #  - 'Matched' : {...}
    #  },
    #  where each {...} must be an instance of results.results_psi_multi_all.fit.
    #  @param save_file (optional)
    #  str \n
    #  File path to save plot.  Default is None, in which case file is not saved.
    #  @returns
    #  matplotlib.pyplot.figure[3,2] \n
    #  Figure of the SH relation vs. particle mass for each of the ULDM (particle mass free) models.
    def Msol_psi_mfree(self,psis_dict_in,psim_dict_in,save_file=None):
        _,axs=plt.subplots(3,2,figsize=(25,45))
        plt_0=0
        for key in psis_dict_in:
            x_tab=np.concatenate((psis_dict_in[key]['Vbulge_none']['params']['Msol'],
                psis_dict_in[key]['Vbulge']['params']['Msol']))
            mtab_in=np.concatenate((psis_dict_in[key]['Vbulge_none']['params']['m22'],
                psis_dict_in[key]['Vbulge']['params']['m22']))
            y_tab=np.concatenate((psis_dict_in[key]['Vbulge_none']['Mhalo'],
                psis_dict_in[key]['Vbulge']['Mhalo']))
            y_tab=1.4e9*mtab_in**(-1)*(y_tab/1e12)**(1/3)
            y_tab_vals=np.asarray([])
            for i in range(len(x_tab)):
                y_tab_vals=np.append(y_tab_vals,y_tab[i].nominal_value)
            x_tab=x_tab/y_tab_vals
            x_tab=unumpy.log10(x_tab)
            mtab_in=unumpy.log10(mtab_in)
            x_tab_vals=np.asarray([])
            x_tab_err=np.asarray([])
            mtab_vals=np.asarray([])
            mtab_err=np.asarray([])
            for i in range(len(x_tab)):
                if (x_tab[i].std_dev!=np.inf and mtab_in[i].std_dev!=np.inf):
                    x_tab_vals=np.append(x_tab_vals,x_tab[i].nominal_value)
                    x_tab_err=np.append(x_tab_err,x_tab[i].std_dev)
                    mtab_vals=np.append(mtab_vals,mtab_in[i].nominal_value)
                    mtab_err=np.append(mtab_err,mtab_in[i].std_dev)
            axs[0,plt_0].errorbar(mtab_vals,x_tab_vals,xerr=mtab_err,yerr=x_tab_err,c='k',fmt='o')
            mtab_vals=np.asarray([])
            x_tab_vals=np.asarray([])
            mtab_err=np.asarray([])
            for i in range(len(x_tab)):
                if (mtab_in[i].std_dev!=np.inf and x_tab[i].std_dev==np.inf):
                    mtab_vals=np.append(mtab_vals,mtab_in[i].nominal_value)
                    mtab_err=np.append(mtab_err,mtab_in[i].std_dev)
                    x_tab_vals=np.append(x_tab_vals,x_tab[i].nominal_value)  
            axs[0,plt_0].errorbar(mtab_vals,x_tab_vals,xerr=mtab_err,yerr=None,c='b',fmt='v')
            mtab_vals=np.asarray([])
            x_tab_vals=np.asarray([])
            x_tab_err=np.asarray([])
            for i in range(len(x_tab)):
                if (mtab_in[i].std_dev==np.inf and x_tab[i].std_dev!=np.inf):
                    mtab_vals=np.append(mtab_vals,mtab_in[i].nominal_value)
                    x_tab_vals=np.append(x_tab_vals,x_tab[i].nominal_value)
                    x_tab_err=np.append(x_tab_err,x_tab[i].std_dev)    
            axs[0,plt_0].errorbar(mtab_vals,x_tab_vals,xerr=None,yerr=x_tab_err,c='r',fmt='s')
            mtab_vals=np.asarray([])
            x_tab_vals=np.asarray([])
            for i in range(len(x_tab)):
                if (mtab_in[i].std_dev==np.inf and x_tab[i].std_dev==np.inf):
                    mtab_vals=np.append(mtab_vals,mtab_in[i].nominal_value)
                    x_tab_vals=np.append(x_tab_vals,x_tab[i].nominal_value)  
            axs[0,plt_0].scatter(mtab_vals,x_tab_vals,c='g',marker='x')
            axs[0,plt_0].set_ylabel(r'$\log_{10}\left(M_\mathrm{sol}/M_{\mathrm{SH}}\right)$')
            axs[0,plt_0].set_xlabel(r'$\log_{10} \left(m \, [10^{-22}\,\mathrm{eV}]\right)$')
            axs[0,plt_0].set_title('Single, ' + str(key))
            plt_0+=1
        plt_0=0
        for key in psim_dict_in:
            x_tab=np.concatenate((psim_dict_in[key]['Vbulge_none']['params']['Msol'],
                psim_dict_in[key]['Vbulge']['params']['Msol']))
            mtab_1_in=np.concatenate((psim_dict_in[key]['Vbulge_none']['params']['m22'],
                psim_dict_in[key]['Vbulge']['params']['m22']))
            y_tab=np.concatenate((psim_dict_in[key]['Vbulge_none']['Mhalo_1'],
                psim_dict_in[key]['Vbulge']['Mhalo_1']))
            y_tab=1.4e9*mtab_1_in**(-1)*(y_tab/1e12)**(1/3)
            mtab_in=mtab_1_in
            y_tab_vals=np.asarray([])
            for i in range(len(x_tab)):
                y_tab_vals=np.append(y_tab_vals,y_tab[i].nominal_value)
            x_tab=x_tab/y_tab_vals
            x_tab=unumpy.log10(x_tab)
            mtab_in=unumpy.log10(mtab_in)
            x_tab_vals=np.asarray([])
            x_tab_err=np.asarray([])
            mtab_vals=np.asarray([])
            mtab_err=np.asarray([])
            for i in range(len(x_tab)):
                if (x_tab[i].std_dev!=np.inf and mtab_in[i].std_dev!=np.inf):
                    x_tab_vals=np.append(x_tab_vals,x_tab[i].nominal_value)
                    x_tab_err=np.append(x_tab_err,x_tab[i].std_dev)
                    mtab_vals=np.append(mtab_vals,mtab_in[i].nominal_value)
                    mtab_err=np.append(mtab_err,mtab_in[i].std_dev)
                    y_tab_vals=np.append(y_tab_vals,y_tab[i].nominal_value)
            axs[1,plt_0].errorbar(mtab_vals,x_tab_vals,xerr=mtab_err,yerr=x_tab_err,c='k',fmt='o')
            mtab_vals=np.asarray([])
            x_tab_vals=np.asarray([])
            mtab_err=np.asarray([])
            for i in range(len(x_tab)):
                if (mtab_in[i].std_dev!=np.inf and x_tab[i].std_dev==np.inf):
                    mtab_vals=np.append(mtab_vals,mtab_in[i].nominal_value)
                    mtab_err=np.append(mtab_err,mtab_in[i].std_dev)
                    x_tab_vals=np.append(x_tab_vals,x_tab[i].nominal_value)  
            axs[1,plt_0].errorbar(mtab_vals,x_tab_vals,xerr=mtab_err,yerr=None,c='b',fmt='v')
            mtab_vals=np.asarray([])
            x_tab_vals=np.asarray([])
            x_tab_err=np.asarray([])
            for i in range(len(x_tab)):
                if (mtab_in[i].std_dev==np.inf and x_tab[i].std_dev!=np.inf):
                    mtab_vals=np.append(mtab_vals,mtab_in[i].nominal_value)
                    x_tab_vals=np.append(x_tab_vals,x_tab[i].nominal_value)
                    x_tab_err=np.append(x_tab_err,x_tab[i].std_dev)    
            axs[1,plt_0].errorbar(mtab_vals,x_tab_vals,xerr=None,yerr=x_tab_err,c='r',fmt='s')
            mtab_vals=np.asarray([])
            x_tab_vals=np.asarray([])
            for i in range(len(x_tab)):
                if (mtab_in[i].std_dev==np.inf and x_tab[i].std_dev==np.inf):
                    mtab_vals=np.append(mtab_vals,mtab_in[i].nominal_value)
                    x_tab_vals=np.append(x_tab_vals,x_tab[i].nominal_value)  
            axs[1,plt_0].scatter(mtab_vals,x_tab_vals,c='g',marker='x')
            axs[1,plt_0].set_ylabel(r'$\log_{10}\left(M_\mathrm{sol}/M_{\mathrm{SH}}\right)$')
            axs[1,plt_0].set_xlabel(r'$\log_{10} \left(m \, [10^{-22}\,\mathrm{eV}]\right)$')
            axs[1,plt_0].set_title('Double, ' + str(key))
            x_tab=np.concatenate((psim_dict_in[key]['Vbulge_none']['params']['Msol_2'],
                psim_dict_in[key]['Vbulge']['params']['Msol_2']))
            mtab_2_in=np.concatenate((psim_dict_in[key]['Vbulge_none']['params']['m22_2'],
                psim_dict_in[key]['Vbulge']['params']['m22_2']))
            y_tab=np.concatenate((psim_dict_in[key]['Vbulge_none']['Mhalo_2'],
                psim_dict_in[key]['Vbulge']['Mhalo_2']))
            y_tab=1.4e9*mtab_2_in**(-1)*(y_tab/1e12)**(1/3)
            mtab_in=mtab_1_in/mtab_2_in
            y_tab_vals=np.asarray([])
            for i in range(len(x_tab)):
                y_tab_vals=np.append(y_tab_vals,y_tab[i].nominal_value)
            x_tab=x_tab/y_tab_vals
            x_tab=unumpy.log10(x_tab)
            mtab_in=unumpy.log10(mtab_in)
            x_tab_vals=np.asarray([])
            x_tab_err=np.asarray([])
            mtab_vals=np.asarray([])
            mtab_err=np.asarray([])
            for i in range(len(x_tab)):
                if (x_tab[i].std_dev!=np.inf and mtab_in[i].std_dev!=np.inf):
                    x_tab_vals=np.append(x_tab_vals,x_tab[i].nominal_value)
                    x_tab_err=np.append(x_tab_err,x_tab[i].std_dev)
                    mtab_vals=np.append(mtab_vals,mtab_in[i].nominal_value)
                    mtab_err=np.append(mtab_err,mtab_in[i].std_dev)
                    y_tab_vals=np.append(y_tab_vals,y_tab[i].nominal_value)
            axs[2,plt_0].errorbar(mtab_vals,x_tab_vals,xerr=mtab_err,yerr=x_tab_err,c='k',fmt='o')
            mtab_vals=np.asarray([])
            x_tab_vals=np.asarray([])
            mtab_err=np.asarray([])
            for i in range(len(x_tab)):
                if (mtab_in[i].std_dev!=np.inf and x_tab[i].std_dev==np.inf):
                    mtab_vals=np.append(mtab_vals,mtab_in[i].nominal_value)
                    mtab_err=np.append(mtab_err,mtab_in[i].std_dev)
                    x_tab_vals=np.append(x_tab_vals,x_tab[i].nominal_value)  
            axs[2,plt_0].errorbar(mtab_vals,x_tab_vals,xerr=mtab_err,yerr=None,c='b',fmt='v')
            mtab_vals=np.asarray([])
            x_tab_vals=np.asarray([])
            x_tab_err=np.asarray([])
            for i in range(len(x_tab)):
                if (mtab_in[i].std_dev==np.inf and x_tab[i].std_dev!=np.inf):
                    mtab_vals=np.append(mtab_vals,mtab_in[i].nominal_value)
                    x_tab_vals=np.append(x_tab_vals,x_tab[i].nominal_value)
                    x_tab_err=np.append(x_tab_err,x_tab[i].std_dev)    
            axs[2,plt_0].errorbar(mtab_vals,x_tab_vals,xerr=None,yerr=x_tab_err,c='r',fmt='s')
            mtab_vals=np.asarray([])
            x_tab_vals=np.asarray([])
            for i in range(len(x_tab)):
                if (mtab_in[i].std_dev==np.inf and x_tab[i].std_dev==np.inf):
                    mtab_vals=np.append(mtab_vals,mtab_in[i].nominal_value)
                    x_tab_vals=np.append(x_tab_vals,x_tab[i].nominal_value)  
            axs[2,plt_0].scatter(mtab_vals,x_tab_vals,c='g',marker='x')
            axs[2,plt_0].set_ylabel(r'$\log_{10}\left(M_\mathrm{sol}/M_{\mathrm{SH}}\right)$')
            axs[2,plt_0].set_xlabel(r'$\log_{10}\left(m_1/m_2\right)$')
            axs[2,plt_0].set_title('Double, ' + str(key))
            plt_0+=1
        for i in range(2):
            for j in range(3):
                axs[j,i].axhline(np.log10(1),linestyle='--',c='k')
                axs[j,i].axhline(np.log10(2),c='b')
                axs[j,i].axhline(np.log10(0.5),c='b')
                axs[j,i].set_xlim(-3.5,3.5)
                axs[j,i].set_ylim(-2.5,2.5)
                axs[j,i].grid(visible=True,axis='both',c='k',alpha=0.4)
        if save_file!=None:
            plt.savefig(save_file)
        return

    ## Define the plot of the reduced chi-square distributions for the ULDM (particle mass free) models.
    #  This defines the plot of the reduced chi-square distributions for each of the ULDM (particle mass free) models.
    #  @param self
    #  object pointer
    #  @param psis_dict_in
    #  dictionary \n
    #  The dictionary for the single flavor models.
    #  See fits_Einasto_all.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be of the form: \n
    #  psis_dict_in = { 
    #  - 'Summed' : {...}
    #  - 'Matched' : {...}
    #  },
    #  where each {...} must be an instance of results.results_psi_single_all.fit.
    #  @param psim_dict_in
    #  dictionary \n
    #  The dictionary for the double flavor models.
    #  See fits_Einasto_all.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be of the form: \n
    #  psim_dict_in = { 
    #  - 'Summed' : {...}
    #  - 'Matched' : {...}
    #  },
    #  where each {...} must be an instance of results.results_psi_multi_all.fit.
    #  @param bins (optional)
    #  int \n
    #  Number of bins for histogram plot.
    #  @param save_file (optional)
    #  str \n
    #  File path to save plot.  Default is None, in which case file is not saved.
    #  @returns
    #  matplotlib.pyplot.figure[2,2] \n
    #  Figure of the reduced chi-square distributions for each of the ULDM (particle mass free) models.
    def chi_dist_psi_mfree(self,psis_dict_in,psim_dict_in,bins=50,save_file=None):
        _,axs=plt.subplots(2,2,figsize=(25,25))
        model_tab=['Single, Summed','Single, Matched','Double, Summed','Double, Matched']
        in_tab=[]
        in_tabb=[]
        for key in psis_dict_in:
            in_tab.append(psis_dict_in[key]['Vbulge_none'])
            in_tabb.append(psis_dict_in[key]['Vbulge'])
        for key in psim_dict_in:
            in_tab.append(psim_dict_in[key]['Vbulge_none'])
            in_tabb.append(psim_dict_in[key]['Vbulge'])
        k=0
        j=0
        for i in range(len(in_tab)):
            mask=np.isfinite(in_tab[i]['Chi_sq'])
            mod_new=in_tab[i]['Chi_sq'][mask]
            mask=np.isfinite(in_tabb[i]['Chi_sq'])
            mod_new_1=in_tabb[i]['Chi_sq'][mask]
            axs[j,k].hist(mod_new,bins,label=r'No $V_\mathrm{bulge}$')
            axs[j,k].hist(mod_new_1,bins,label=r'With $V_\mathrm{bulge}$')
            axs[j,k].set_xlabel(r'$\chi^2_{\nu,\mathrm{ULDM}}$')
            axs[j,k].set_title(model_tab[i])
            axs[j,k].set_ylabel('Number of galaxies')
            if k<1:
                k+=1
            else:
                k=0
                j+=1
        if save_file!=None:
            plt.savefig(save_file)
        return

    ## Define the plot of the parameter distributions for the ULDM (particle mass free) models.
    #  This defines the plot of the parameter distributions for each of the ULDM (particle mass free) models.
    #  @param self
    #  object pointer
    #  @param psis_dict_in
    #  dictionary \n
    #  The dictionary for the single flavor models.
    #  See fits_Einasto_all.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be of the form: \n
    #  psis_dict_in = { 
    #  - 'Summed' : {...}
    #  - 'Matched' : {...}
    #  },
    #  where each {...} must be an instance of results.results_psi_single_all.fit.
    #  @param psim_dict_in
    #  dictionary \n
    #  The dictionary for the double flavor models.
    #  See fits_Einasto_all.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be of the form: \n
    #  psim_dict_in = { 
    #  - 'Summed' : {...}
    #  - 'Matched' : {...}
    #  },
    #  where each {...} must be an instance of results.results_psi_multi_all.fit.
    #  @param bins (optional)
    #  int \n
    #  Number of bins for histogram plot.
    #  @param save_file (optional)
    #  str \n
    #  File path to save plot.  Default is None, in which case file is not saved.
    #  @returns
    #  matplotlib.pyplot.figure[4,4] \n
    #  Figure of the parameter distributions for each of the ULDM (particle mass free) models.
    def params_dist_psi_mfree(self,psis_dict_in,psim_dict_in,bins=50,save_file=None):
        _,axs=plt.subplots(4,4,figsize=(50,50))
        title_tab=['Single, Summed','Single, Matched','Double, Summed','Double, Matched']
        in_tab=[]
        in_tabb=[]
        for key in psis_dict_in:
            in_tab.append(psis_dict_in[key]['Vbulge_none']['params'])
            in_tabb.append(psis_dict_in[key]['Vbulge']['params'])
        for key in psim_dict_in:
            in_tab.append(psim_dict_in[key]['Vbulge_none']['params'])
            in_tabb.append(psim_dict_in[key]['Vbulge']['params'])
        for i in range(len(in_tab)):
            input_arr=in_tab[i]
            input_arrb=in_tabb[i]
            in_vals=np.asarray([])
            for j in range(len(input_arr['c200'])):
                    in_vals=np.append(in_vals,input_arr['c200'][j].nominal_value)
            in_valsb=np.asarray([])
            for j in range(len(input_arrb['c200'])):
                    in_valsb=np.append(in_valsb,input_arrb['c200'][j].nominal_value)
            axs[0,i].hist(in_vals,bins,label=r'No $V_\mathrm{bulge}$');
            axs[0,i].hist(in_valsb,bins,label=r'With $V_\mathrm{bulge}$');
            axs[0,i].set_xlabel(r'$c_{200}$')
            in_vals=np.asarray([])
            for j in range(len(input_arr['v200'])):
                    in_vals=np.append(in_vals,input_arr['v200'][j].nominal_value)
            in_valsb=np.asarray([])
            for j in range(len(input_arrb['v200'])):
                    in_valsb=np.append(in_valsb,input_arrb['v200'][j].nominal_value)
            axs[1,i].hist(in_vals,bins,label=r'No $V_\mathrm{bulge}$');
            axs[1,i].hist(in_valsb,bins,label=r'With $V_\mathrm{bulge}$');
            axs[1,i].set_xlabel(r'$V_{200} \, [\mathrm{km/s}]$') 
        for i in range(len(in_tab)-2):
            input_arr=in_tab[i]
            input_arrb=in_tabb[i]
            in_vals=np.asarray([])
            for j in range(len(input_arr['MLd'])):
                    in_vals=np.append(in_vals,input_arr['MLd'][j].nominal_value)
            in_valsb=np.asarray([])
            for j in range(len(input_arrb['MLd'])):
                    in_valsb=np.append(in_valsb,input_arrb['MLd'][j].nominal_value)
            axs[2,i].hist(in_vals,bins,label=r'No $V_\mathrm{bulge}$');
            axs[2,i].hist(in_valsb,bins,label=r'With $V_\mathrm{bulge}$');
            axs[2,i].set_xlabel(r'$\tilde{\Upsilon}_d \, [M_\odot/L_\odot]$')  
            in_valsb=np.asarray([])
            for j in range(len(input_arrb['MLb'])):
                    in_valsb=np.append(in_valsb,input_arrb['MLb'][j].nominal_value) 
            axs[3,i].hist(in_valsb,bins,label=r'With $V_\mathrm{bulge}$');
            axs[3,i].set_xlabel(r'$\tilde{\Upsilon}_b \, [M_\odot/L_\odot]$')    
        for i in range(len(in_tab)-2):
            input_arr=in_tab[i+2]
            input_arrb=in_tabb[i+2]
            in_vals=np.asarray([])
            for j in range(len(input_arr['c200_2'])):
                    in_vals=np.append(in_vals,input_arr['c200_2'][j].nominal_value)
            in_valsb=np.asarray([])
            for j in range(len(input_arrb['c200_2'])):
                    in_valsb=np.append(in_valsb,input_arrb['c200_2'][j].nominal_value)
            axs[0,i+2].hist(in_vals,bins,label=r'No $V_\mathrm{bulge}$');
            axs[0,i+2].hist(in_valsb,bins,label=r'With $V_\mathrm{bulge}$');
            in_vals=np.asarray([])
            for j in range(len(input_arr['v200_2'])):
                    in_vals=np.append(in_vals,input_arr['v200_2'][j].nominal_value)
            in_valsb=np.asarray([])
            for j in range(len(input_arrb['v200_2'])):
                    in_valsb=np.append(in_valsb,input_arrb['v200_2'][j].nominal_value)
            axs[1,i+2].hist(in_vals,bins,label=r'No $V_\mathrm{bulge}$');
            axs[1,i+2].hist(in_valsb,bins,label=r'With $V_\mathrm{bulge}$');
            in_vals=np.asarray([])
            for j in range(len(input_arr['MLd'])):
                    in_vals=np.append(in_vals,input_arr['MLd'][j].nominal_value)
            in_valsb=np.asarray([])
            for j in range(len(input_arrb['MLd'])):
                    in_valsb=np.append(in_valsb,input_arrb['MLd'][j].nominal_value)
            axs[2,i+2].hist(in_vals,bins,label=r'No $V_\mathrm{bulge}$');
            axs[2,i+2].hist(in_valsb,bins,label=r'With $V_\mathrm{bulge}$');
            axs[2,i+2].set_xlabel(r'$\tilde{\Upsilon}_d \, [M_\odot/L_\odot]$')  
            in_valsb=np.asarray([])
            for j in range(len(input_arrb['MLb'])):
                    in_valsb=np.append(in_valsb,input_arrb['MLd'][j].nominal_value)
            axs[3,i+2].hist(in_valsb,bins,label=r'With $V_\mathrm{bulge}$');
            axs[3,i+2].set_xlabel(r'$\tilde{\Upsilon}_b \, [M_\odot/L_\odot]$')  
        for i in range(len(in_tab)):
            input_arr=in_tab[i]
            input_arrb=in_tabb[i]
            for j in range(len(in_tab)):
                axs[j,i].legend()
                axs[j,i].set_ylabel('Number of galaxies')
                axs[j,i].set_title(title_tab[i]) 
        if save_file!=None:      
            plt.savefig(save_file)
        return

    ## Define the plot of the empirical relations for the ULDM (particle mass free) models.
    #  This defines the plot of the empirical relations for each of the ULDM (particle mass free) models.
    #  @param self
    #  object pointer
    #  @param psis_dict_in
    #  dictionary \n
    #  The dictionary for the single flavor models.
    #  See fits_Einasto_all.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be of the form: \n
    #  psis_dict_in = { 
    #  - 'Summed' : {...}
    #  - 'Matched' : {...}
    #  },
    #  where each {...} must be an instance of results.results_psi_single_all.fit.
    #  @param psim_dict_in
    #  dictionary \n
    #  The dictionary for the double flavor models.
    #  See fits_Einasto_all.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be of the form: \n
    #  psim_dict_in = { 
    #  - 'Summed' : {...}
    #  - 'Matched' : {...}
    #  },
    #  where each {...} must be an instance of results.results_psi_multi_all.fit.
    #  @param save_file (optional)
    #  str \n
    #  File path to save plot.  Default is None, in which case file is not saved.
    #  @returns
    #  matplotlib.pyplot.figure[4,4] \n
    #  Figure of the empirical relations for each of the ULDM (particle mass free) models.
    def relations_psi_mfree(self,psis_dict_in,psim_dict_in,save_file=None):
        model_tab=['psi_single','psi_multi']
        model_tab_1=['Summed','Matched']
        gbar={}
        gtot={}
        gMOND={}
        fit_results={}
        for mod in model_tab:
            gbar[mod]={}
            gtot[mod]={}
            gMOND[mod]={}
            fit_results[mod]={}
            for mod1 in model_tab_1:
                gbar[mod][mod1]=np.asarray([])
                gtot[mod][mod1]=np.asarray([])
                gMOND[mod][mod1]=np.asarray([])
                fit_results[mod][mod1]=0
        fit_dict_ex=constants.fitting_dict(sol_mfree=True)
        for key in gbar:
            fit_init=model_fit.grar_fit(key,ULDM_fits=True,fit_dict_in=fit_dict_ex)
            fit=fit_init.fit()
            gbar[key]['Summed']=fit['gbar']
            gtot[key]['Summed']=fit['gtot']
            gMOND[key]['Summed']=fit['gMOND']
            fit_results[key]['Summed']=fit['fit']
        fit_dict_ex=constants.fitting_dict(sol_mfree=True,sol_match=True)
        for key in gbar:
            fit_init=model_fit.grar_fit(key,ULDM_fits=True,fit_dict_in=fit_dict_ex)
            fit=fit_init.fit()
            gbar[key]['Matched']=fit['gbar']
            gtot[key]['Matched']=fit['gtot']
            gMOND[key]['Matched']=fit['gMOND']
            fit_results[key]['Matched']=fit['fit']
        print(fit_results['psi_single']['Summed'].params)
        print(fit_results['psi_single']['Matched'].params)
        print(fit_results['psi_multi']['Summed'].params)
        print(fit_results['psi_multi']['Matched'].params)
        params_arr=unumpy.uarray([fit_results['psi_single']['Summed'].params['gdag'].value,
            fit_results['psi_single']['Matched'].params['gdag'].value,
            fit_results['psi_multi']['Summed'].params['gdag'].value,
        fit_results['psi_multi']['Matched'].params['gdag'].value],
        [fit_results['psi_single']['Summed'].params['gdag'].stderr,
            fit_results['psi_single']['Matched'].params['gdag'].stderr,
            fit_results['psi_multi']['Summed'].params['gdag'].stderr,
        fit_results['psi_multi']['Matched'].params['gdag'].stderr])
        params_arr_log=unumpy.log10(params_arr)
        label_tab=np.asarray([r'Model : $\log_{10} \, g^{\dagger} = $' 
            + str(round(params_arr_log[0].nominal_value,2))
            + r'$\pm$' + str(round(params_arr_log[0].std_dev,4)) 
            + r'$\, [\log_{10}\left(\mathrm{m} \, s^{-2}\right)]$',
            r'Model : $\log_{10} \, g^{\dagger} = $' 
            + str(round(params_arr_log[1].nominal_value,2))
            + r'$\pm$' + str(round(params_arr_log[1].std_dev,4)) 
            + r'$\, [\log_{10}\left(\mathrm{m} \, s^{-2}\right)]$',
            r'Model : $\log_{10} \, g^{\dagger} = $' 
            + str(round(params_arr_log[2].nominal_value,2))
            + r'$\pm$' + str(round(params_arr_log[2].std_dev,4)) 
            + r'$\, [\log_{10}\left(\mathrm{m} \, s^{-2}\right)]$',
            r'Model : $\log_{10} \, g^{\dagger} = $' 
            + str(round(params_arr_log[3].nominal_value,2))
            + r'$\pm$' + str(round(params_arr_log[3].std_dev,4)) 
            + r'$\, [\log_{10}\left(\mathrm{m} \, s^{-2}\right)]$'])
        model_tab=np.asarray(['psi_single','psi_single','psi_multi','psi_multi'])
        model_tab_1=np.asarray(['Summed','Matched','Summed','Matched'])
        fit_dict_tab=np.asarray([
            constants.fitting_dict(sol_mfree=True),constants.fitting_dict(sol_mfree=True,sol_match=True),
            constants.fitting_dict(sol_mfree=True),constants.fitting_dict(sol_mfree=True,sol_match=True)
            ])
        title_tab=np.asarray(['Single, Summed','Single, Matched','Double, Summed','Double, Matched'])
        _,axs=plt.subplots(4,4,figsize=(50,50))
        for i in range(4):
            axs[0,i].scatter(gbar[model_tab[i]][model_tab_1[i]],gtot[model_tab[i]][model_tab_1[i]],label=r'Data points',c='k')
            axs[0,i].scatter(gbar[model_tab[i]][model_tab_1[i]],gMOND[model_tab[i]][model_tab_1[i]],
                label=r'MOND : $\log_{10} \, g^{\dagger} = $' 
                    + str(round(np.log10(1.2*1e-10),2))
                    + r'$\, [\log_{10}\left(\mathrm{m} \, s^{-2}\right)]$')
            fit_init=model_fit.grar_fit(model_tab[i],ULDM_fits=True,fit_dict_in=fit_dict_tab[i])
            axs[0,i].scatter(gbar[model_tab[i]][model_tab_1[i]],
                fit_init.grar_model(gbar[model_tab[i]][model_tab_1[i]],fit_results[model_tab[i]][model_tab_1[i]].params),
                label=label_tab[i])
            axs[0,i].axline([0,0],[10**(-8.5),10**(-8.5)],linestyle='-',c='k')
            axs[0,i].set_xlabel(r'$g_\mathrm{bar} \, [\mathrm{m}/\mathrm{s}^2]$')
            axs[0,i].set_ylabel(r'$g_\mathrm{tot} \, [\mathrm{m}/\mathrm{s}^2]$')
            axs[0,i].set_xscale('log')
            axs[0,i].set_yscale('log')
            axs[0,i].set_title(str(title_tab[i]))
            axs[0,i].grid(visible=True,axis='both',c='k',alpha=0.4)
            axs[0,i].legend()
        model_tab=['Summed','Matched','Summed','Matched']
        title_tab=np.asarray(['Single, Summed','Single, Matched','Double, Summed','Double, Matched'])
        model_tab_1=[psis_dict_in,psis_dict_in,psim_dict_in,psim_dict_in]
        plt_0=0
        for mod in model_tab:
            in0=unumpy.log10(model_tab_1[plt_0][mod]['Vbulge_none']['params']['c200'])
            in1=unumpy.log10(model_tab_1[plt_0][mod]['Vbulge']['params']['c200'])
            c200_in=np.concatenate((in0,in1))
            in0=np.log10(model_tab_1[plt_0][mod]['Vbulge_none']['Mvir'])
            in1=np.log10(model_tab_1[plt_0][mod]['Vbulge']['Mvir'])
            m200_in=np.concatenate((in0,in1))
            args_sort=np.argsort(m200_in)
            m200_in=m200_in[args_sort]
            c200CMR_Wa=np.asarray([])
            for i in range(len(m200_in)):
                c200CMR_Wa=np.append(c200CMR_Wa,model_fit.conc_mass_rel_Wa(10**(m200_in[i])))
            c200CMR_Wa_val=np.log10(c200CMR_Wa)
            axs[1,plt_0].plot(m200_in,c200CMR_Wa_val,label=r'CMR_W : $M_{200}$',c='k',linestyle='solid')
            c200CMR_Du=model_fit.conc_mass_rel_Du(10**(m200_in))
            c200CMR_Du_val=c200CMR_Du
            axs[1,plt_0].plot(m200_in,c200CMR_Du_val,label=r'CMR_D : $M_{200}$',c='k',linestyle='dashed')
            c200_vals=np.asarray([])
            c200_err=np.asarray([])
            m200_vals=np.asarray([])
            for i in range(len(c200_in)):
                if c200_in[i].std_dev!=np.inf:
                    c200_vals=np.append(c200_vals,c200_in[i].nominal_value)
                    c200_err=np.append(c200_err,c200_in[i].std_dev)
                    m200_vals=np.append(m200_vals,m200_in[i])
            axs[1,plt_0].errorbar(m200_vals,c200_vals,xerr=None,yerr=c200_err,c='k',fmt='o')
            c200_vals=np.asarray([])
            m200_vals=np.asarray([])
            for i in range(len(c200_in)):
                if c200_in[i].std_dev==np.inf:
                    c200_vals=np.append(c200_vals,c200_in[i].nominal_value)
                    m200_vals=np.append(m200_vals,m200_in[i])
            axs[1,plt_0].scatter(m200_vals,c200_vals,c='g',marker='x')
            if model_tab_1[plt_0]==psim_dict_in:
                in0=unumpy.log10(model_tab_1[plt_0][mod]['Vbulge_none']['params']['c200_2'])
                in1=unumpy.log10(model_tab_1[plt_0][mod]['Vbulge']['params']['c200_2'])
                c200_in=np.concatenate((in0,in1))
            c200_vals=np.asarray([])
            c200_err=np.asarray([])
            m200_vals=np.asarray([])
            for i in range(len(c200_in)):
                if c200_in[i].std_dev!=np.inf:
                    c200_vals=np.append(c200_vals,c200_in[i].nominal_value)
                    c200_err=np.append(c200_err,c200_in[i].std_dev)
                    m200_vals=np.append(m200_vals,m200_in[i])
            axs[1,plt_0].errorbar(m200_vals,c200_vals,xerr=None,yerr=c200_err,c='k',fmt='o')
            c200_vals=np.asarray([])
            m200_vals=np.asarray([])
            for i in range(len(c200_in)):
                if c200_in[i].std_dev==np.inf:
                    c200_vals=np.append(c200_vals,c200_in[i].nominal_value)
                    m200_vals=np.append(m200_vals,m200_in[i])
            axs[1,plt_0].scatter(m200_vals,c200_vals,c='g',marker='x')
            axs[1,plt_0].set_ylabel(r'$\log_{10}c_{200}$')
            axs[1,plt_0].set_xlabel(r'$\log_{10}\left(M_{200} \, [M_{\odot}]\right)$')
            axs[1,plt_0].set_title(title_tab[plt_0])
            axs[1,plt_0].legend()
            axs[1,plt_0].grid(visible=True,axis='both',c='k',alpha=0.4)
            axs[1,plt_0].set_xlim(8.5,15.5)
            axs[1,plt_0].set_ylim(-0.25,2.25)
            plt_0+=1
        plt_0=0
        for mod in model_tab:
            in0=np.log10(model_tab_1[plt_0][mod]['Vbulge_none']['params']['Vf'])
            in1=np.log10(model_tab_1[plt_0][mod]['Vbulge']['params']['Vf'])
            Vf_in=np.concatenate((in0,in1))
            axs[2,plt_0].plot(Vf_in,model_fit.BTFR(10**(Vf_in)),label='BTFR',c='k',linestyle='solid')
            mass_in=model_tab_1[plt_0][mod]['Vbulge_none']['params']['mstar']+model_tab_1[plt_0][mod]['Vbulge_none']['params']['mgas']
            in0=unumpy.log10((mass_in)*1e9)
            mass_in=model_tab_1[plt_0][mod]['Vbulge']['params']['mstar']+model_tab_1[plt_0][mod]['Vbulge']['params']['mgas']
            in1=unumpy.log10((mass_in)*1e9)
            mstar_in=np.concatenate((in0,in1))
            mstar_vals=np.asarray([])
            mstar_err=np.asarray([])
            Vf_vals=np.asarray([])
            for i in range(len(mstar_in)):
                if mstar_in[i].std_dev!=np.inf:
                    mstar_vals=np.append(mstar_vals,mstar_in[i].nominal_value)
                    mstar_err=np.append(mstar_err,mstar_in[i].std_dev)
                    Vf_vals=np.append(Vf_vals,Vf_in[i])
            axs[2,plt_0].errorbar(Vf_vals,mstar_vals,xerr=None,yerr=mstar_err,c='k',fmt='o')
            mstar_vals=np.asarray([])
            Vf_vals=np.asarray([])
            for i in range(len(mstar_in)):
                if mstar_in[i].std_dev==np.inf:
                    mstar_vals=np.append(mstar_vals,mstar_in[i].nominal_value)
                    Vf_vals=np.append(Vf_vals,Vf_in[i])
            axs[2,plt_0].scatter(Vf_vals,mstar_vals,c='g',marker='x')
            axs[2,plt_0].set_ylabel(r'$\log_{10} \left(M_b \, [M_\odot]\right)$')
            axs[2,plt_0].set_xlabel(r'$\log_{10} \left(V_f \, [km/s]\right)$')
            axs[2,plt_0].set_title(title_tab[plt_0])
            axs[2,plt_0].legend()
            axs[2,plt_0].grid(visible=True,axis='both',c='k',alpha=0.4)
            axs[2,plt_0].set_ylim(7.5,12.5)
            plt_0+=1
        plt_0=0
        for mod in model_tab:
            in0=model_tab_1[plt_0][mod]['Vbulge_none']['params']['mstar']*1e9
            in1=model_tab_1[plt_0][mod]['Vbulge']['params']['mstar']*1e9
            mstar_in=np.concatenate((in0,in1))
            mstar_in=unumpy.log10(mstar_in)
            in0=model_tab_1[plt_0][mod]['Vbulge_none']['Mvir']
            in1=model_tab_1[plt_0][mod]['Vbulge']['Mvir']
            m200_in=np.concatenate((in0,in1))
            args_sort=np.argsort(m200_in)
            m200_in=m200_in[args_sort]
            mstar_ex=np.log10(model_fit.abund_match_rel(m200_in))
            axs[3,plt_0].plot(np.log10(m200_in),mstar_ex,label='AMR',c='k',linestyle='solid')
            mstar_vals=np.asarray([])
            mstar_err=np.asarray([])
            m200_vals=np.asarray([])
            for i in range(len(mstar_in)):
                if mstar_in[i].std_dev!=np.inf:
                    mstar_vals=np.append(mstar_vals,mstar_in[i].nominal_value)
                    mstar_err=np.append(mstar_err,mstar_in[i].std_dev)
                    m200_vals=np.append(m200_vals,m200_in[i])
            axs[3,plt_0].errorbar(np.log10(m200_vals),mstar_vals,xerr=None,yerr=mstar_err,c='k',fmt='o')
            mstar_vals=np.asarray([])
            m200_vals=np.asarray([])
            for i in range(len(mstar_in)):
                if mstar_in[i].std_dev==np.inf:
                    mstar_vals=np.append(mstar_vals,mstar_in[i].nominal_value)
                    m200_vals=np.append(m200_vals,m200_in[i])
            axs[3,plt_0].scatter(np.log10(m200_vals),mstar_vals,c='g',marker='x')
            axs[3,plt_0].set_ylabel(r'$\log_{10}\left(M_* \, [M_{\odot}]\right)$')
            axs[3,plt_0].set_xlabel(r'$\log_{10}\left(M_{200} \, [M_{\odot}]\right)$')
            axs[3,plt_0].set_title(title_tab[plt_0])
            axs[3,plt_0].legend()
            axs[3,plt_0].grid(visible=True,axis='both',c='k',alpha=0.4)
            axs[3,plt_0].set_xlim(8.5,15.5)
            axs[3,plt_0].set_ylim(4.5,12.5)
            plt_0+=1
        if save_file!=None:
            plt.savefig(save_file)
        return

    ## Define the plot of the reduced chi-square for the ULDM (particle mass fixed and scanned) and Einasto models.
    #  This defines the plot of the reduced chi-square comparisons vs. particle mass between 
    #  each of the ULDM (particle mass fixed and scanned) models and the Einasto model.
    #  @param self
    #  object pointer
    #  @param psis_dict_in
    #  dictionary \n
    #  The dictionary for the single flavor models.
    #  See fits_Einasto_all.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be of the form: \n
    #  psis_dict_in = { 
    #  - 'Summed' : {...}
    #  - 'Matched' : {...}
    #  },
    #  where each {...} must be an instance of results.results_psi_single_all.fit.
    #  @param psim_dict_in
    #  dictionary \n
    #  The dictionary for the double flavor models.
    #  See fits_Einasto_all.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be of the form: \n
    #  psim_dict_in = { 
    #  - 'Summed' : {...}
    #  - 'Matched' : {...}
    #  },
    #  where each {...} must be an instance of results.results_psi_multi_all.fit.
    #  @param CDM_dict_in
    #  dictionary \n
    #  The dictionary for the Einasto.
    #  See fits_Einasto_all.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be an instance of results.results_CDM_all.fit.
    #  @param lwin (optional)
    #  int \n
    #  Width of lines for plot.
    #  @param fit_dict_in (optional)
    #  constants.fitting_dict instance \n
    #  Instance of the constants.fitting_dict class.
    #  Contains the mass ranges to be used.
    #  @param save_file (optional)
    #  str \n
    #  File path to save plot.  Default is None, in which case file is not saved.
    #  @returns
    #  matplotlib.pyplot.figure[2,2] \n
    #  Figure of the reduced chi-square comparisons vs. particle mass between 
    #  each of the ULDM (particle mass fixed and scanned) models and the Einasto model. 
    def chi_psi_mfix(self,psis_dict_in,psim_dict_in,CDM_dict_in,lwin=4,fit_dict_in=fitting_dict_in,save_file=None):
        _,axs=plt.subplots(2,2,figsize=(35,25))
        mtab=fit_dict_in.sol_m22_tab
        mtab=mtab
        mtab_prime=fit_dict_in.sol_m22_tab_prime
        model_tab=['psis','psim']
        model_tab_1=['Summed','Matched']
        label_tab=np.asarray([['Single, Summed', 'Single, Matched'],['Double, Summed', 'Double, Matched']])
        chisq_sum={}
        for mod in model_tab:
            chisq_sum[mod]={}
            for mod1 in model_tab_1:
                chisq_sum[mod][mod1]=np.asarray([])
        chisq_CDM=np.concatenate((CDM_dict_in['Vbulge_none']['Chi_sq'],CDM_dict_in['Vbulge']['Chi_sq']))
        chisq_sum_CDM=np.nansum(chisq_CDM)
        for k in range(len(mtab_prime)):
            psi_chisq=np.concatenate((psis_dict_in['Summed']['Vbulge_none']['m='+str(np.log10(mtab_prime[k]))]['Chi_sq'],
                                    psis_dict_in['Summed']['Vbulge']['m='+str(np.log10(mtab_prime[k]))]['Chi_sq']))
            chisq_sum['psis']['Summed']=np.append(chisq_sum['psis']['Summed'],np.nansum(psi_chisq))

            psi_chisq=np.concatenate((psim_dict_in['Summed']['Vbulge_none']['m='+str(np.log10(mtab_prime[k]))]['Chi_sq'],
                                    psim_dict_in['Summed']['Vbulge']['m='+str(np.log10(mtab_prime[k]))]['Chi_sq']))
            chisq_sum['psim']['Summed']=np.append(chisq_sum['psim']['Summed'],np.nansum(psi_chisq))
        for k in range(len(mtab)):
            psi_chisq=np.concatenate((psis_dict_in['Matched']['Vbulge_none']['m='+str(np.log10(mtab[k]))]['Chi_sq'],
                                    psis_dict_in['Matched']['Vbulge']['m='+str(np.log10(mtab[k]))]['Chi_sq']))
            chisq_sum['psis']['Matched']=np.append(chisq_sum['psis']['Matched'],np.nansum(psi_chisq))

            psi_chisq=np.concatenate((psim_dict_in['Matched']['Vbulge_none']['m='+str(np.log10(mtab[k]))]['Chi_sq'],
                                    psim_dict_in['Matched']['Vbulge']['m='+str(np.log10(mtab[k]))]['Chi_sq']))
            chisq_sum['psim']['Matched']=np.append(chisq_sum['psim']['Matched'],np.nansum(psi_chisq))
        for i in range(2):
            for j in range(2):
                diff_in=(chisq_sum_CDM-chisq_sum[model_tab[i]][model_tab_1[j]])/chisq_sum_CDM
                if model_tab_1[j]=='Summed':
                    axs[i,j].plot(mtab_prime,diff_in,c='k',linewidth=lwin)
                else:
                    axs[i,j].plot(mtab,diff_in,c='k',linewidth=lwin)
                axs[i,j].set_xscale('log')
                axs[i,j].set_xlabel(r'$m \,[10^{-22}\,\mathrm{eV}]$')
                axs[i,j].set_title(label_tab[i,j])
                axs[i,j].axhline(0,linestyle='--',c='k',linewidth=lwin)
                axs[i,j].set_ylabel(r'$1 - \sum \chi^2_{\nu,\mathrm{ULDM}} / \sum \chi^2_{\nu,\mathrm{Einasto}}$')
                axs[i,j].tick_params(axis='both', which='major', labelsize=20, width=2.5, length=10)
                axs[i,j].tick_params(axis='both', which='minor', labelsize=15, width=2, length=5)
        axs[1,1].set_xlim(3,110)
        axs[1,1].set_ylim(-12,2)
        axs[0,0].grid(visible=True,axis='both',c='k',alpha=0.4)
        axs[1,0].grid(visible=True,axis='both',c='k',alpha=0.4)
        axs[0,1].grid(visible=True,axis='both',which='both',c='k',alpha=0.4)
        axs[1,1].grid(visible=True,axis='both',which='both',c='k',alpha=0.4)
        if save_file!=None:          
            plt.savefig(save_file)
        return

    ## Define the plot of the cumulative reduced chi-square for single flavor ULDM (particle mass fixed and scanned) 
    #  and Einasto models.
    #  This defines the plot of the cumulative reduced chi-square comparisons vs. galaxy between 
    #  each of the single flavor ULDM (particle mass fixed and scanned) models and the Einasto model.
    #  @param self
    #  object pointer
    #  @param psis_dict_in
    #  dictionary \n
    #  The dictionary for the single flavor models.
    #  See fits_Einasto_all.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be of the form: \n
    #  psis_dict_in = { 
    #  - 'Summed' : {...}
    #  - 'Matched' : {...}
    #  },
    #  where each {...} must be an instance of results.results_psi_single_all.fit.
    #  @param psim_dict_in
    #  dictionary \n
    #  The dictionary for the double flavor models.
    #  See fits_Einasto_all.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be of the form: \n
    #  psim_dict_in = { 
    #  - 'Summed' : {...}
    #  - 'Matched' : {...}
    #  },
    #  where each {...} must be an instance of results.results_psi_multi_all.fit.
    #  @param CDM_dict_in
    #  dictionary \n
    #  The dictionary for the Einasto.
    #  See fits_Einasto_all.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be an instance of results.results_CDM_all.fit.
    #  @param lwin (optional)
    #  int \n
    #  Width of lines for plot.
    #  @param fit_dict_in (optional)
    #  constants.fitting_dict instance \n
    #  Instance of the constants.fitting_dict class.
    #  Contains the mass ranges to be used.
    #  @param save_file (optional)
    #  str \n
    #  File path to save plot.  Default is None, in which case file is not saved.
    #  @returns
    #  matplotlib.pyplot.figure[2,1] \n
    #  Figure of the cumulative reduced chi-square vs. galaxy for 
    #  each of the single flavor ULDM (particle mass fixed and scanned) models and the Einasto model. 
    def chi_gal_psi_mfix(self,psis_dict_in,CDM_dict_in,lwin=4,fit_dict_in=fitting_dict_in,save_file=None):
        _,axs=plt.subplots(2,1,figsize=(45,35))
        mtab_prime=fit_dict_in.sol_m22_tab_prime
        name_tab=np.concatenate((psis_dict_in['Summed']['Vbulge_none']['m='+str(np.log10(mtab_prime[0]))]['Name'],
            psis_dict_in['Summed']['Vbulge']['m='+str(np.log10(mtab_prime[0]))]['Name']))
        chisq_sum={}
        ind_0=2
        ind_1=9
        chisq_sum['m='+str(np.log10(mtab_prime[ind_0]))]=np.asarray([])
        chisq_sum['m='+str(np.log10(mtab_prime[ind_1]))]=np.asarray([])
        chisq_CDM=np.concatenate((CDM_dict_in['Vbulge_none']['Chi_sq'],CDM_dict_in['Vbulge']['Chi_sq']))
        chisq_sum_CDM=np.asarray([])
        for i in range(len(chisq_CDM)):
            chisq_sum_CDM=np.append(chisq_sum_CDM,np.nansum(chisq_CDM[0:i+1]))
        psi_chisq=np.concatenate((psis_dict_in['Summed']['Vbulge_none']['m='+str(np.log10(mtab_prime[ind_0]))]['Chi_sq'],
                                psis_dict_in['Summed']['Vbulge']['m='+str(np.log10(mtab_prime[ind_0]))]['Chi_sq']))
        for i in range(len(psi_chisq)):
            chisq_sum['m='+str(np.log10(mtab_prime[ind_0]))]=np.append(chisq_sum['m='+str(np.log10(mtab_prime[ind_0]))],
                np.nansum(psi_chisq[0:i+1]))
        psi_chisq=np.concatenate((psis_dict_in['Summed']['Vbulge_none']['m='+str(np.log10(mtab_prime[ind_1]))]['Chi_sq'],
                                psis_dict_in['Summed']['Vbulge']['m='+str(np.log10(mtab_prime[ind_1]))]['Chi_sq']))
        for i in range(len(psi_chisq)):
            chisq_sum['m='+str(np.log10(mtab_prime[ind_1]))]=np.append(chisq_sum['m='+str(np.log10(mtab_prime[ind_1]))],
                np.nansum(psi_chisq[0:i+1]))
        gal_ind=np.arange(0,len(chisq_sum_CDM),1)
        axs[0].plot(gal_ind,chisq_sum['m='+str(np.log10(mtab_prime[ind_0]))],c='k',
            linewidth=lwin,label=r'$\log_{10}m = $'+str(np.log10(mtab_prime[ind_0]))+r'$\,m_{22}$')
        axs[0].plot(gal_ind,chisq_sum['m='+str(np.log10(mtab_prime[ind_1]))],c='b',
            linewidth=lwin,label=r'$\log_{10}m = $'+str(np.log10(mtab_prime[ind_1]))+r'$\,m_{22}$')
        axs[0].plot(gal_ind,chisq_sum_CDM,c='r',linewidth=lwin,label=r'Einasto')
        axs[0].set_yscale('log')
        axs[0].set_ylabel(r'$\sum \chi^2$')
        axs[0].legend()
        axs[0].set_xticks(gal_ind)
        axs[0].set_xticklabels(name_tab,rotation=90);
        axs[0].set_title(r'Single, Summed')
        mtab=fit_dict_in.sol_m22_tab
        chisq_sum={}
        ind_0=0
        ind_1=10
        chisq_sum['m='+str(np.log10(mtab[ind_0]))]=np.asarray([])
        chisq_sum['m='+str(np.log10(mtab[ind_1]))]=np.asarray([])
        psi_chisq=np.concatenate((psis_dict_in['Matched']['Vbulge_none']['m='+str(np.log10(mtab[ind_0]))]['Chi_sq'],
                                psis_dict_in['Matched']['Vbulge']['m='+str(np.log10(mtab[ind_0]))]['Chi_sq']))
        for i in range(len(psi_chisq)):
            chisq_sum['m='+str(np.log10(mtab[ind_0]))]=np.append(chisq_sum['m='+str(np.log10(mtab[ind_0]))],np.nansum(psi_chisq[0:i+1]))
        psi_chisq=np.concatenate((psis_dict_in['Matched']['Vbulge_none']['m='+str(np.log10(mtab[ind_1]))]['Chi_sq'],
                                psis_dict_in['Matched']['Vbulge']['m='+str(np.log10(mtab[ind_1]))]['Chi_sq']))
        for i in range(len(psi_chisq)):
            chisq_sum['m='+str(np.log10(mtab[ind_1]))]=np.append(chisq_sum['m='+str(np.log10(mtab[ind_1]))],np.nansum(psi_chisq[0:i+1]))
        gal_ind=np.arange(0,len(chisq_sum_CDM),1)
        axs[1].plot(gal_ind,chisq_sum['m='+str(np.log10(mtab[ind_0]))],c='k',linewidth=lwin,
            label=r'$\log_{10}m = $'+str(round(np.log10(mtab[ind_0]),2))+r'$\,m_{22}$')
        axs[1].plot(gal_ind,chisq_sum['m='+str(np.log10(mtab[ind_1]))],c='b',linewidth=lwin,
            label=r'$\log_{10}m = $'+str(round(np.log10(mtab[ind_1]),2))+r'$\,m_{22}$')
        axs[1].plot(gal_ind,chisq_sum_CDM,c='r',linewidth=lwin,label=r'Einasto')
        axs[1].set_yscale('log')
        axs[1].set_ylabel(r'$\sum \chi^2$')
        axs[1].legend()
        axs[1].set_xticks(gal_ind)
        axs[1].set_xticklabels(name_tab,rotation=90);
        axs[1].set_title(r'Single, Matched')
        for i in range(2):
            axs[i].grid(visible=True,axis='y',which='both',c='k',alpha=0.4)
            axs[i].grid(visible=True,axis='x',which='major',c='k',alpha=1)
            axs[i].tick_params(axis='y', which='major', labelsize=20, width=2.5, length=10)
            axs[i].tick_params(axis='y', which='minor', labelsize=15, width=2, length=5)
        if save_file!=None:
            plt.savefig(save_file)
        return

    ## Define the plot of the soliton-halo (SH) relation for the ULDM (particle mass fixed and scanned) models.
    #  This defines the plot of the SH relation for each of the ULDM (particle mass fixed and scanned) models.
    #  @param self
    #  object pointer
    #  @param psis_dict_in
    #  dictionary \n
    #  The dictionary for the single flavor models.
    #  See fits_Einasto_all.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be of the form: \n
    #  psis_dict_in = { 
    #  - 'Summed' : {...}
    #  - 'Matched' : {...}
    #  },
    #  where each {...} must be an instance of results.results_psi_single_all.fit.
    #  @param psim_dict_in
    #  dictionary \n
    #  The dictionary for the double flavor models.
    #  See fits_Einasto_all.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be of the form: \n
    #  psim_dict_in = { 
    #  - 'Summed' : {...}
    #  - 'Matched' : {...}
    #  },
    #  where each {...} must be an instance of results.results_psi_multi_all.fit.
    #  @param bins (optional)
    #  int \n
    #  Number of bins for histogram plot.
    #  @param fit_dict_in (optional)
    #  constants.fitting_dict instance \n
    #  Instance of the constants.fitting_dict class.
    #  Contains the mass ranges to be used.
    #  @param save_file (optional)
    #  str \n
    #  File path to save plot.  Default is None, in which case file is not saved.
    #  @returns
    #  matplotlib.pyplot.figure[3,2] \n
    #  Figure of the SH relation for each of the ULDM (particle mass fixed and scanned) models.
    def Msol_psi_mfix(self,psis_dict_in,psim_dict_in,bins=50,fit_dict_in=fitting_dict_in,save_file=None):
        _,axs=plt.subplots(3,2,figsize=(25,35))
        mtab=fit_dict_in.sol_m22_tab
        mtab_prime=fit_dict_in.sol_m22_tab_prime
        bins=50
        plt_0=0
        for key in psis_dict_in:
            if key=='Matched':
                mtab_in=mtab
            else:
                mtab_in=mtab_prime
            x_tab_mean=np.asarray([])
            x_tab_med=np.asarray([])
            for i in range(len(mtab_in)):
                x_tab=np.concatenate((psis_dict_in[key]['Vbulge_none']['m='+str(np.log10(mtab_in[i]))]['params']['Msol'],
                                        psis_dict_in[key]['Vbulge']['m='+str(np.log10(mtab_in[i]))]['params']['Msol']))
                mtab_ex=np.repeat(mtab_in[i],len(x_tab))
                y_tab=np.concatenate((psis_dict_in[key]['Vbulge_none']['m='+str(np.log10(mtab_in[i]))]['Mhalo'],
                                        psis_dict_in[key]['Vbulge']['m='+str(np.log10(mtab_in[i]))]['Mhalo']))
                y_tab=1.4e9*mtab_ex**(-1)*(y_tab/1e12)**(1/3)
                x_tab=x_tab/y_tab
                x_tab_vals=np.asarray([])
                for j in range(len(x_tab)):
                    x_tab_vals=np.append(x_tab_vals,np.log10(x_tab[j].nominal_value))
                x_tab_mean=np.append(x_tab_mean,np.nanmean(x_tab_vals))
                x_tab_med=np.append(x_tab_med,np.nanmedian(x_tab_vals))
            axs[0,plt_0].plot(np.log10(mtab_in),x_tab_mean,c='k',marker='o',label=r'Mean')
            axs[0,plt_0].plot(np.log10(mtab_in),x_tab_med,c='k',marker='x',label=r'Median')
            axs[0,plt_0].set_ylabel(r'$\log_{10}\left(M_\mathrm{sol}/M_{\mathrm{SH}}\right)$')
            axs[0,plt_0].set_xlabel(r'$\log_{10} \left(m \, [10^{-22}\,\mathrm{eV}]\right)$')
            axs[0,plt_0].set_title('Single, ' + str(key))
            axs[0,plt_0].axhline(np.log10(1),linestyle='--',c='k')
            axs[0,plt_0].axhline(np.log10(2),c='b')
            axs[0,plt_0].axhline(np.log10(0.5),c='b')
            axs[0,plt_0].tick_params(axis='both', which='major', labelsize=20, width=2.5, length=10)
            axs[0,plt_0].tick_params(axis='both', which='minor', labelsize=15, width=2, length=5)
            axs[0,plt_0].set_ylim(-2.5,2.5)
            axs[0,plt_0].legend()
            plt_0+=1
        plt_0=0
        for key in psim_dict_in:
            if key=='Matched':
                mtab_in=mtab
            else:
                mtab_in=mtab_prime
            x_tab_vals_all=np.asarray([])
            for i in range(len(mtab_in)):
                x_tab=np.concatenate((psim_dict_in[key]['Vbulge_none']['m='+str(np.log10(mtab_in[i]))]['params']['Msol'],
                                        psim_dict_in[key]['Vbulge']['m='+str(np.log10(mtab_in[i]))]['params']['Msol']))
                mtab_ex=np.repeat(np.float(fit_dict_in.sol_m22),len(x_tab))
                y_tab=np.concatenate((psim_dict_in[key]['Vbulge_none']['m='+str(np.log10(mtab_in[i]))]['Mhalo_1'],
                                        psim_dict_in[key]['Vbulge']['m='+str(np.log10(mtab_in[i]))]['Mhalo_1']))
                y_tab=1.4e9*mtab_ex**(-1)*(y_tab/1e12)**(1/3)
                x_tab=x_tab/y_tab
                x_tab_vals=np.asarray([])
                for i in range(len(x_tab)):
                    x_tab_vals=np.append(x_tab_vals,np.log10(x_tab[i].nominal_value))
                x_tab_vals_all=np.append(x_tab_vals_all,x_tab_vals)
            axs[1,plt_0].hist(x_tab_vals_all,color='k',bins=bins)
            axs[1,plt_0].set_ylabel(r'Number of galaxies')
            axs[1,plt_0].set_xlabel(r'$\log_{10}\left(M_{\mathrm{sol},1}/M_{\mathrm{SH},1}\right)$')
            axs[1,plt_0].set_title('Double, ' + str(key))
            axs[1,plt_0].axvline(np.log10(1),linestyle='--',c='k')
            axs[1,plt_0].axvline(np.log10(2),c='b')
            axs[1,plt_0].axvline(np.log10(0.5),c='b')
            axs[1,plt_0].tick_params(axis='both', which='major', labelsize=20, width=2.5, length=10)
            axs[1,plt_0].set_xlim(-2.5,2.5)
            plt_0+=1
        plt_0=0
        for key in psim_dict_in:
            if key=='Matched':
                mtab_in=mtab
            else:
                mtab_in=mtab_prime
            x_tab_mean=np.asarray([])
            x_tab_med=np.asarray([])
            for i in range(len(mtab_in)):
                x_tab=np.concatenate((psim_dict_in[key]['Vbulge_none']['m='+str(np.log10(mtab_in[i]))]['params']['Msol_2'],
                                        psim_dict_in[key]['Vbulge']['m='+str(np.log10(mtab_in[i]))]['params']['Msol_2']))
                mtab_ex=np.repeat(mtab_in[i],len(x_tab))
                y_tab=np.concatenate((psim_dict_in[key]['Vbulge_none']['m='+str(np.log10(mtab_in[i]))]['Mhalo_2'],
                                        psim_dict_in[key]['Vbulge']['m='+str(np.log10(mtab_in[i]))]['Mhalo_2']))
                y_tab=1.4e9*mtab_ex**(-1)*(y_tab/1e12)**(1/3)
                x_tab=x_tab/y_tab
                x_tab_vals=np.asarray([])
                for i in range(len(x_tab)):
                    x_tab_vals=np.append(x_tab_vals,np.log10(x_tab[i].nominal_value))
                x_tab_mean=np.append(x_tab_mean,np.nanmean(x_tab_vals))
                x_tab_med=np.append(x_tab_med,np.nanmedian(x_tab_vals))
            axs[2,plt_0].plot(np.log10(mtab_in),x_tab_mean,c='k',marker='o',label=r'Mean')
            axs[2,plt_0].plot(np.log10(mtab_in),x_tab_med,c='k',marker='x',label=r'Median')
            axs[2,plt_0].set_ylabel(r'$\log_{10}\left(M_{\mathrm{sol},2}/M_{\mathrm{SH},2}\right)$')
            axs[2,plt_0].set_xlabel(r'$\log_{10} \left(m_2 \, [10^{-22}\,\mathrm{eV}]\right)$')
            axs[2,plt_0].set_title('Double, ' + str(key))
            axs[2,plt_0].axhline(np.log10(1),linestyle='--',c='k')
            axs[2,plt_0].axhline(np.log10(2),c='b')
            axs[2,plt_0].axhline(np.log10(0.5),c='b')
            axs[2,plt_0].tick_params(axis='both', which='major', labelsize=20, width=2.5, length=10)
            axs[2,plt_0].tick_params(axis='both', which='minor', labelsize=15, width=2, length=5)
            axs[2,plt_0].set_ylim(-2.5,2.5)
            axs[2,plt_0].legend()
            plt_0+=1
        axs[0,0].grid(visible=True,axis='both',c='k',alpha=0.4)
        axs[2,0].grid(visible=True,axis='both',c='k',alpha=0.4)
        axs[0,1].grid(visible=True,axis='y',which='major',c='k',alpha=0.4)
        axs[0,1].grid(visible=True,axis='x',which='both',c='k',alpha=0.4)
        axs[2,1].grid(visible=True,axis='y',which='major',c='k',alpha=0.4)
        axs[2,1].grid(visible=True,axis='x',which='both',c='k',alpha=0.4)
        if save_file!=None:
            plt.savefig(save_file)
        return

    ## Define the plot of the BIC differences distributions between the ULDM (particle mass fixed) and Einasto models.
    #  This defines the plot of the distributions of the differences in BIC between
    #  each of the ULDM (particle mass fixed) models and the Einasto model.
    #  @param self
    #  object pointer
    #  @param psis_dict_in
    #  dictionary \n
    #  The dictionary for the single flavor models.
    #  See fits_Einasto_all.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be of the form: \n
    #  psis_dict_in = { 
    #  - 'Summed' : {...}
    #  - 'Matched' : {...}
    #  },
    #  where each {...} must be an instance of results.results_psi_single_all.fit.
    #  @param psim_dict_in
    #  dictionary \n
    #  The dictionary for the double flavor models.
    #  See fits_Einasto_all.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be of the form: \n
    #  psim_dict_in = { 
    #  - 'Summed' : {...}
    #  - 'Matched' : {...}
    #  },
    #  where each {...} must be an instance of results.results_psi_multi_all.fit.
    #  @param CDM_dict_in
    #  dictionary \n
    #  The dictionary for the Einasto.
    #  See fits_Einasto_all.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be an instance of results.results_CDM_all.fit.
    #  @param m22_ex (optional)
    #  float \n
    #  Value for particle mass (single flavor models) or particle mass one (double flavor models).
    #  @param m22_2_ex (optional)
    #  float \n
    #  Value for particle mass two (double flavor models).
    #  @param lwin (optional)
    #  int \n
    #  Width of lines for plot.
    #  @param bins (optional)
    #  int \n
    #  Number of bins for histogram plot.
    #  @param save_file (optional)
    #  str \n
    #  File path to save plot.  Default is None, in which case file is not saved.
    #  @returns
    #  matplotlib.pyplot.figure[2,2] \n
    #  Figure of the BIC differences distributions between each ULDM (particle mass fixed) model and the Einasto model.
    def BIC_psi_mfix_ex(self,psis_dict_in,psim_dict_in,CDM_dict_in,m22_ex=10**(1.5),m22_2_ex=10**(1.8),bins=50,lwin=3,save_file=None):
        _,axs=plt.subplots(2,2,figsize=(25,25))
        interv_tab=np.asarray([[-np.inf,-10],[-10,-6],[-6,-2],[-2,2],[2,6],[6,10],[10,np.inf]])
        CDM_in=np.concatenate((CDM_dict_in['Vbulge_none']['BIC'],CDM_dict_in['Vbulge']['BIC']))
        plt_0=0
        for key in psis_dict_in:
            psi_in=np.concatenate((psis_dict_in[key]['Vbulge_none']['m='+str(np.log10(m22_ex))]['BIC'],
                psis_dict_in[key]['Vbulge']['m='+str(np.log10(m22_ex))]['BIC']))
            diff_in=CDM_in-psi_in
            axs[0,plt_0].hist(diff_in,bins=bins,color='black')
            axs[0,plt_0].set_xlabel(r'$\mathrm{BIC}_{\mathrm{Einasto}} - \mathrm{BIC}_{\mathrm{ULDM}}$')
            axs[0,plt_0].set_title(r'Single, ' + str(key))
            textstr_in={}
            for i in range(len(interv_tab)):
                sum_in=0
                for j in range(len(diff_in)):
                    if interv_tab[i,0]<diff_in[j]<=interv_tab[i,1]:
                        sum_in+=1
                if (interv_tab[i,0]!=-np.inf and interv_tab[i,1]!=np.inf):
                    textstr_in[str(i)]=str(interv_tab[i,0])+r'$<\Delta\mathrm{BIC}\leq$'+str(interv_tab[i,1])+r': '+str(round(sum_in/len(diff_in),2))
                elif interv_tab[i,0]==-np.inf:
                    textstr_in[str(i)]=r'$\Delta\mathrm{BIC}\leq$'+str(interv_tab[i,1])+r': '+str(round(sum_in/len(diff_in),2))
                else:
                    textstr_in[str(i)]=str(interv_tab[i,0])+r'$<\Delta\mathrm{BIC}$'+r': '+str(round(sum_in/len(diff_in),2))
            textstr=textstr_in['0']
            for i in range(len(interv_tab)-1):
                textstr = '\n'.join((
                    textstr,
                    textstr_in[str(i+1)],
                    ))
            axs[0,plt_0].text(0.05,0.95,textstr,transform=axs[0,plt_0].transAxes,fontsize=22,verticalalignment='top',
                bbox=dict(facecolor='yellow',alpha=0.8))
            plt_0+=1
        bins=250
        plt_0=0
        for key in psim_dict_in:
            psi_in=np.concatenate((psim_dict_in[key]['Vbulge_none']['m='+str(np.log10(m22_2_ex))]['BIC'],
                psim_dict_in[key]['Vbulge']['m='+str(np.log10(m22_2_ex))]['BIC']))
            diff_in=CDM_in-psi_in
            axs[1,plt_0].hist(diff_in,bins=bins,color='black')
            axs[1,plt_0].set_xlabel(r'$\mathrm{BIC}_{\mathrm{Einasto}} - \mathrm{BIC}_{\mathrm{ULDM}}$')
            axs[1,plt_0].set_title(r'Double, ' + str(key))
            textstr_in={}
            for i in range(len(interv_tab)):
                sum_in=0
                for j in range(len(diff_in)):
                    if interv_tab[i,0]<diff_in[j]<=interv_tab[i,1]:
                        sum_in+=1
                if (interv_tab[i,0]!=-np.inf and interv_tab[i,1]!=np.inf):
                    textstr_in[str(i)]=str(interv_tab[i,0])+r'$<\Delta\mathrm{BIC}\leq$'+str(interv_tab[i,1])+r': '+str(round(sum_in/len(diff_in),2))
                elif interv_tab[i,0]==-np.inf:
                    textstr_in[str(i)]=r'$\Delta\mathrm{BIC}\leq$'+str(interv_tab[i,1])+r': '+str(round(sum_in/len(diff_in),2))
                else:
                    textstr_in[str(i)]=str(interv_tab[i,0])+r'$<\Delta\mathrm{BIC}$'+r': '+str(round(sum_in/len(diff_in),2))
            textstr=textstr_in['0']
            for i in range(len(interv_tab)-1):
                textstr = '\n'.join((
                    textstr,
                    textstr_in[str(i+1)],
                    ))
            axs[1,plt_0].text(0.05,0.95,textstr,transform=axs[1,plt_0].transAxes,fontsize=22,verticalalignment='top',
                bbox=dict(facecolor='yellow',alpha=0.8))
            plt_0+=1
        for i in range(2):
            for j in range(2):
                axs[i,j].axvline(0,linestyle='--',c='k',linewidth=lwin)
                axs[i,j].axvline(2,c='b',linewidth=lwin)
                axs[i,j].axvline(-2,c='b',linewidth=lwin)
                axs[i,j].axvline(6,c='r',linewidth=lwin)
                axs[i,j].axvline(-6,c='r',linewidth=lwin)
                axs[i,j].axvline(10,c='g',linewidth=lwin)
                axs[i,j].axvline(-10,c='g',linewidth=lwin)
                axs[i,j].set_ylabel('Number of galaxies')
                axs[i,j].set_xlim(-100,100)
                axs[i,j].tick_params(axis='both', which='major', labelsize=20, width=2.5, length=10)
                axs[i,j].tick_params(axis='both', which='minor', labelsize=15, width=2, length=5)
        if save_file!=None:
            plt.savefig(save_file)
        return

    ## Define the plot of the reduced chi-square for the ULDM (particle mass fixed) and Einasto models.
    #  This defines the plot of the reduced chi-square comparisons between
    #  each of the ULDM (particle mass fixed) models and the Einasto model.
    #  @param self
    #  object pointer
    #  @param psis_dict_in
    #  dictionary \n
    #  The dictionary for the single flavor models.
    #  See fits_Einasto_all.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be of the form: \n
    #  psis_dict_in = { 
    #  - 'Summed' : {...}
    #  - 'Matched' : {...}
    #  },
    #  where each {...} must be an instance of results.results_psi_single_all.fit.
    #  @param psim_dict_in
    #  dictionary \n
    #  The dictionary for the double flavor models.
    #  See fits_Einasto_all.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be of the form: \n
    #  psim_dict_in = { 
    #  - 'Summed' : {...}
    #  - 'Matched' : {...}
    #  },
    #  where each {...} must be an instance of results.results_psi_multi_all.fit.
    #  @param CDM_dict_in
    #  dictionary \n
    #  The dictionary for the Einasto.
    #  See fits_Einasto_all.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be an instance of results.results_CDM_all.fit.
    #  @param m22_ex (optional)
    #  float \n
    #  Value for particle mass (single flavor models) or particle mass one (double flavor models).
    #  @param m22_2_ex (optional)
    #  float \n
    #  Value for particle mass two (double flavor models).
    #  @param splt (optional)
    #  int \n
    #  Point size for plot.
    #  @param lwin (optional)
    #  int \n
    #  Width of lines for plot.
    #  @param save_file (optional)
    #  str \n
    #  File path to save plot.  Default is None, in which case file is not saved.
    #  @returns
    #  matplotlib.pyplot.figure[2,2] \n
    #  Figure of the reduced chi-square comparisons between each ULDM (particle mass fixed) model and the Einasto model.
    def chi_psi_mfix_ex(self,psis_dict_in,psim_dict_in,CDM_dict_in,m22_ex=10**(1.5),m22_2_ex=10**(1.8),splt=50,lwin=3,save_file=None):
        _,axs=plt.subplots(2,2,figsize=(30,25))
        model_tab=['psis','psim']
        chisq_in={'CDM':{},'psis':{},'psim':{}}
        chisq_in['CDM']=np.concatenate((CDM_dict_in['Vbulge_none']['Chi_sq'],CDM_dict_in['Vbulge']['Chi_sq']))
        for key in psis_dict_in:
            chisq_in['psis'][key]=np.concatenate((psis_dict_in[key]['Vbulge_none']['m='+str(np.log10(m22_ex))]['Chi_sq'],
                psis_dict_in[key]['Vbulge']['m='+str(np.log10(m22_ex))]['Chi_sq']))
            chisq_in['psim'][key]=np.concatenate((psim_dict_in[key]['Vbulge_none']['m='+str(np.log10(m22_2_ex))]['Chi_sq'],
                psim_dict_in[key]['Vbulge']['m='+str(np.log10(m22_2_ex))]['Chi_sq']))
        plt_0=0
        plt_1=0
        lab_ind=0
        for i in range(len(model_tab)):
            for key in chisq_in['psis']:
                mask=np.isfinite(chisq_in[model_tab[i]][key])
                mod_new=chisq_in[model_tab[i]][key][mask]
                mod_new_1=chisq_in['CDM'][mask]
                mask=np.isfinite(mod_new_1)
                mod_new=mod_new[mask]
                mod_new_1=mod_new_1[mask]
                xy = np.vstack([mod_new,mod_new_1])
                z = gaussian_kde(xy)(xy)
                axs[plt_1,plt_0].scatter(mod_new,mod_new_1,c=z,s=splt,cmap='Greys')
                axs[plt_1,plt_0].set_ylabel(r'$\chi^2_{\nu,\mathrm{Einasto}}$')
                axs[plt_1,plt_0].set_xlabel(r'$\chi^2_{\nu,\mathrm{ULDM}}$')
                if plt_0<1:
                    plt_0+=1
                else:
                    plt_0=0
                    plt_1+=1
                lab_ind+=1
        for i in range(2):
            for j in range(2):
                axs[i,j].axline([0,0],[1,1],linestyle='--',c='k',linewidth=lwin)
                axs[i,j].axhline(1,c='b',linewidth=lwin)
                axs[i,j].axvline(1,c='b',linewidth=lwin)
                axs[i,j].set_xscale('log')
                axs[i,j].set_yscale('log')
                axs[i,j].grid(visible=True,axis='both',c='k',alpha=0.4)
                axs[i,j].tick_params(axis='both', which='major', labelsize=20, width=2.5, length=10)
                axs[i,j].tick_params(axis='both', which='minor', labelsize=15, width=2, length=5)
        axs[0,0].set_title(r'Single, Summed')
        axs[0,1].set_title(r'Single, Matched')
        axs[1,0].set_title(r'Double, Summed')
        axs[1,1].set_title(r'Double, Matched');
        if save_file!=None:
            plt.savefig(save_file)
        return

    ## Define the plot of the soliton-halo (SH) for the ULDM (particle mass fixed) and Einasto models.
    #  This defines the plot of the SH relation for each of the ULDM (particle mass fixed) models and the Einasto model.
    #  @param self
    #  object pointer
    #  @param psis_dict_in
    #  dictionary \n
    #  The dictionary for the single flavor models.
    #  See fits_Einasto_all.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be of the form: \n
    #  psis_dict_in = { 
    #  - 'Summed' : {...}
    #  - 'Matched' : {...}
    #  },
    #  where each {...} must be an instance of results.results_psi_single_all.fit.
    #  @param psim_dict_in
    #  dictionary \n
    #  The dictionary for the double flavor models.
    #  See fits_Einasto_all.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be of the form: \n
    #  psim_dict_in = { 
    #  - 'Summed' : {...}
    #  - 'Matched' : {...}
    #  },
    #  where each {...} must be an instance of results.results_psi_multi_all.fit.
    #  @param m22_ex (optional)
    #  float \n
    #  Value for particle mass (single flavor models) or particle mass one (double flavor models).
    #  @param m22_2_ex (optional)
    #  float \n
    #  Value for particle mass two (double flavor models).
    #  @param save_file (optional)
    #  str \n
    #  File path to save plot.  Default is None, in which case file is not saved.
    #  @returns
    #  matplotlib.pyplot.figure[3,2] \n
    #  Figure of the SH relation for each ULDM (particle mass fixed) model and the Einasto model.
    def Msol_psi_mfix_ex(self,psis_dict_in,psim_dict_in,m22_ex=10**(1.5),m22_2_ex=10**(1.8),save_file=None):
        _,axs=plt.subplots(3,2,figsize=(25,35))
        plt_0=0
        for key in psis_dict_in:
            x_tab=np.concatenate((psis_dict_in[key]['Vbulge_none']['m='+str(np.log10(m22_ex))]['params']['Msol'],
                psis_dict_in[key]['Vbulge']['m='+str(np.log10(m22_ex))]['params']['Msol']))
            mtab_in=np.concatenate((psis_dict_in[key]['Vbulge_none']['m='+str(np.log10(m22_ex))]['params']['m22'],
                psis_dict_in[key]['Vbulge']['m='+str(np.log10(m22_ex))]['params']['m22']))
            y_tab=np.concatenate((psis_dict_in[key]['Vbulge_none']['m='+str(np.log10(m22_ex))]['Mhalo'],
                psis_dict_in[key]['Vbulge']['m='+str(np.log10(m22_ex))]['Mhalo']))
            y_tab=1.4e9*mtab_in**(-1)*(y_tab/1e12)**(1/3)
            y_tab=unumpy.log10(x_tab/y_tab)
            x_tab=unumpy.log10(x_tab)
            x_vals=np.asarray([])
            x_err=np.asarray([])
            y_vals=np.asarray([])
            y_err=np.asarray([])
            for i in range(len(x_tab)):
                if (x_tab[i].std_dev!=np.inf and y_tab[i].std_dev!=np.inf):
                    x_vals=np.append(x_vals,x_tab[i].nominal_value)
                    x_err=np.append(x_err,x_tab[i].std_dev)
                    y_vals=np.append(y_vals,y_tab[i].nominal_value)
                    y_err=np.append(y_err,y_tab[i].std_dev)
            axs[0,plt_0].errorbar(x_vals,y_vals,xerr=x_err,yerr=y_err,c='k',fmt='o')
            x_vals=np.asarray([])
            x_err=np.asarray([])
            y_vals=np.asarray([])
            for i in range(len(x_tab)):
                if (x_tab[i].std_dev!=np.inf and y_tab[i].std_dev==np.inf):
                    x_vals=np.append(x_vals,x_tab[i].nominal_value)
                    x_err=np.append(x_err,x_tab[i].std_dev)
                    y_vals=np.append(y_vals,y_tab[i].nominal_value)
            axs[0,plt_0].errorbar(x_vals,y_vals,xerr=x_err,yerr=None,c='b',fmt='v')
            x_vals=np.asarray([])
            y_vals=np.asarray([])
            y_err=np.asarray([])
            for i in range(len(x_tab)):
                if (x_tab[i].std_dev==np.inf and y_tab[i].std_dev!=np.inf):
                    x_vals=np.append(x_vals,x_tab[i].nominal_value)
                    y_vals=np.append(y_vals,y_tab[i].nominal_value)
                    y_err=np.append(y_err,y_tab[i].std_dev)   
            axs[0,plt_0].errorbar(x_vals,y_vals,xerr=None,yerr=y_err,c='r',fmt='s')
            x_vals=np.asarray([])
            y_vals=np.asarray([])
            for i in range(len(x_tab)):
                if (x_tab[i].std_dev==np.inf and y_tab[i].std_dev==np.inf):
                    x_vals=np.append(x_vals,x_tab[i].nominal_value)
                    y_vals=np.append(y_vals,y_tab[i].nominal_value)
            axs[0,plt_0].scatter(x_vals,y_vals,c='g',marker='x')
            axs[0,plt_0].set_ylabel(r'$\log_{10}\left(M_\mathrm{sol}/M_{\mathrm{SH}}\right)$')
            axs[0,plt_0].set_xlabel(r'$\log_{10}\left(M_\mathrm{sol} \, [M_\odot]\right)$')
            axs[0,plt_0].set_title('Single, ' + str(key))
            plt_0+=1
        plt_0=0
        for key in psim_dict_in:
            x_tab=np.concatenate((psim_dict_in[key]['Vbulge_none']['m='+str(np.log10(m22_2_ex))]['params']['Msol'],
                psim_dict_in[key]['Vbulge']['m='+str(np.log10(m22_2_ex))]['params']['Msol']))
            mtab_1_in=np.concatenate((psim_dict_in[key]['Vbulge_none']['m='+str(np.log10(m22_2_ex))]['params']['m22'],
                psim_dict_in[key]['Vbulge']['m='+str(np.log10(m22_2_ex))]['params']['m22']))
            y_tab=np.concatenate((psim_dict_in[key]['Vbulge_none']['m='+str(np.log10(m22_2_ex))]['Mhalo_1'],
                psim_dict_in[key]['Vbulge']['m='+str(np.log10(m22_2_ex))]['Mhalo_1']))
            y_tab=1.4e9*mtab_1_in**(-1)*(y_tab/1e12)**(1/3)
            y_tab=unumpy.log10(x_tab/y_tab)
            x_tab=unumpy.log10(x_tab)
            x_vals=np.asarray([])
            x_err=np.asarray([])
            y_vals=np.asarray([])
            y_err=np.asarray([])
            for i in range(len(x_tab)):
                if (x_tab[i].std_dev!=np.inf and y_tab[i].std_dev!=np.inf):
                    x_vals=np.append(x_vals,x_tab[i].nominal_value)
                    x_err=np.append(x_err,x_tab[i].std_dev)
                    y_vals=np.append(y_vals,y_tab[i].nominal_value)
                    y_err=np.append(y_err,y_tab[i].std_dev)
            axs[1,plt_0].errorbar(x_vals,y_vals,xerr=x_err,yerr=y_err,c='k',fmt='o')
            x_vals=np.asarray([])
            x_err=np.asarray([])
            y_vals=np.asarray([])
            for i in range(len(x_tab)):
                if (x_tab[i].std_dev!=np.inf and y_tab[i].std_dev==np.inf):
                    x_vals=np.append(x_vals,x_tab[i].nominal_value)
                    x_err=np.append(x_err,x_tab[i].std_dev)
                    y_vals=np.append(y_vals,y_tab[i].nominal_value)
            axs[1,plt_0].errorbar(x_vals,y_vals,xerr=x_err,yerr=None,c='b',fmt='v')
            x_vals=np.asarray([])
            y_vals=np.asarray([])
            y_err=np.asarray([])
            for i in range(len(x_tab)):
                if (x_tab[i].std_dev==np.inf and y_tab[i].std_dev!=np.inf):
                    x_vals=np.append(x_vals,x_tab[i].nominal_value)
                    y_vals=np.append(y_vals,y_tab[i].nominal_value)
                    y_err=np.append(y_err,y_tab[i].std_dev)   
            axs[1,plt_0].errorbar(x_vals,y_vals,xerr=None,yerr=y_err,c='r',fmt='s')
            x_vals=np.asarray([])
            y_vals=np.asarray([])
            for i in range(len(x_tab)):
                if (x_tab[i].std_dev==np.inf and y_tab[i].std_dev==np.inf):
                    x_vals=np.append(x_vals,x_tab[i].nominal_value)
                    y_vals=np.append(y_vals,y_tab[i].nominal_value)
            axs[1,plt_0].scatter(x_vals,y_vals,c='g',marker='x')
            axs[1,plt_0].set_ylabel(r'$\log_{10}\left(M_{\mathrm{sol},1}/M_{\mathrm{SH},1}\right)$')
            axs[1,plt_0].set_xlabel(r'$\log_{10}\left(M_{\mathrm{sol},1} \, [M_\odot]\right)$')
            axs[1,plt_0].set_title('Double, ' + str(key))
            x_tab=np.concatenate((psim_dict_in[key]['Vbulge_none']['m='+str(np.log10(m22_2_ex))]['params']['Msol_2'],
                psim_dict_in[key]['Vbulge']['m='+str(np.log10(m22_2_ex))]['params']['Msol_2']))
            mtab_2_in=np.concatenate((psim_dict_in[key]['Vbulge_none']['m='+str(np.log10(m22_2_ex))]['params']['m22_2'],
                psim_dict_in[key]['Vbulge']['m='+str(np.log10(m22_2_ex))]['params']['m22_2']))
            y_tab=np.concatenate((psim_dict_in[key]['Vbulge_none']['m='+str(np.log10(m22_2_ex))]['Mhalo_2'],
                psim_dict_in[key]['Vbulge']['m='+str(np.log10(m22_2_ex))]['Mhalo_2']))
            y_tab=1.4e9*mtab_2_in**(-1)*(y_tab/1e12)**(1/3)
            y_tab=unumpy.log10(x_tab/y_tab)
            x_tab=unumpy.log10(x_tab)
            x_vals=np.asarray([])
            x_err=np.asarray([])
            y_vals=np.asarray([])
            y_err=np.asarray([])
            for i in range(len(x_tab)):
                if (x_tab[i].std_dev!=np.inf and y_tab[i].std_dev!=np.inf):
                    x_vals=np.append(x_vals,x_tab[i].nominal_value)
                    x_err=np.append(x_err,x_tab[i].std_dev)
                    y_vals=np.append(y_vals,y_tab[i].nominal_value)
                    y_err=np.append(y_err,y_tab[i].std_dev)
            axs[2,plt_0].errorbar(x_vals,y_vals,xerr=x_err,yerr=y_err,c='k',fmt='o')
            x_vals=np.asarray([])
            x_err=np.asarray([])
            y_vals=np.asarray([])
            for i in range(len(x_tab)):
                if (x_tab[i].std_dev!=np.inf and y_tab[i].std_dev==np.inf):
                    x_vals=np.append(x_vals,x_tab[i].nominal_value)
                    x_err=np.append(x_err,x_tab[i].std_dev)
                    y_vals=np.append(y_vals,y_tab[i].nominal_value)
            axs[2,plt_0].errorbar(x_vals,y_vals,xerr=x_err,yerr=None,c='b',fmt='v')
            x_vals=np.asarray([])
            y_vals=np.asarray([])
            y_err=np.asarray([])
            for i in range(len(x_tab)):
                if (x_tab[i].std_dev==np.inf and y_tab[i].std_dev!=np.inf):
                    x_vals=np.append(x_vals,x_tab[i].nominal_value)
                    y_vals=np.append(y_vals,y_tab[i].nominal_value)
                    y_err=np.append(y_err,y_tab[i].std_dev)   
            axs[2,plt_0].errorbar(x_vals,y_vals,xerr=None,yerr=y_err,c='r',fmt='s')
            x_vals=np.asarray([])
            y_vals=np.asarray([])
            for i in range(len(x_tab)):
                if (x_tab[i].std_dev==np.inf and y_tab[i].std_dev==np.inf):
                    x_vals=np.append(x_vals,x_tab[i].nominal_value)
                    y_vals=np.append(y_vals,y_tab[i].nominal_value)
            axs[2,plt_0].scatter(x_vals,y_vals,c='g',marker='x')
            axs[2,plt_0].set_ylabel(r'$\log_{10}\left(M_{\mathrm{sol},2}/M_{\mathrm{SH},2}\right)$')
            axs[2,plt_0].set_xlabel(r'$\log_{10}\left(M_{\mathrm{sol},2} \, [M_\odot]\right)$')
            axs[2,plt_0].set_title('Double, ' + str(key))
            plt_0+=1
        for i in range(2):
            for j in range(3):
                axs[j,i].axhline(np.log10(1),linestyle='--',c='k')
                axs[j,i].axhline(np.log10(2),c='b')
                axs[j,i].axhline(np.log10(0.5),c='b')
                axs[j,i].set_xlim(4,12.5)
                axs[j,i].set_ylim(-2.5,2.5)
                axs[j,i].tick_params(axis='both', which='major', labelsize=20, width=2.5, length=10)
                axs[j,i].tick_params(axis='both', which='minor', labelsize=15, width=2, length=5)
        if save_file!=None:
            plt.savefig(save_file)
        return

    ## Define the plot of the reduced chi-square distributions for the ULDM (particle mass fixed) models.
    #  This defines the plot of the reduced chi-square distributions for each of the ULDM (particle mass fixed) models.
    #  @param self
    #  object pointer
    #  @param psis_dict_in
    #  dictionary \n
    #  The dictionary for the single flavor models.
    #  See fits_Einasto_all.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be of the form: \n
    #  psis_dict_in = { 
    #  - 'Summed' : {...}
    #  - 'Matched' : {...}
    #  },
    #  where each {...} must be an instance of results.results_psi_single_all.fit.
    #  @param psim_dict_in
    #  dictionary \n
    #  The dictionary for the double flavor models.
    #  See fits_Einasto_all.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be of the form: \n
    #  psim_dict_in = { 
    #  - 'Summed' : {...}
    #  - 'Matched' : {...}
    #  },
    #  where each {...} must be an instance of results.results_psi_multi_all.fit.
    #  @param m22_ex (optional)
    #  float \n
    #  Value for particle mass (single flavor models) or particle mass one (double flavor models).
    #  @param m22_2_ex (optional)
    #  float \n
    #  Value for particle mass two (double flavor models).
    #  @param bins (optional)
    #  int \n
    #  Number of bins for histogram plot.
    #  @param save_file (optional)
    #  str \n
    #  File path to save plot.  Default is None, in which case file is not saved.
    #  @returns
    #  matplotlib.pyplot.figure[2,2] \n
    #  Figure of the reduced chi-square distributions for each of the ULDM (particle mass fixed) models.
    def chi_dist_psi_mfix_ex(self,psis_dict_in,psim_dict_in,m22_ex=10**(1.5),m22_2_ex=10**(1.8),bins=50,save_file=None):
        _,axs=plt.subplots(2,2,figsize=(25,25))
        model_tab=['Single, Summed','Single, Matched','Double, Summed','Double, Matched']
        in_tab=[]
        in_tabb=[]
        for key in psis_dict_in:
            in_tab.append(psis_dict_in[key]['Vbulge_none']['m='+str(np.log10(m22_ex))])
            in_tabb.append(psis_dict_in[key]['Vbulge']['m='+str(np.log10(m22_ex))])
        for key in psim_dict_in:
            in_tab.append(psim_dict_in[key]['Vbulge_none']['m='+str(np.log10(m22_2_ex))])
            in_tabb.append(psim_dict_in[key]['Vbulge']['m='+str(np.log10(m22_2_ex))])
        k=0
        j=0
        for i in range(len(in_tab)):
            mask=np.isfinite(in_tab[i]['Chi_sq'])
            mod_new=in_tab[i]['Chi_sq'][mask]
            mask=np.isfinite(in_tabb[i]['Chi_sq'])
            mod_new_1=in_tabb[i]['Chi_sq'][mask]
            axs[j,k].hist(mod_new,bins,label=r'No $V_\mathrm{bulge}$')
            axs[j,k].hist(mod_new_1,bins,label=r'With $V_\mathrm{bulge}$')
            axs[j,k].set_xlabel(r'$\chi^2_{\nu,\mathrm{ULDM}}$')
            axs[j,k].set_title(model_tab[i])
            axs[j,k].set_ylabel('Number of galaxies')
            axs[j,k].tick_params(axis='both', which='major', labelsize=20, width=2.5, length=10)
            axs[j,k].tick_params(axis='both', which='minor', labelsize=15, width=2, length=5)
            if k<1:
                k+=1
            else:
                k=0
                j+=1
        if save_file!=None:
            plt.savefig(save_file)
        return

    ## Define the plot of the parameter distributions for the ULDM (particle mass fixed) models.
    #  This defines the plot of the parameter distributions for each of the ULDM (particle mass fixed) models.
    #  @param self
    #  object pointer
    #  @param psis_dict_in
    #  dictionary \n
    #  The dictionary for the single flavor models.
    #  See fits_Einasto_all.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be of the form: \n
    #  psis_dict_in = { 
    #  - 'Summed' : {...}
    #  - 'Matched' : {...}
    #  },
    #  where each {...} must be an instance of results.results_psi_single_all.fit.
    #  @param psim_dict_in
    #  dictionary \n
    #  The dictionary for the double flavor models.
    #  See fits_Einasto_all.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be of the form: \n
    #  psim_dict_in = { 
    #  - 'Summed' : {...}
    #  - 'Matched' : {...}
    #  },
    #  where each {...} must be an instance of results.results_psi_multi_all.fit.
    #  @param bins (optional)
    #  int \n
    #  Number of bins for histogram plot.
    #  @param m22_ex (optional)
    #  float \n
    #  Value for particle mass (single flavor models) or particle mass one (double flavor models).
    #  @param m22_2_ex (optional)
    #  float \n
    #  Value for particle mass two (double flavor models).
    #  @param save_file (optional)
    #  str \n
    #  File path to save plot.  Default is None, in which case file is not saved.
    #  @returns
    #  matplotlib.pyplot.figure[4,4] \n
    #  Figure of the parameter distributions for each of the ULDM (particle mass fixed) models.
    def params_dist_psi_mfix_ex(self,psis_dict_in,psim_dict_in,m22_ex=10**(1.5),m22_2_ex=10**(1.8),bins=50,save_file=None):
        _,axs=plt.subplots(4,4,figsize=(50,50))
        title_tab=['Single, Summed','Single, Matched','Double, Summed','Double, Matched']
        in_tab=[]
        in_tabb=[]
        for key in psis_dict_in:
                in_tab.append(psis_dict_in[key]['Vbulge_none']['m='+str(np.log10(m22_ex))]['params'])
                in_tabb.append(psis_dict_in[key]['Vbulge']['m='+str(np.log10(m22_ex))]['params'])
        for key in psim_dict_in:
                in_tab.append(psim_dict_in[key]['Vbulge_none']['m='+str(np.log10(m22_2_ex))]['params'])
                in_tabb.append(psim_dict_in[key]['Vbulge']['m='+str(np.log10(m22_2_ex))]['params'])
        for i in range(len(in_tab)):
            input_arr=in_tab[i]
            input_arrb=in_tabb[i]
            in_vals=np.asarray([])
            for j in range(len(input_arr['c200'])):
                    in_vals=np.append(in_vals,input_arr['c200'][j].nominal_value)
            in_valsb=np.asarray([])
            for j in range(len(input_arrb['c200'])):
                    in_valsb=np.append(in_valsb,input_arrb['c200'][j].nominal_value)
            axs[0,i].hist(in_vals,bins,label=r'No $V_\mathrm{bulge}$');
            axs[0,i].hist(in_valsb,bins,label=r'With $V_\mathrm{bulge}$');
            axs[0,i].set_xlabel(r'$c_{200}$')
            in_vals=np.asarray([])
            for j in range(len(input_arr['v200'])):
                    in_vals=np.append(in_vals,input_arr['v200'][j].nominal_value)
            in_valsb=np.asarray([])
            for j in range(len(input_arrb['v200'])):
                    in_valsb=np.append(in_valsb,input_arrb['v200'][j].nominal_value)
            axs[1,i].hist(in_vals,bins,label=r'No $V_\mathrm{bulge}$');
            axs[1,i].hist(in_valsb,bins,label=r'With $V_\mathrm{bulge}$');
            axs[1,i].set_xlabel(r'$V_{200} \, [\mathrm{km/s}]$') 
        for i in range(len(in_tab)-2):
            input_arr=in_tab[i]
            input_arrb=in_tabb[i]
            in_vals=np.asarray([])
            for j in range(len(input_arr['MLd'])):
                    in_vals=np.append(in_vals,input_arr['MLd'][j].nominal_value)
            in_valsb=np.asarray([])
            for j in range(len(input_arrb['MLd'])):
                    in_valsb=np.append(in_valsb,input_arrb['MLd'][j].nominal_value)
            axs[2,i].hist(in_vals,bins,label=r'No $V_\mathrm{bulge}$');
            axs[2,i].hist(in_valsb,bins,label=r'With $V_\mathrm{bulge}$');
            axs[2,i].set_xlabel(r'$\tilde{\Upsilon}_d \, [M_\odot/L_\odot]$') 
            input_arr=in_tab[i]
            input_arrb=in_tabb[i]
            in_valsb=np.asarray([])
            for j in range(len(input_arrb['MLb'])):
                    in_valsb=np.append(in_valsb,input_arrb['MLb'][j].nominal_value) 
            axs[3,i].hist(in_valsb,bins,label=r'With $V_\mathrm{bulge}$');
            axs[3,i].set_xlabel(r'$\tilde{\Upsilon}_b \, [M_\odot/L_\odot]$')    
        for i in range(len(in_tab)-2):
            input_arr=in_tab[i+2]
            input_arrb=in_tabb[i+2]
            in_vals=np.asarray([])
            for j in range(len(input_arr['c200_2'])):
                    in_vals=np.append(in_vals,input_arr['c200_2'][j].nominal_value)
            in_valsb=np.asarray([])
            for j in range(len(input_arrb['c200_2'])):
                    in_valsb=np.append(in_valsb,input_arrb['c200_2'][j].nominal_value)
            axs[0,i+2].hist(in_vals,bins,label=r'No $V_\mathrm{bulge}$');
            axs[0,i+2].hist(in_valsb,bins,label=r'With $V_\mathrm{bulge}$');
            in_vals=np.asarray([])
            for j in range(len(input_arr['v200_2'])):
                    in_vals=np.append(in_vals,input_arr['v200_2'][j].nominal_value)
            in_valsb=np.asarray([])
            for j in range(len(input_arrb['v200_2'])):
                    in_valsb=np.append(in_valsb,input_arrb['v200_2'][j].nominal_value)
            axs[1,i+2].hist(in_vals,bins,label=r'No $V_\mathrm{bulge}$');
            axs[1,i+2].hist(in_valsb,bins,label=r'With $V_\mathrm{bulge}$');
            in_vals=np.asarray([])
            for j in range(len(input_arr['MLd'])):
                    in_vals=np.append(in_vals,input_arr['MLd'][j].nominal_value)
            in_valsb=np.asarray([])
            for j in range(len(input_arrb['MLd'])):
                    in_valsb=np.append(in_valsb,input_arrb['MLd'][j].nominal_value)
            axs[2,i+2].hist(in_vals,bins,label=r'No $V_\mathrm{bulge}$');
            axs[2,i+2].hist(in_valsb,bins,label=r'With $V_\mathrm{bulge}$');
            axs[2,i+2].set_xlabel(r'$\tilde{\Upsilon}_d \, [M_\odot/L_\odot]$')  
            in_valsb=np.asarray([])
            for j in range(len(input_arrb['MLb'])):
                    in_valsb=np.append(in_valsb,input_arrb['MLb'][j].nominal_value)
            axs[3,i+2].hist(in_valsb,bins,label=r'With $V_\mathrm{bulge}$');
            axs[3,i+2].set_xlabel(r'$\tilde{\Upsilon}_b \, [M_\odot/L_\odot]$')  
        for i in range(len(in_tab)):
            input_arr=in_tab[i]
            input_arrb=in_tabb[i]
            for j in range(len(in_tab)):
                axs[j,i].legend()
                axs[j,i].set_ylabel('Number of galaxies')
                axs[j,i].set_title(title_tab[i]) 
                axs[j,i].tick_params(axis='both', which='major', labelsize=20, width=2.5, length=10)
                axs[j,i].tick_params(axis='both', which='minor', labelsize=15, width=2, length=5)
        if save_file!=None:       
            plt.savefig(save_file)
        return

    ## Define the plot of the empirical relations for the ULDM (particle mass fixed) models.
    #  This defines the plot of the empirical relations for each of the ULDM (particle mass fixed) models.
    #  @param self
    #  object pointer
    #  @param psis_dict_in
    #  dictionary \n
    #  The dictionary for the single flavor models.
    #  See fits_Einasto_all.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be of the form: \n
    #  psis_dict_in = { 
    #  - 'Summed' : {...}
    #  - 'Matched' : {...}
    #  },
    #  where each {...} must be an instance of results.results_psi_single_all.fit.
    #  @param psim_dict_in
    #  dictionary \n
    #  The dictionary for the double flavor models.
    #  See fits_Einasto_all.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be of the form: \n
    #  psim_dict_in = { 
    #  - 'Summed' : {...}
    #  - 'Matched' : {...}
    #  },
    #  where each {...} must be an instance of results.results_psi_multi_all.fit.
    #  @param m22_ex (optional)
    #  float \n
    #  Value for particle mass (single flavor models) or particle mass one (double flavor models).
    #  @param m22_2_ex (optional)
    #  float \n
    #  Value for particle mass two (double flavor models).
    #  @param save_file (optional)
    #  str \n
    #  File path to save plot.  Default is None, in which case file is not saved.
    #  @returns
    #  matplotlib.pyplot.figure[4,4] \n
    #  Figure of the empirical relations for each of the ULDM (particle mass fixed) models.
    def relations_psi_mfix_ex(self,psis_dict_in,psim_dict_in,m22_ex=10**(1.5),m22_2_ex=10**(1.8),save_file=None):
        model_tab=['psi_single','psi_multi']
        model_tab_1=['Summed','Matched']
        gbar={}
        gtot={}
        gMOND={}
        fit_results={}
        for mod in model_tab:
            gbar[mod]={}
            gtot[mod]={}
            gMOND[mod]={}
            fit_results[mod]={}
            for mod1 in model_tab_1:
                gbar[mod][mod1]=np.asarray([])
                gtot[mod][mod1]=np.asarray([])
                gMOND[mod][mod1]=np.asarray([])
                fit_results[mod][mod1]=0
        fit_dict_ex=constants.fitting_dict(sol_m22_tab_prime=np.asarray([m22_ex]),sol_m22_2_tab_prime=np.asarray([m22_2_ex]))
        for key in gbar:
            fit_init=model_fit.grar_fit(key,ULDM_fits=True,fit_dict_in=fit_dict_ex)
            fit=fit_init.fit()
            gbar[key]['Summed']=fit['gbar']
            gtot[key]['Summed']=fit['gtot']
            gMOND[key]['Summed']=fit['gMOND']
            fit_results[key]['Summed']=fit['fit']
        fit_dict_ex=constants.fitting_dict(sol_match=True,sol_m22_tab=np.asarray([m22_ex]),sol_m22_2_tab=np.asarray([m22_2_ex]))
        for key in gbar:
            fit_init=model_fit.grar_fit(key,ULDM_fits=True,fit_dict_in=fit_dict_ex)
            fit=fit_init.fit()
            gbar[key]['Matched']=fit['gbar']
            gtot[key]['Matched']=fit['gtot']
            gMOND[key]['Matched']=fit['gMOND']
            fit_results[key]['Matched']=fit['fit']
        print(fit_results['psi_single']['Summed'].params)
        print(fit_results['psi_single']['Matched'].params)
        print(fit_results['psi_multi']['Summed'].params)
        print(fit_results['psi_multi']['Matched'].params)
        params_arr=unumpy.uarray([fit_results['psi_single']['Summed'].params['gdag'].value,
            fit_results['psi_single']['Matched'].params['gdag'].value,
            fit_results['psi_multi']['Summed'].params['gdag'].value,
        fit_results['psi_multi']['Matched'].params['gdag'].value],
        [fit_results['psi_single']['Summed'].params['gdag'].stderr,
            fit_results['psi_single']['Matched'].params['gdag'].stderr,
            fit_results['psi_multi']['Summed'].params['gdag'].stderr,
        fit_results['psi_multi']['Matched'].params['gdag'].stderr])
        params_arr_log=unumpy.log10(params_arr)
        label_tab=np.asarray([r'Model : $\log_{10} \, g^{\dagger} = $' 
            + str(round(params_arr_log[0].nominal_value,2))
            + r'$\pm$' + str(round(params_arr_log[0].std_dev,4)) 
            + r'$\, [\log_{10}\left(\mathrm{m} \, s^{-2}\right)]$',
            r'Model : $\log_{10} \, g^{\dagger} = $' 
            + str(round(params_arr_log[1].nominal_value,2))
            + r'$\pm$' + str(round(params_arr_log[1].std_dev,4)) 
            + r'$\, [\log_{10}\left(\mathrm{m} \, s^{-2}\right)]$',
            r'Model : $\log_{10} \, g^{\dagger} = $' 
            + str(round(params_arr_log[2].nominal_value,2))
            + r'$\pm$' + str(round(params_arr_log[2].std_dev,4)) 
            + r'$\, [\log_{10}\left(\mathrm{m} \, s^{-2}\right)]$',
            r'Model : $\log_{10} \, g^{\dagger} = $' 
            + str(round(params_arr_log[3].nominal_value,2))
            + r'$\pm$' + str(round(params_arr_log[3].std_dev,4)) 
            + r'$\, [\log_{10}\left(\mathrm{m} \, s^{-2}\right)]$'])
        model_tab=np.asarray(['psi_single','psi_single','psi_multi','psi_multi'])
        model_tab_1=np.asarray(['Summed','Matched','Summed','Matched'])
        fit_dict_tab=np.asarray([
            constants.fitting_dict(sol_m22_tab_prime=np.asarray([m22_ex]),sol_m22_2_tab_prime=np.asarray([m22_2_ex])),
            constants.fitting_dict(sol_match=True,sol_m22_tab=np.asarray([m22_ex]),sol_m22_2_tab=np.asarray([m22_2_ex])),
            constants.fitting_dict(sol_m22_tab_prime=np.asarray([m22_ex]),sol_m22_2_tab_prime=np.asarray([m22_2_ex])),
            constants.fitting_dict(sol_match=True,sol_m22_tab=np.asarray([m22_ex]),sol_m22_2_tab=np.asarray([m22_2_ex]))
            ])
        title_tab=np.asarray(['Single, Summed','Single, Matched','Double, Summed','Double, Matched'])
        _,axs=plt.subplots(4,4,figsize=(50,50))
        for i in range(4):
            axs[0,i].scatter(gbar[model_tab[i]][model_tab_1[i]],gtot[model_tab[i]][model_tab_1[i]],label=r'Data points',c='k')
            axs[0,i].scatter(gbar[model_tab[i]][model_tab_1[i]],gMOND[model_tab[i]][model_tab_1[i]],
                label=r'MOND : $\log_{10} \, g^{\dagger} = $' 
                    + str(round(np.log10(1.2*1e-10),2))
                    + r'$\, [\log_{10}\left(\mathrm{m} \, s^{-2}\right)]$')
            fit_init=model_fit.grar_fit(model_tab[i],ULDM_fits=True,fit_dict_in=fit_dict_tab[i])
            axs[0,i].scatter(gbar[model_tab[i]][model_tab_1[i]],
                fit_init.grar_model(gbar[model_tab[i]][model_tab_1[i]],fit_results[model_tab[i]][model_tab_1[i]].params),
                label=label_tab[i])
            axs[0,i].axline([0,0],[10**(-8.5),10**(-8.5)],linestyle='-',c='k')
            axs[0,i].set_xlabel(r'$g_\mathrm{bar} \, [\mathrm{m}/\mathrm{s}^2]$')
            axs[0,i].set_ylabel(r'$g_\mathrm{tot} \, [\mathrm{m}/\mathrm{s}^2]$')
            axs[0,i].set_xscale('log')
            axs[0,i].set_yscale('log')
            axs[0,i].set_title(str(title_tab[i]))
            axs[0,i].grid(visible=True,axis='both',c='k',alpha=0.4)
            axs[0,i].legend()
        model_tab=['Summed','Matched','Summed','Matched']
        title_tab=np.asarray(['Single, Summed','Single, Matched','Double, Summed','Double, Matched'])
        model_tab_1=[psis_dict_in,psis_dict_in,psim_dict_in,psim_dict_in]
        plt_0=0
        for mod in model_tab:
            if model_tab_1[plt_0]==psis_dict_in:
                m22_in=m22_ex
            else:
                m22_in=m22_2_ex
            in0=unumpy.log10(model_tab_1[plt_0][mod]['Vbulge_none']['m='+str(np.log10(m22_in))]['params']['c200'])
            in1=unumpy.log10(model_tab_1[plt_0][mod]['Vbulge']['m='+str(np.log10(m22_in))]['params']['c200'])
            c200_in=np.concatenate((in0,in1))
            in0=np.log10(model_tab_1[plt_0][mod]['Vbulge_none']['m='+str(np.log10(m22_in))]['Mvir'])
            in1=np.log10(model_tab_1[plt_0][mod]['Vbulge']['m='+str(np.log10(m22_in))]['Mvir'])
            m200_in=np.concatenate((in0,in1))
            args_sort=np.argsort(m200_in)
            m200_in=m200_in[args_sort]
            c200CMR_Wa=np.asarray([])
            for i in range(len(m200_in)):
                c200CMR_Wa=np.append(c200CMR_Wa,model_fit.conc_mass_rel_Wa(10**(m200_in[i])))
            c200CMR_Wa_val=np.log10(c200CMR_Wa)
            axs[1,plt_0].plot(m200_in,c200CMR_Wa_val,label=r'CMR_W : $M_{200}$',c='k',linestyle='solid')
            c200CMR_Du=model_fit.conc_mass_rel_Du(10**(m200_in))
            c200CMR_Du_val=c200CMR_Du
            axs[1,plt_0].plot(m200_in,c200CMR_Du_val,label=r'CMR_D : $M_{200}$',c='k',linestyle='dashed')
            c200_vals=np.asarray([])
            c200_err=np.asarray([])
            m200_vals=np.asarray([])
            for i in range(len(c200_in)):
                if c200_in[i].std_dev!=np.inf:
                    c200_vals=np.append(c200_vals,c200_in[i].nominal_value)
                    c200_err=np.append(c200_err,c200_in[i].std_dev)
                    m200_vals=np.append(m200_vals,m200_in[i])
            axs[1,plt_0].errorbar(m200_vals,c200_vals,xerr=None,yerr=c200_err,c='k',fmt='o')
            c200_vals=np.asarray([])
            m200_vals=np.asarray([])
            for i in range(len(c200_in)):
                if c200_in[i].std_dev==np.inf:
                    c200_vals=np.append(c200_vals,c200_in[i].nominal_value)
                    m200_vals=np.append(m200_vals,m200_in[i])
            axs[1,plt_0].scatter(m200_vals,c200_vals,c='g',marker='x')
            if model_tab_1[plt_0]==psim_dict_in:
                in0=unumpy.log10(model_tab_1[plt_0][mod]['Vbulge_none']['m='+str(np.log10(m22_in))]['params']['c200_2'])
                in1=unumpy.log10(model_tab_1[plt_0][mod]['Vbulge']['m='+str(np.log10(m22_in))]['params']['c200_2'])
                c200_in=np.concatenate((in0,in1))
            c200_vals=np.asarray([])
            c200_err=np.asarray([])
            m200_vals=np.asarray([])
            for i in range(len(c200_in)):
                if c200_in[i].std_dev!=np.inf:
                    c200_vals=np.append(c200_vals,c200_in[i].nominal_value)
                    c200_err=np.append(c200_err,c200_in[i].std_dev)
                    m200_vals=np.append(m200_vals,m200_in[i])
            axs[1,plt_0].errorbar(m200_vals,c200_vals,xerr=None,yerr=c200_err,c='k',fmt='o')
            c200_vals=np.asarray([])
            m200_vals=np.asarray([])
            for i in range(len(c200_in)):
                if c200_in[i].std_dev==np.inf:
                    c200_vals=np.append(c200_vals,c200_in[i].nominal_value)
                    m200_vals=np.append(m200_vals,m200_in[i])
            axs[1,plt_0].scatter(m200_vals,c200_vals,c='g',marker='x')
            axs[1,plt_0].set_ylabel(r'$\log_{10}c_{200}$')
            axs[1,plt_0].set_xlabel(r'$\log_{10}\left(M_{200} \, [M_{\odot}]\right)$')
            axs[1,plt_0].set_title(title_tab[plt_0])
            axs[1,plt_0].legend()
            axs[1,plt_0].grid(visible=True,axis='both',c='k',alpha=0.4)
            axs[1,plt_0].set_xlim(8.5,15.5)
            axs[1,plt_0].set_ylim(-0.25,2.25)
            plt_0+=1
        plt_0=0
        for mod in model_tab:
            if model_tab_1[plt_0]==psis_dict_in:
                m22_in=m22_ex
            else:
                m22_in=m22_2_ex
            in0=np.log10(model_tab_1[plt_0][mod]['Vbulge_none']['m='+str(np.log10(m22_in))]['params']['Vf'])
            in1=np.log10(model_tab_1[plt_0][mod]['Vbulge']['m='+str(np.log10(m22_in))]['params']['Vf'])
            Vf_in=np.concatenate((in0,in1))
            axs[2,plt_0].plot(Vf_in,model_fit.BTFR(10**(Vf_in)),label='BTFR',c='k',linestyle='solid')
            mass_in=model_tab_1[plt_0][mod]['Vbulge_none']['m='+str(np.log10(m22_in))]['params']['mstar']+model_tab_1[plt_0][mod]['Vbulge_none']['m='+str(np.log10(m22_in))]['params']['mgas']
            in0=unumpy.log10((mass_in)*1e9)
            mass_in=model_tab_1[plt_0][mod]['Vbulge']['m='+str(np.log10(m22_in))]['params']['mstar']+model_tab_1[plt_0][mod]['Vbulge']['m='+str(np.log10(m22_in))]['params']['mgas']
            in1=unumpy.log10((mass_in)*1e9)
            mstar_in=np.concatenate((in0,in1))
            mstar_vals=np.asarray([])
            mstar_err=np.asarray([])
            Vf_vals=np.asarray([])
            for i in range(len(mstar_in)):
                if mstar_in[i].std_dev!=np.inf:
                    mstar_vals=np.append(mstar_vals,mstar_in[i].nominal_value)
                    mstar_err=np.append(mstar_err,mstar_in[i].std_dev)
                    Vf_vals=np.append(Vf_vals,Vf_in[i])
            axs[2,plt_0].errorbar(Vf_vals,mstar_vals,xerr=None,yerr=mstar_err,c='k',fmt='o')
            mstar_vals=np.asarray([])
            Vf_vals=np.asarray([])
            for i in range(len(mstar_in)):
                if mstar_in[i].std_dev==np.inf:
                    mstar_vals=np.append(mstar_vals,mstar_in[i].nominal_value)
                    Vf_vals=np.append(Vf_vals,Vf_in[i])
            axs[2,plt_0].scatter(Vf_vals,mstar_vals,c='g',marker='x')
            axs[2,plt_0].set_ylabel(r'$\log_{10} \left(M_b \, [M_\odot]\right)$')
            axs[2,plt_0].set_xlabel(r'$\log_{10} \left(V_f \, [km/s]\right)$')
            axs[2,plt_0].set_title(title_tab[plt_0])
            axs[2,plt_0].legend()
            axs[2,plt_0].grid(visible=True,axis='both',c='k',alpha=0.4)
            axs[2,plt_0].set_ylim(7.5,12.5)
            plt_0+=1
        plt_0=0
        for mod in model_tab:
            if model_tab_1[plt_0]==psis_dict_in:
                m22_in=m22_ex
            else:
                m22_in=m22_2_ex
            in0=model_tab_1[plt_0][mod]['Vbulge_none']['m='+str(np.log10(m22_in))]['params']['mstar']*1e9
            in1=model_tab_1[plt_0][mod]['Vbulge']['m='+str(np.log10(m22_in))]['params']['mstar']*1e9
            mstar_in=np.concatenate((in0,in1))
            mstar_in=unumpy.log10(mstar_in)
            in0=model_tab_1[plt_0][mod]['Vbulge_none']['m='+str(np.log10(m22_in))]['Mvir']
            in1=model_tab_1[plt_0][mod]['Vbulge']['m='+str(np.log10(m22_in))]['Mvir']
            m200_in=np.concatenate((in0,in1))
            args_sort=np.argsort(m200_in)
            m200_in=m200_in[args_sort]
            mstar_ex=np.log10(model_fit.abund_match_rel(m200_in))
            axs[3,plt_0].plot(np.log10(m200_in),mstar_ex,label='AMR',c='k',linestyle='solid')
            mstar_vals=np.asarray([])
            mstar_err=np.asarray([])
            m200_vals=np.asarray([])
            for i in range(len(mstar_in)):
                if mstar_in[i].std_dev!=np.inf:
                    mstar_vals=np.append(mstar_vals,mstar_in[i].nominal_value)
                    mstar_err=np.append(mstar_err,mstar_in[i].std_dev)
                    m200_vals=np.append(m200_vals,m200_in[i])
            axs[3,plt_0].errorbar(np.log10(m200_vals),mstar_vals,xerr=None,yerr=mstar_err,c='k',fmt='o')
            mstar_vals=np.asarray([])
            m200_vals=np.asarray([])
            for i in range(len(mstar_in)):
                if mstar_in[i].std_dev==np.inf:
                    mstar_vals=np.append(mstar_vals,mstar_in[i].nominal_value)
                    m200_vals=np.append(m200_vals,m200_in[i])
            axs[3,plt_0].scatter(np.log10(m200_vals),mstar_vals,c='g',marker='x')
            axs[3,plt_0].set_ylabel(r'$\log_{10}\left(M_* \, [M_{\odot}]\right)$')
            axs[3,plt_0].set_xlabel(r'$\log_{10}\left(M_{200} \, [M_{\odot}]\right)$')
            axs[3,plt_0].set_title(title_tab[plt_0])
            axs[3,plt_0].legend()
            axs[3,plt_0].grid(visible=True,axis='both',c='k',alpha=0.4)
            axs[3,plt_0].set_xlim(8.5,15.5)
            axs[3,plt_0].set_ylim(4.5,12.5)
            plt_0+=1
        if save_file!=None:
            plt.savefig(save_file)
        return

    ## Define the plot of the reduced chi-square distributions for the CDM models (checks).
    #  This defines the plot of the reduced chi-square distribution for each CDM model analyzed (checks).
    #  @param self
    #  object pointer
    #  @param dict_in
    #  dictionary \n
    #  See checks_CDM_all.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be of the form: \n
    #  dict_in = { 
    #  - 'Burkert' : {...}
    #  - 'DC14' : {...}
    #  - 'Einasto' : {...}
    #  - 'NFW' : {...}
    #  },
    #  where each {...} must be an instance of results.results_CDM_check.fit.
    #  @param bins (optional)
    #  int \n
    #  Number of bins for histogram plot.
    #  @returns
    #  matplotlib.pyplot.figure[1,4] \n
    #  Figure of the reduced chi-square distributions for each CDM model (checks).
    def chi_dist_CDM_checks(self,dict_in,bins=50):
        _,axs=plt.subplots(1,4,figsize=(30,5))
        in_tab=[]
        in_tabb=[]
        for key in dict_in:
            in_tab.append(dict_in[key]['Vbulge_none'])
            in_tabb.append(dict_in[key]['Vbulge'])
        title_tab=['Burkert','DC14','Einasto','NFW']
        for i in range(len(in_tab)):
            axs[i].hist(in_tab[i]['Chi_sq'],bins,label=r'No $V_\mathrm{bulge}$');
            axs[i].hist(in_tabb[i]['Chi_sq'],bins,label=r'With $V_\mathrm{bulge}$');
            axs[i].legend()
            axs[i].set_xlabel(r'$\chi^2_\nu$')
            axs[i].set_ylabel('Number of galaxies')
            axs[i].set_title(title_tab[i])
        return

    ## Define the plot of the parameter distributions for the CDM models (checks).
    #  This defines the plot of the parameter distributions for each CDM model analyzed (checks).
    #  @param self
    #  object pointer
    #  @param dict_in
    #  dictionary \n
    #  See checks_CDM_all.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be of the form: \n
    #  dict_in = { 
    #  - 'Burkert' : {...}
    #  - 'DC14' : {...}
    #  - 'Einasto' : {...}
    #  - 'NFW' : {...}
    #  },
    #  where each {...} must be an instance of results.results_CDM_check.fit.
    #  @param bins (optional)
    #  int \n
    #  Number of bins for histogram plot.
    #  @returns
    #  matplotlib.pyplot.figure[4,4] \n
    #  Figure of the parameter distributions for each CDM model (checks).
    def params_dist_CDM_checks(self,dict_in,bins=50):
        _,axs=plt.subplots(4,4,figsize=(50,50))
        in_tab=[]
        in_tabb=[]
        for key in dict_in:
            in_tab.append(dict_in[key]['Vbulge_none']['params'])
            in_tabb.append(dict_in[key]['Vbulge']['params'])
        title_tab=['Burkert','DC14','Einasto','NFW']
        label_tab=[r'$c_{200}$',r'$V_{200} \, [\mathrm{km/s}]$',r'$\tilde{\Upsilon}_d \, [M_\odot/L_\odot]$']
        for i in range(len(in_tab)):
            input_arr=np.asarray(in_tab[i])
            input_arrb=np.asarray(in_tabb[i])
            for j in range(3):
                axs[j,i].hist(input_arr[:,j],bins,label=r'No $V_\mathrm{bulge}$');
                axs[j,i].hist(input_arrb[:,j],bins,label=r'With $V_\mathrm{bulge}$');
                axs[j,i].set_xlabel(label_tab[j])
            axs[3,i].hist(input_arrb[:,3],bins,label=r'With $V_\mathrm{bulge}$');
            axs[3,i].set_xlabel(r'$\tilde{\Upsilon}_b \, [M_\odot/L_\odot]$')  
            for j in range(4):
                axs[j,i].legend()
                axs[j,i].set_ylabel('Number of galaxies')
                axs[j,i].set_title(title_tab[i])  
        return

    ## Define the plot of the reduced chi-square distributions for the DC14 and NFW models (checks).
    #  This defines the plot of the reduced chi-square distribution for the DC14 and NFW models (checks).
    #  @param self
    #  object pointer
    #  @param dict_in
    #  dictionary \n
    #  See checks_DC14_NFW.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be of the form: \n
    #  dict_in = { 
    #  - 'DC14' : {...}
    #  - 'NFW' : {...}
    #  },
    #  where each {...} must be an instance of results.results_DC14_check.fit.
    #  @param bins (optional)
    #  int \n
    #  Number of bins for histogram plot.
    #  @returns
    #  matplotlib.pyplot.figure[1,2] \n
    #  Figure of the reduced chi-square distributions for the DC14 and NFW models (checks).
    def chi_dist_DC14_checks(self,dict_in,bins=50):
        _,axs=plt.subplots(1,2,figsize=(20,5))
        in_tab=[]
        in_tabb=[]
        for key in dict_in:
            in_tab.append(dict_in[key]['Vbulge_none'])
            in_tabb.append(dict_in[key]['Vbulge'])
        title_tab=['DC14','NFW']
        for i in range(len(in_tab)):
            in_tab[i]['Chi_sq']=np.ma.masked_invalid(in_tab[i]['Chi_sq'])
            in_tabb[i]['Chi_sq']=np.ma.masked_invalid(in_tabb[i]['Chi_sq'])
            axs[i].hist(in_tab[i]['Chi_sq'],bins,label=r'No $V_\mathrm{bulge}$');
            axs[i].hist(in_tabb[i]['Chi_sq'],bins,label=r'With $V_\mathrm{bulge}$');
            axs[i].set_xlabel(r'$\chi^2_\nu$')
            axs[i].legend()
            axs[i].set_ylabel('Number of galaxies')
            axs[i].set_title(title_tab[i])
        return

    ## Define the plot of the parameter distributions for the DC14 and NFW models (checks).
    #  This defines the plot of the parameter distributions for the DC14 and NFW models (checks).
    #  @param self
    #  object pointer
    #  @param dict_in
    #  dictionary \n
    #  See checks_DC14_NFW.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be of the form: \n
    #  dict_in = { 
    #  - 'DC14' : {...}
    #  - 'NFW' : {...}
    #  },
    #  where each {...} must be an instance of results.results_DC14_check.fit.
    #  @param bins (optional)
    #  int \n
    #  Number of bins for histogram plot.
    #  @returns
    #  matplotlib.pyplot.figure[1,2] \n
    #  Figure of the reduced chi-square distributions for the DC14 and NFW models (checks).
    def params_dist_DC14_checks(self,dict_in,bins=50):
        _,axs=plt.subplots(3,2,figsize=(30,30))
        in_tab=[]
        in_tabb=[]
        for key in dict_in:
            in_tab.append(dict_in[key]['Vbulge_none']['params'])
            in_tabb.append(dict_in[key]['Vbulge']['params'])
        title_tab=['DC14','NFW']
        label_tab=[r'$c_{200}$',r'$V_{200} \, [\mathrm{km/s}]$',r'$\tilde{\Upsilon}_d \, [M_\odot/L_\odot]$']
        for i in range(len(in_tab)):
            input_arr=np.asarray(in_tab[i])
            input_arrb=np.asarray(in_tabb[i])
            for j in range(3):
                axs[j,i].hist(input_arr[:,j],bins,label=r'No $V_\mathrm{bulge}$');
                axs[j,i].hist(input_arrb[:,j],bins,label=r'With $V_\mathrm{bulge}$');
                axs[j,i].legend()
                axs[j,i].set_ylabel('Number of galaxies')
                axs[j,i].set_title(title_tab[i])
                axs[j,i].set_xlabel(label_tab[j])  
        return

    ## Define the plot of the reduced chi-square for the Einasto and NFW models (checks).
    #  This defines the plot of the reduced chi-square comparison between the Einasto and NFW models (checks).
    #  @param self
    #  object pointer
    #  @param dict_in
    #  dictionary \n
    #  See checks_Einasto_NFW.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be of the form: \n
    #  dict_in = { 
    #  - 'Einasto' : {...}
    #  - 'NFW' : {...}
    #  },
    #  where each {...} must be an instance of results.results_Einasto_check.fit.
    #  @returns
    #  matplotlib.pyplot.figure[1,2] \n
    #  Figure of the reduced chi-square comparisons between the Einasto and NFW models (checks).
    def chi_box_Einasto_checks(self,dict_in):
        chisq={'Einasto':{},'NFW':{}}
        for key in dict_in:
            chisq[key]=np.concatenate((dict_in[key]['Vbulge_none']['Chi_sq'],dict_in[key]['Vbulge']['Chi_sq']))
            print('Reduced chi-squared (Median) - ' + str(key) + ': ' + str(np.nanmedian(chisq[key])))
            print('Reduced chi-squared (Mean) - ' + str(key) + ': ' + str(np.nanmean(chisq[key])))
        fig=plt.figure()
        axs=fig.add_axes([0,0,1,1])
        axs.boxplot((chisq['NFW'],chisq['Einasto']),showfliers=False,labels=('NFW','Einasto'),vert=False);
        axs.set_xlabel(r'$\chi^2_\nu$');
        return

    ## Define the plot of the reduced chi-square distributions for the Einasto and NFW models (checks).
    #  This defines the plot of the reduced chi-square distribution for the Einasto and NFW models (checks).
    #  @param self
    #  object pointer
    #  @param dict_in
    #  dictionary \n
    #  See checks_Einasto_NFW.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be of the form: \n
    #  dict_in = { 
    #  - 'Einasto' : {...}
    #  - 'NFW' : {...}
    #  },
    #  where each {...} must be an instance of results.results_Einasto_check.fit.
    #  @param bins (optional)
    #  int \n
    #  Number of bins for histogram plot.
    #  @returns
    #  matplotlib.pyplot.figure[1,2] \n
    #  Figure of the reduced chi-square distributions for the Einasto and NFW models (checks).
    def chi_dist_Einasto_checks(self,dict_in,bins=50):
        _,axs=plt.subplots(1,2,figsize=(20,5))
        in_tab=[]
        in_tabb=[]
        for key in dict_in:
            in_tab.append(dict_in[key]['Vbulge_none'])
            in_tabb.append(dict_in[key]['Vbulge'])
        title_tab=['Einasto','NFW']
        for i in range(len(in_tab)):
            axs[i].hist(in_tab[i]['Chi_sq'],bins,label=r'No $V_\mathrm{bulge}$');
            axs[i].hist(in_tabb[i]['Chi_sq'],bins,label=r'With $V_\mathrm{bulge}$');
            axs[i].legend()
            axs[i].set_xlabel(r'$\chi^2_\nu$')
            axs[i].set_ylabel('Number of galaxies')
            axs[i].set_title(title_tab[i])
        return

    ## Define the plot of the parameters distributions for the Einasto and NFW models (checks).
    #  This defines the plot of the parameter distributions for the Einasto and NFW models (checks).
    #  @param self
    #  object pointer
    #  @param dict_in
    #  dictionary \n
    #  See checks_Einasto_NFW.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be of the form: \n
    #  dict_in = { 
    #  - 'Einasto' : {...}
    #  - 'NFW' : {...}
    #  },
    #  where each {...} must be an instance of results.results_Einasto_check.fit.
    #  @param bins (optional)
    #  int \n
    #  Number of bins for histogram plot.
    #  @returns
    #  matplotlib.pyplot.figure[3,2] \n
    #  Figure of the parameter distributions for the Einasto and NFW models (checks).
    def params_dist_Einasto_checks(self,dict_in,bins=50):
        _,axs=plt.subplots(3,2,figsize=(30,30))
        in_tab=[]
        in_tabb=[]
        for key in dict_in:
            in_tab.append(dict_in[key]['Vbulge_none']['params'])
            in_tabb.append(dict_in[key]['Vbulge']['params'])
        title_tab=['Einasto','NFW']
        label_tab=[r'$c_{200}$',r'$V_{200} \, [\mathrm{km/s}]$',r'$\tilde{\Upsilon}_d \, [M_\odot/L_\odot]$']
        for i in range(len(in_tab)):
            for j in range(3):
                input_arr=np.asarray(in_tab[i])
                input_arrb=np.asarray(in_tabb[i])
                axs[j,i].hist(input_arr[:,j],bins,label=r'No $V_\mathrm{bulge}$');
                axs[j,i].hist(input_arrb[:,j],bins,label=r'With $V_\mathrm{bulge}$');
                axs[j,i].legend()
                axs[j,i].set_ylabel('Number of galaxies')
                axs[j,i].set_title(title_tab[i])
                axs[j,i].set_xlabel(label_tab[j])    
        return

    ## Define the plot of the halo parameters for the Einasto model (checks).
    #  This defines the plot of the halo parameters for the Einasto model (checks).
    #  @param self
    #  object pointer
    #  @param dict_in
    #  dictionary \n
    #  See checks_Einasto_NFW.ipynb for an example of formatting for the dictionary.
    #  Dictionary must be of the form: \n
    #  dict_in = { 
    #  - 'Einasto' : {...}
    #  - 'NFW' : {...}
    #  },
    #  where each {...} must be an instance of results.results_Einasto_check.fit.
    #  @param fit_dict_in (optional)
    #  constants.fitting_dict instance \n
    #  Instance of the constants.fitting_dict class.
    #  Contains the necessary parameters for fitting.
    #  @returns
    #  matplotlib.pyplot.figure[1,2] \n
    #  Figure of the halo parameters for the Einasto model (checks).
    def params_scatter_Einasto_checks(self,dict_in,fit_dict_in=fitting_dict_in):
        ein_base=cdm_funcs.base_funcs('Einasto',fit_dict_in=fit_dict_in)
        alpha_tab=np.concatenate((np.asarray(dict_in['Einasto']['Vbulge_none']['params'])[:,3],
                np.asarray(dict_in['Einasto']['Vbulge']['params'])[:,4]))
        fit_tab=np.concatenate((dict_in['Einasto']['Vbulge_none']['fit'],dict_in['Einasto']['Vbulge']['fit']))
        rhoc_tab=np.asarray([])
        rc_tab=np.asarray([])
        for i in range(len(fit_tab)):
            rhoc_tab=np.append(rhoc_tab,ein_base.rhoc(fit_tab[i].params))
            rc_tab=np.append(rc_tab,ein_base.rc(fit_tab[i].params))
        _,axs=plt.subplots(1,2,figsize=(20,5))
        axs[0].scatter(rhoc_tab,alpha_tab)
        axs[0].set_ylabel(r'$\alpha$')
        axs[1].scatter(rhoc_tab,rc_tab)
        axs[1].set_ylabel(r'$r_c \, [\mathrm{kpc}]$')
        axs[1].set_yscale('log')
        for i in range(2):
            axs[i].set_xscale('log')
            axs[i].set_xlabel(r'$\rho_c \, [M_\odot/\mathrm{kpc}^3]$');
        return