## @package model_fit
#  The package containing the general fitting procedures.

import numpy as np
from uncertainties import unumpy
import lmfit
import pyfiles.data_models.constants as constants
import pyfiles.data_models.galaxy as galaxy
import pyfiles.data_models.halo as halo

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

## The class containing the fitting procedure assuming a chosen halo model for a chosen galaxy.
class model_fit:

    ## Define the constructor of the model_fit class.
    #  This defines the constructor of the model_fit class.
    #  @param self
    #  object pointer
    #  @param halo
    #  halo.halo instance \n
    #  Instance of the halo.halo class.
    #  Assumed halo model and proper definitions contained in halo class instance.
    #  @param ULDM_fits (optional)
    #  bool \n
    #  True if performing fits for the ULDM models
    def __init__(self,halo,ULDM_fits=False):

        ## @var halo
        #  halo.halo instance \n
        #  Instance of the halo.halo class.
        #  @var args_opts
        #  dictionary \n
        #  Dictionary of rules to be assumed during fitting.
        #  Instance of constants.fitting_dict.args_opts
        #  @var ULDM_fits
        #  bool \n
        #  True if performing fits for the ULDM models
        self.halo=halo
        self.args_opts=self.halo.args_opts
        self.ULDM_fits=ULDM_fits
        
    ## Define the total circular velocity of a galaxy at some radius for the given halo model.
    #  This defines the total circular velocity of a galaxy at a given radius 
    #  assuming the given model parameters and galaxy data.
    #  The total circular velocity is given by,
    #  \f[ V(r) = \sqrt{V_{\mbox{bar}}(r)^2 + V_{\mbox{DM}}(r)^2}, \f]
    #  where,
    #  \f[ V_{\mbox{bar}}(r) = \sqrt{\left|V_{\mbox{gas}}(r)\right|V_{\mbox{gas}}(r) 
    #       + \tilde{\Upsilon}_{\mbox{disk}} \left|V_{\mbox{disk}}(r)\right|V_{\mbox{disk}}(r) 
    #       + \tilde{\Upsilon}_{\mbox{bulge}} \left|V_{\mbox{bulge}}(r)\right|V_{\mbox{bulge}}(r)}.
    #  \f]
    #  Here, \f$ V_{\mbox{DM}}(r) = \f$ halo.halo.velocity, \f$ V_{\mbox{gas}}(r) \f$, \f$ V_{\mbox{disk}}(r) \f$,
    #  and \f$ V_{\mbox{bulge}}(r) \f$ can be found in galaxy.galaxy.data.  Finally,
    #  \f$ \tilde{\Upsilon}_{\mbox{disk}} = \f$ MLD and \f$ \tilde{\Upsilon}_{\mbox{bulge}} = \f$ MLB in 
    #  cdm_params.halo.params (for CDM models) or alp_params.soliton.params (for ULDM models).
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
    #  @param data
    #  dictionary \n
    #  Dictionary containing all the necessary data for a given galaxy.
    #  This can be created using galaxy.galaxy.
    #  @returns
    #  ndarray[N] \n
    #  Numpy array of N circular velocity values in units of km/s.
    def velocity_tot(self,params,r,data):
        if self.args_opts['fitting_routine']['DC14_check']==True:
            MLd=10**(params['MLd'])
        else:
            MLd=params['MLd']
        Vgas=data['Vgas']
        Vdisk=data['Vdisk']
        Vbul=data['Vbulge']
        if (Vbul.all()==0):
            Vmod=np.sqrt(self.halo.velocity(params,r)**2+np.abs(Vgas)*Vgas+MLd*np.abs(Vdisk)*Vdisk)
        else:
            if (self.args_opts['fitting_routine']['DC14_check']==False):
                MLb=params['MLb']
                Vmod=np.sqrt(self.halo.velocity(params,r)**2+np.abs(Vgas)*Vgas
                    +MLd*np.abs(Vdisk)*Vdisk+MLb*np.abs(Vbul)*Vbul)
            else:
                Vmod=np.sqrt(self.halo.velocity(params,r)**2+np.abs(Vgas)*Vgas
                    +MLd*(np.abs(Vdisk)*Vdisk+np.abs(Vbul)*Vbul))
        return np.asarray(Vmod)

    ## Define the residual of the modeled total circular velocity from the data for a chosen galaxy.
    #  This calculates the residual of the modeled total circular velocity from the data for a galaxy 
    #  at a given radius assuming the given model parameters and galaxy data.
    #  This is used as the fitting residual in the lmfit.Minimizer class and is given by,
    #  \f[ \mbox{res} = \frac{V_{\mbox{obs}}(r) - V_{\mbox{model}}(r)}{V_{\mbox{obs,err}}(r)}, \f]
    #  where \f$ V_{\mbox{obs}}(r) \f$ and \f$ V_{\mbox{obs,err}}(r) \f$ can be found in galaxy.galaxy.data
    #  and \f$ V_{\mbox{model}}(r) = \f$ model_fit.model_fit.velocity_tot.
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
    #  @param data
    #  dictionary \n
    #  Dictionary containing all the necessary data for a given galaxy.
    #  This can be created using galaxy.galaxy.
    #  @returns
    #  ndarray[N] \n
    #  Numpy array of N residual values.
    def residual(self,params,r,data):
        vobs=data['Vobs']
        vobserr=data['eVobs']
        if self.args_opts['fitting_routine']['Einasto_check']==False:
            res=(vobs-self.velocity_tot(params,r,data))/vobserr
        else:
            Upsilon_bar=0.5
            eUpsilon=0.25*0.5
            MLd=params['MLd']
            piece0=1/len(vobs)*((MLd-Upsilon_bar)/eUpsilon)**2
            piece1=((vobs-self.velocity_tot(params,r,data))/vobserr)**2
            res=np.sqrt(piece0+piece1)
        return np.asarray(res)

    ## Define the fit result for the modeled total circular velocity for a galaxy.
    #  This defines the fit result for the modeled total circular velocity assuming the given model parameters
    #  and galaxy data.
    #  @param self
    #  object pointer
    #  @param params
    #  lmfit.Parameters instance \n
    #  Instance of the lmfit.Parameters class.
    #  All model parameters contained here.
    #  This can be created using cdm_params.halo.params (for CDM models) or alp_params.soliton.params (for ULDM models)
    #  @param data
    #  dictionary \n
    #  Dictionary containing all the necessary data for a given galaxy.
    #  This can be created using galaxy.galaxy.
    #  @param calc_covar_in (optional)
    #  bool \n
    #  True(False) if covariance matrix is to be(is not to be) calculated in fitting procedure
    #  @returns 
    #  lmfit.Minimizer.minimize fit result \n
    #  Fit result contains the optimized parameters and several goodness-of-fit statistics.
    def fit(self,params,data,calc_covar_in=True):
        r=data['Radius']
        def fit_init(params_in):
            if self.ULDM_fits==False:
                method='nelder'
            else:
                method='leastsq'
            mini=lmfit.Minimizer(self.residual,params_in,fcn_args=(r,data),nan_policy='propagate',calc_covar=False)
            out=mini.minimize(method)
            mini=lmfit.Minimizer(self.residual,out.params,fcn_args=(r,data),nan_policy='propagate',calc_covar=calc_covar_in)
            out=mini.minimize(method)
            return out
        out_fin=fit_init(params)
        def fit_mid(out_fin_in):
            if self.ULDM_fits==True and self.halo.model!='Einasto':
                if (np.isnan(out_fin_in.redchi)==True or out_fin_in.redchi>10):
                    if self.halo.model=='psi_single':
                        params['Msol'].value=1e5
                    else:
                        params['Msol'].value=1e5
                        params['Msol_2'].value=1e5
                    out1=fit_init(params)
                    if (np.isnan(out1.redchi)!=True and out1.redchi<=10):
                        out_fin_in=out1
                    else:
                        if self.halo.model=='psi_single':
                            params['Msol'].value=1e6
                        else:
                            params['Msol'].value=1e6
                            params['Msol_2'].value=1e6
                        out2=fit_init(params)
                        if (np.isnan(out2.redchi)!=True and out2.redchi<=10):
                            out_fin_in=out2  
                        else:
                            if self.halo.model=='psi_single':
                                params['Msol'].value=1e7
                            else:
                                params['Msol'].value=1e7
                                params['Msol_2'].value=1e7
                            out3=fit_init(params)
                            if (np.isnan(out3.redchi)!=True and out3.redchi<=10):
                                out_fin_in=out3
                            else:
                                if self.halo.model=='psi_single':
                                    params['Msol'].value=1e8
                                else:
                                    params['Msol'].value=1e8
                                    params['Msol_2'].value=1e8
                                out4=fit_init(params)
                                if (np.isnan(out4.redchi)!=True and out4.redchi<=10):
                                    out_fin_in=out4
                                else:
                                    if self.halo.model=='psi_single':
                                        params['Msol'].value=1e10
                                    else:
                                        params['Msol'].value=1e10
                                        params['Msol_2'].value=1e10
                                    out5=fit_init(params)
                                    if (np.isnan(out5.redchi)!=True and out5.redchi<=10):
                                        out_fin_in=out5
                                    else:
                                        if self.halo.model=='psi_single':
                                            params['Msol'].value=1e11
                                        else:
                                            params['Msol'].value=1e11
                                            params['Msol_2'].value=1e11
                                        out6=fit_init(params)
                                        outtab=np.asarray([out_fin_in,out1,out2,out3,out4,out5,out6])
                                        redchimin=np.nanmin((out_fin_in.redchi,out1.redchi,out2.redchi,out3.redchi,out4.redchi,out5.redchi,out6.redchi))
                                        for key in outtab:
                                            if key.redchi==redchimin:
                                                out_fin_in=key  
            return out_fin_in                                    
        out_fin=fit_mid(out_fin)
        return out_fin

    ## Define the BIC value for a given fit.
    #  This defines the BIC value for a given fit.
    #  @param self
    #  object pointer
    #  @param fit
    #  model_fit.model_fit.fit instance \n
    #  Instance of the model_fit.model_fit.fit function.
    #  @returns 
    #  float \n
    #  BIC value for a given fit.
    def bic(self,fit):
        bicin=fit.chisqr+np.log(fit.ndata)*fit.nvarys
        return bicin

    ## Define the plot of the model of the total circular velocity for a galaxy.
    #  This defines the plot of the model of total circular velocity for a galaxy assuming the best fit parameters.
    #  @param self
    #  object pointer
    #  @param halo
    #  halo.halo instance \n
    #  Instance of the halo.halo class.
    #  Assumed halo model and proper definitions contained in halo class instance.
    #  @param gal
    #  galaxy.galaxy instance \n
    #  Instance of the galaxy.galaxy class.
    #  Contains all necessary galaxy data.
    #  @param axs
    #  matplotlib axis \n
    #  Axis in which matplotlib plot will be contained
    #  @returns
    #  matplotlib plot \n
    #  Plot of the circular velocity of the different galaxy contributions 
    #  including the DM halo given the assumed model and best fit parameters.
    def plot(self,halo,gal,axs):
        name=gal.name
        data=gal.data
        ind_sort=np.argsort(data['Radius'])
        r=data['Radius'][ind_sort]
        vobs=data['Vobs'][ind_sort]
        evobs=data['eVobs'][ind_sort]
        vgas=data['Vgas'][ind_sort]
        vdisk=data['Vdisk'][ind_sort]
        vbul=data['Vbulge'][ind_sort]
        params=halo.params
        fit=self.fit(params,data)
        if self.args_opts['fitting_routine']['DC14_check']==True:
            MLd=10**(fit.params['MLd'])
        else:
            MLd=fit.params['MLd'].value
        if (vbul.all()==0):
            vstar=np.sqrt(MLd*np.abs(vdisk)*vdisk)
        else:
            if (self.args_opts['fitting_routine']['DC14_check']==False):
                MLb=fit.params['MLb'].value
                vstar=np.sqrt(MLd*np.abs(vdisk)*vdisk+MLb*np.abs(vbul)*vbul)
            else:
                vstar=np.sqrt(MLd*(np.abs(vdisk)*vdisk+np.abs(vbul)*vbul))
        bf=self.halo.velocity(fit.params,r)
        fit_tot=self.velocity_tot(fit.params,r,data)
        title_font = {'size':'30','color':'black','weight':'normal'}
        labs_font = {'size':'26','color':'black','weight':'normal'}
        axs.errorbar(r,vobs,yerr=evobs,xerr=None, fmt='ko',label='$V_{\mathrm{obs}}$');
        axs.plot(r,vgas,'g',linestyle='-',label='$V_{\mathrm{gas}}$')
        axs.plot(r,vstar,'m',linestyle='-',label='$V_{\mathrm{star}}$')
        axs.plot(r,bf,'k',label='$V_{\mathrm{DM}}$')
        axs.plot(r,fit_tot,'r',label='$V_{\mathrm{tot}}$')
        if (halo.model=='psi_single'):
            str_in='Single'
        elif (halo.model=='psi_multi'):
            str_in='Double'
        else:
            str_in=str(halo.model)
        axs.set_title('Galaxy: '+str(name)+': '+str_in,**title_font)
        axs.set_xlabel('Radius [kpc]',**labs_font)
        axs.set_ylabel(r'Circular velocity [km/s]',**labs_font)
        axs.legend(fontsize=22,loc='lower right')
        if self.args_opts['fitting_routine']['DC14_check']==False:
            if self.halo.fit_dict_in.sol_mass_ind==0:
                textstr = '\n'.join((
                    r'$\chi^2_\nu=$' + str(round(fit.redchi,2)),
                    r'$c_{200}=$' + str(round(fit.params['c200'].value,2)),
                    r'$V_{200}=$' + str(round(fit.params['v200'].value,2)),
                    r'$\tilde{\Upsilon}_d=$' + str(round(fit.params['MLd'].value,2))
                    ))
                if (vbul.any()!=0):
                    textstr = '\n'.join((
                        textstr,
                        r'$\tilde{\Upsilon}_b=$' + str(round(fit.params['MLb'].value,2))
                        ))
                else:
                    pass
                if (halo.model=='Einasto' or 
                    ((halo.model=='psi_single' or halo.model=='psi_multi') and halo.fit_dict_in.sol_cdm_halo=='Einasto')):
                    textstr = '\n'.join((
                        textstr,
                        r'$\alpha=$' + str(round(np.real(fit.params['alpha'].value),2))
                    ))
                else:
                    pass
                if (halo.model=='psi_single' or halo.model=='psi_multi'):
                    textstr = '\n'.join((
                        textstr,
                        r'$\log_{10} \, m=$' + str(round(np.log10(fit.params['m22'].value),2)),
                        r'$\log_{10} \, M_{\mathrm{sol}}=$' + str(round(np.log10(fit.params['Msol'].value),2))
                        ))
            else:
                textstr = '\n'.join((
                    r'$\chi^2_\nu=$' + str(round(fit.redchi,2)),
                    r'$c_{200,1}=$' + str(round(fit.params['c200'].value,2)),
                    r'$V_{200,1}=$' + str(round(fit.params['v200'].value,2)),
                    r'$c_{200,2}=$' + str(round(fit.params['c200_2'].value,2)),
                    r'$V_{200,2}=$' + str(round(fit.params['v200_2'].value,2)),
                    r'$\tilde{\Upsilon}_d=$' + str(round(fit.params['MLd'].value,2))
                    ))
                if (vbul.any()!=0):
                    textstr = '\n'.join((
                        textstr,
                        r'$\tilde{\Upsilon}_b=$' + str(round(fit.params['MLb'].value,2))
                        ))
                else:
                    pass
                if (halo.model=='Einasto' or 
                    ((halo.model=='psi_single' or halo.model=='psi_multi') and halo.fit_dict_in.sol_cdm_halo=='Einasto')):
                    textstr = '\n'.join((
                        textstr,
                        r'$\alpha_1=$' + str(round(np.real(fit.params['alpha'].value),2)),
                        r'$\alpha_2=$' + str(round(np.real(fit.params['alpha_2'].value),2))
                    ))
                else:
                    pass
                if (halo.model=='psi_multi'):
                    textstr = '\n'.join((
                        textstr,
                        r'$\log_{10} \, m_1=$' + str(round(np.log10(fit.params['m22'].value),2)),
                        r'$\log_{10} \, m_2=$' + str(round(np.log10(fit.params['m22_2'].value),2)),
                        r'$\log_{10} \, M_{\mathrm{sol,1}}=$' + str(round(np.log10(fit.params['Msol'].value),2)),
                        r'$\log_{10} \, M_{\mathrm{sol,2}}=$' + str(round(np.log10(fit.params['Msol_2'].value),2))
                        ))
        else:
            textstr = '\n'.join((
                r'$\chi^2_\nu=$' + str(round(fit.redchi,2)),
                r'$c_{200}=$' + str(round(10**(fit.params['c200'].value),2)),
                r'$V_{200}=$' + str(round(10**(fit.params['v200'].value),2)),
                r'$\tilde{\Upsilon}_d=$' + str(round(10**(fit.params['MLd'].value),2))
                ))
        axs.text(0.5,0.05,textstr,transform=axs.transAxes,fontsize=22,verticalalignment='bottom',
            bbox=dict(facecolor='yellow',alpha=0.8))
        return

## The class containing the fitting procedure for the gravitational acceleration relation.
#  See See Stacy S. McGaugh, Federico Lelli, and James M. Schombert Phys. Rev. Lett. 117, 201101 : 
#  https://doi.org/10.1103/PhysRevLett.117.201101 for more information
class grar_fit:

    ## Define the constructor of the grar_fit class.
    #  This defines the constructor of the grar_fit class.
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
    #  @param ULDM_fits (optional)
    #  bool \n
    #  True if performing fits for the ULDM models
    #  @param fit_dict_in (optional)
    #  constants.fitting_dict instance \n
    #  Instance of the constants.fitting_dict class.
    #  Contains all the necessary rules and values to be used in fitting.
    def __init__(self,model,ULDM_fits=False,fit_dict_in=fitting_dict_in):

        ## @var model
        #  str \n
        #  Model to assume for DM halo.
        #  @var ULDM_fits
        #  bool \n
        #  True if performing fits for the ULDM models
        #  @var fit_dict_in
        #  constants.fitting_dict instance \n
        #  Instance of the constants.fitting_dict class.
        #  @var args_opts
        #  dictionary \n
        #  Dictionary of rules to be assumed during fitting
        #  Instance of constants.fitting_dict.args_opts
        self.model=model
        self.fit_dict_in=fit_dict_in
        self.args_opts=self.fit_dict_in.args_opts()
        self.ULDM_fits=ULDM_fits

    ## Define the total radial gravitational acceleration of the galaxy.
    #  This defines the total radial gravitational acceleration of the galaxy 
    #  assuming the given model parameters and galaxy data.
    #  The total radial gravitational acceleration is given by,
    #  \f[ g_{\mbox{tot}}(r) = V_{\mbox{tot}}(r)^2/r, \f]
    #  where \f$ V_{\mbox{tot}}(r) = \f$ model_fit.model_fit.velocity_tot.
    #  @param self
    #  object pointer
    #  @param params
    #  lmfit.Parameters instance \n
    #  Instance of the lmfit.Parameters class.
    #  All model parameters contained here.  
    #  This can be created using cdm_params.halo.params (for CDM models) or alp_params.soliton.params (for ULDM models)
    #  @param data
    #  dictionary \n
    #  Dictionary containing all the necessary data for a given galaxy.
    #  This can be created using galaxy.galaxy.
    #  @returns
    #  ndarray[N] \n
    #  Numpy array of N values of the total radial gravitational acceleration in units of \f$\mbox{m}\,\mbox{s}^{-2}\f$.
    def g_tot(self,params,data):
        MLd=params['MLd']
        Vgas=data['Vgas']
        Vdisk=data['Vdisk']
        Vbul=data['Vbulge']
        r=data['Radius']
        halo_init=halo.halo(self.model,data,fit_dict_in=self.fit_dict_in)
        if (Vbul.all()==0):
            Vmod=np.sqrt(halo_init.velocity(params,r)**2+np.abs(Vgas)*Vgas+MLd*np.abs(Vdisk)*Vdisk)
            rad=r*kpcTOGeV/kmTOGeV
            gmod=Vmod**2/rad*1e3
        else:
            MLb=params['MLb']
            Vmod=np.sqrt(halo_init.velocity(params,r)**2+np.abs(Vgas)*Vgas+MLd*np.abs(Vdisk)*Vdisk+MLb*np.abs(Vbul)*Vbul)
            rad=r*kpcTOGeV/kmTOGeV
            gmod=Vmod**2/rad*1e3
        return gmod
    
    ## Define the radial gravitational acceleration due to baryonic matter.
    #  This defines the radial gravitational acceleration due to baryonic matter 
    #  assuming the given model parameters and galaxy data.
    #  The radial gravitational acceleration due to baryonic matter is given by,
    #  \f[ g_{\mbox{bar}}(r) = V_{\mbox{bar}}(r)^2/r, \f]
    #  where,
    #  \f[ V_{\mbox{bar}}(r) = \sqrt{\left|V_{\mbox{gas}}(r)\right|V_{\mbox{gas}}(r) 
    #       + \tilde{\Upsilon}_{\mbox{disk}} \left|V_{\mbox{disk}}(r)\right|V_{\mbox{disk}}(r) 
    #       + \tilde{\Upsilon}_{\mbox{bulge}} \left|V_{\mbox{bulge}}(r)\right|V_{\mbox{bulge}}(r)}. 
    #  \f]
    #  Here, \f$ V_{\mbox{gas}}(r) \f$, \f$ V_{\mbox{disk}}(r) \f$, and \f$ V_{\mbox{bulge}}(r) \f$ 
    #  can be found in galaxy.galaxy.data.  Finally,
    #  \f$ \tilde{\Upsilon}_{\mbox{disk}} = \f$ MLD and \f$ \tilde{\Upsilon}_{\mbox{bulge}} = \f$ MLB in 
    #  cdm_params.halo.params (for CDM models) or alp_params.soliton.params (for ULDM models).
    #  @param self
    #  object pointer
    #  @param params
    #  lmfit.Parameters instance \n
    #  Instance of the lmfit.Parameters class.
    #  All model parameters contained here.  
    #  This can be created using cdm_params.halo.params (for CDM models) or alp_params.soliton.params (for ULDM models)
    #  @param data
    #  dictionary \n
    #  Dictionary containing all the necessary data for a given galaxy.
    #  This can be created using galaxy.galaxy.
    #  @returns
    #  ndarray[N] \n
    #  Numpy array of N values of the radial gravitational acceleration due to baryons in units of
    #  \f$\mbox{m}\,\mbox{s}^{-2}\f$.
    def g_bar(self,params,data):
        MLd=params['MLd']
        Vgas=data['Vgas']
        Vdisk=data['Vdisk']
        Vbul=data['Vbulge']
        r=data['Radius']
        if (Vbul.all()==0):
            Vmod=np.sqrt(np.abs(Vgas)*Vgas+MLd*np.abs(Vdisk)*Vdisk)
            rad=r*kpcTOGeV/kmTOGeV
            gmod=Vmod**2/rad*1e3
        else:
            MLb=params['MLb']
            Vmod=np.sqrt(np.abs(Vgas)*Vgas+MLd*np.abs(Vdisk)*Vdisk+MLb*np.abs(Vbul)*Vbul)
            rad=r*kpcTOGeV/kmTOGeV
            gmod=Vmod**2/rad*1e3
        return gmod

    ## Define the radial gravitational acceleration relation.
    #  This defines the radial gravitational acceleration relation 
    #  assuming the given relation constant, gdag, model parameters and galaxy data.
    #  The radial gravitational acceleration relation is given by,
    #  \f[ g_{\mbox{tot}}(r) = \frac{g_{\mbox{bar}}(r)}{1 - \exp\left( -\sqrt{g_{\mbox{bar}}(r)/g_\dagger} \right)}, \f]
    #  where \f$ g_{\mbox{bar}}(r) = \f$ model_fit.grar_fit.g_bar and \f$ g_\dagger \f$ is some constant.
    #  @param self
    #  object pointer
    #  @param gdag
    #  float \n
    #  Constant in the gravitational acceleration relation.
    #  Must be in units of \f$\mbox{m}\,\mbox{s}^{-2}\f$.
    #  @param params
    #  lmfit.Parameters instance \n
    #  Instance of the lmfit.Parameters class.
    #  All model parameters contained here.  
    #  This can be created using cdm_params.halo.params (for CDM models) or alp_params.soliton.params (for ULDM models)
    #  @param data
    #  dictionary \n
    #  Dictionary containing all the necessary data for a given galaxy.
    #  This can be created using galaxy.galaxy.
    #  @returns
    #  ndarray[N] \n
    #  Numpy array of N values of the total radial gravitational acceleration relation
    #  in units of \f$\mbox{m}\,\mbox{s}^{-2}\f$.
    def g_rar(self,gdag,params,data):
        gmod=self.g_bar(params,data)/(1-np.exp(-np.sqrt(self.g_bar(params,data)/gdag)))
        return gmod

    ## Define the general model of the radial gravitational acceleration relation.
    #  This defines the general model of the radial gravitational acceleration relation
    #  for the model parameters, gbarin and gdag.
    #  This is the form of the radial gravitational acceleration relation to be used in fitting and is given by,
    #  \f[ g_{\mbox{tot}}(r) = \frac{g_{\mbox{bar}}(r)}{1-\exp\left( -\sqrt{g_{\mbox{bar}}(r)/g_\dagger} \right)}. \f]
    #  Here, the fit parameter \f$ g_\dagger \f$ can be found by fitting this model using
    #  \f$ g_{\mbox{tot}}(r) = \f$ model_fit.grar_fit.g_tot and \f$ g_{\mbox{bar}} = \f$ model_fit.grar_fit.g_bar.
    #  @param self
    #  object pointer
    #  @param gbarin
    #  ndarray[N] \n
    #  Numpy array of N values of baryonic gravitational accerlation
    #  in units of \f$\mbox{m}\,\mbox{s}^{-2}\f$.
    #  @param gdag
    #  ndarray[N] \n
    #  Constant in the gravitational acceleration relation
    #  in units of \f$\mbox{m}\,\mbox{s}^{-2}\f$.      
    #  @returns
    #  ndarray[N] \n
    #   Numpy array of N values of the total modeled radial gravitational acceleration
    #  in units of \f$\mbox{m}\,\mbox{s}^{-2}\f$.
    def grar_model(self,gbarin,gdag):
        res=gbarin/(1-np.exp(-np.sqrt(gbarin/gdag)))
        return res   

    ## Define the fit result for the modeled gravitational acceleration relation.
    #  This defines the fit result for the modeled gravitational acceleration relation.
    #  @param self
    #  object pointer
    #  @returns
    #  lmfit.Minimizer.minimize fit result \n
    #  Fit results contain the optimized parameters and several goodness-of-fit statistics.
    def fit(self):
        results_all={'gbar':np.asarray([]),'gtot':np.asarray([]),'gMOND':np.asarray([]),'fit':0}
        fit_params_in=lmfit.Parameters()
        gdag_MOND=standard_consts_in['MOND_gdag']
        fit_params_in.add(name='gdag',value=gdag_MOND,vary=False)
        gal=galaxy.galaxy('random')
        for key in gal.df1['galaxy']:
            gal=galaxy.galaxy(key)
            data=gal.data
            if (data['Quality']!=3 and data['Inclination']>30 and data['Vf']!=0):
                if self.ULDM_fits==False:
                    halo_init=halo.halo(self.model,data,fit_dict_in=self.fit_dict_in)
                    params=halo_init.params
                    fit_init=model_fit(halo_init)
                    fit=fit_init.fit(params,data,calc_covar_in=False)
                    gbar_in=self.g_bar(fit.params,gal.data)
                    gtot_in=self.g_tot(fit.params,gal.data)
                    grar_in=self.g_rar(fit_params_in,fit.params,gal.data)
                    for k in range(len(gbar_in)):
                        if (np.isnan(gbar_in[k])==False and np.isnan(gtot_in[k])==False and np.isnan(grar_in[k])==False):
                            results_all['gbar']=np.append(results_all['gbar'],gbar_in[k])
                            results_all['gtot']=np.append(results_all['gtot'],gtot_in[k])
                            results_all['gMOND']=np.append(results_all['gMOND'],grar_in[k])
                else:
                    if data['Vbulge'].all()==0:
                        if (len(data['Vobs'])-11)>0:
                            halo_init=halo.halo(self.model,data,fit_dict_in=self.fit_dict_in)
                            params=halo_init.params
                            fit_init=model_fit(halo_init)
                            fit=fit_init.fit(params,data,calc_covar_in=False)
                            gbar_in=self.g_bar(fit.params,gal.data)
                            gtot_in=self.g_tot(fit.params,gal.data)
                            grar_in=self.g_rar(fit_params_in,fit.params,gal.data)
                            for k in range(len(gbar_in)):
                                if (np.isnan(gbar_in[k])==False and np.isnan(gtot_in[k])==False and np.isnan(grar_in[k])==False):
                                    results_all['gbar']=np.append(results_all['gbar'],gbar_in[k])
                                    results_all['gtot']=np.append(results_all['gtot'],gtot_in[k])
                                    results_all['gMOND']=np.append(results_all['gMOND'],grar_in[k])
                    else:
                        if (len(data['Vobs'])-12)>0:
                            halo_init=halo.halo(self.model,data,fit_dict_in=self.fit_dict_in)
                            params=halo_init.params
                            fit_init=model_fit(halo_init)
                            fit=fit_init.fit(params,data,calc_covar_in=False)
                            gbar_in=self.g_bar(fit.params,gal.data)
                            gtot_in=self.g_tot(fit.params,gal.data)
                            grar_in=self.g_rar(fit_params_in,fit.params,gal.data)
                            for k in range(len(gbar_in)):
                                if (np.isnan(gbar_in[k])==False and np.isnan(gtot_in[k])==False and np.isnan(grar_in[k])==False):
                                    results_all['gbar']=np.append(results_all['gbar'],gbar_in[k])
                                    results_all['gtot']=np.append(results_all['gtot'],gtot_in[k])
                                    results_all['gMOND']=np.append(results_all['gMOND'],grar_in[k])

        def res_g(pars,gtotin,gbarin):
            gdag=pars['gdag']
            res=gtotin-self.grar_model(gbarin,gdag)
            return res
        fit_params=lmfit.Parameters()
        fit_params.add('gdag',gdag_MOND,min=0)
        mini=lmfit.Minimizer(res_g,fit_params,fcn_args=(results_all['gtot'],results_all['gbar']),nan_policy='propagate')
        if self.ULDM_fits==False:
            method='nelder'
        else:
            method='nelder'
        out=mini.minimize(method)
        mini=lmfit.Minimizer(res_g,out.params,fcn_args=(results_all['gtot'],results_all['gbar']),nan_policy='propagate')
        results_all['fit']=mini.minimize(method)
        return results_all

## Define the concentration-mass relation (CMR).
#  This defines the CMR for a given halo mass.
#  See Monthly Notices of the Royal Astronomical Society, Volume 441, Issue 4, 11 July 2014, Pages 3359–3374 : 
#  https://doi.org/10.1093/mnras/stu742 for more information.
#  The CMR is given by, 
#  \f[ \log_{10}c_{200} = 0.905-0.101 \log_{10}\left(\frac{M_{200}}{10^{12} h^{-1} \, M_\odot} \right), \f]
#  where \f$ h^{-1} = 0.73 \f$ and \f$ M_\odot \f$ is the mass of sun.
#  @param m200
#  float \n
#  Total mass of the DM halo in units of solar masses.
#  @returns
#  float \n
#  Concentration parameter assuming the CMR.
def conc_mass_rel_Du(m200):
    c200_in=0.905-0.101*unumpy.log10(0.6777*m200/10**12)
    return c200_in

## Define the concentration-mass relation (CMR).
#  This defines the CMR for a given halo mass.
#  See Nature volume 585, pages 39–42 (2020) : 
#  https://doi.org/10.1038/s41586-020-2642-9 for more information.
#  The CMR is given by, 
#  \f[ c_{200} = \sum_{i=0}^5 c_i\ln\left(\frac{M_{200}}{h^{-1} \, M_\odot} \right)^i, \f]
#  where \f$ h^{-1} = 0.6777 \f$, \f$ M_\odot \f$ is the mass of sun,
#  \f$c_0=27.112\f$, \f$c_1=-0.381\f$, \f$c_2=-1.853\times 10^{-3}\f$,
#  \f$c_3=-4.141\times 10^{-4}\f$, and \f$c_5=3.208\times 10^{-7}\f$.
#  @param m200
#  float \n
#  Total mass of the DM halo in units of solar masses.
#  @returns
#  float \n
#  Concentration parameter assuming the CMR.
def conc_mass_rel_Wa(m200):
    c0=27.112
    c1=-0.381
    c2=-1.853*1e-3
    c3=-4.141*1e-4
    c4=-4.334*1e-6
    c5=3.208*1e-7
    ctab=np.asarray([c0,c1,c2,c3,c4,c5])
    mtab=np.asarray([])
    for i in range(6):
        mtab=np.append(mtab,unumpy.log(0.6777*m200)**i)
    c200_in=np.sum(ctab*mtab)
    return c200_in

## Define the baryonic Tully-Fisher relation (BTFR).
#  This defines the BTFR for a given maximum circular velocity of a galaxy.
#  See Federico Lelli et al 2016 ApJL 816 L14 : 
#  https://doi.org/10.3847/2041-8205/816/1/l14 for more information.
#  The BTFR is given by, 
#  \f[ \log_{10}\frac{M_b}{M_\odot} = s \, \log_{10}\left( \frac{V_f}{\mbox{km}\,\mbox{s}^{-1}} \right) + \log_{10} A, \f]
#  where \f$ M_\odot \f$ is the mass of sun, \f$ s = 3.71 \pm 0.08 \f$, and \f$ \log_{10} A = 2.27 \pm 0.18 \f$.
#  @param Vf
#  float \n
#  Maximum circular velocity in units of \f$ \mbox{km}\,\mbox{s}^{-1} \f$.
#  @returns
#  float \n
#  Total baryonic mass of a galaxy in units of solar masses.
def BTFR(Vf):
    s=3.71
    a=2.27
    Mb_in=s*np.log10(Vf)+a
    return Mb_in

## Define the abundance matching relation (AMR).
#  This defines the AMR for a given halo mass.
#  See Peter S. Behroozi et al 2013 ApJ 770 57 : 
#  https://doi.org/10.1088/0004-637x/770/1/57 and
#  Peter S. Behroozi et al 2013 ApJL 762 L31 : 
#  https://doi.org/10.1088/2041-8205/762/2/l31 for more information.
#  The AMR is given by,
#  \f[ \frac{M_*}{M_{200}} = 2 N \left[ \left( \frac{M_{200}}{M_1 \, M_\odot} \right)^{-\beta} 
#  + \left( \frac{M_{200}}{M_1 \, M_\odot} \right)^{-\gamma} \right]^{-1}, \f]
#  where \f$ N=0.0351 \f$, \f$ \beta = 1.375 \f$,
#  \f$ \gamma = 0.608 \f$, and \f$ \log_{10}(M_1) = 11.59 \f$.
#  @param m200
#  float \n
#  Total mass of the DM halo in units of solar masses.
#  @returns
#  float \n
#  Total stellar mass of a galaxy in units of solar masses.
def abund_match_rel(m200):
    nin=0.0351
    beta=1.376
    gamma=0.608
    m1=10**(11.59)
    mstarin=2*m200*nin*((m200/m1)**(-beta)+(m200/m1)**(-gamma))**(-1)
    return mstarin