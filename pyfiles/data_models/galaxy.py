## @package galaxy
#  The package containing information about all SPARC catalog galaxies. \n
#  All data has been taken from SPARC database : http://astroweb.case.edu/SPARC/ \n
#  Data files can be found in:
#  - DM_halos_models/pyfiles/data/MassModels_Lelli2016c.txt
#  - DM_halos_models/pyfiles/data/SPARC_Lelli2016c.txt

import pandas as pd
import random

## The class containing data for all SPARC catalog galaxies.
class galaxy:

    ## Define the constructor of the galaxy class.
    #  This defines the constructor of the galaxy class.
    #  @param self
    #  object pointer
    #  @param name
    #  str \n
    #  Can be equal to :
    #  - Any galaxy name in the SPARC catalog (e.g. 'CamB') \n
    #    The galaxy with galaxy name will be analyzed.
    #  - 'random' \n
    #    A random galaxy in the SPARC catalog will be analyzed.
    def __init__(self,name):

        ## @var name
        #  str \n
        #  Galaxy name
        #  @var df
        #  Pandas dataframe \n
        #  Dataframe that contains data from SPARC catalog. 
        #  Contains data from DM_halo_models/pyfiles/data/MassModels_Lelli2016c.txt
        #  @var df1
        #  Pandas dataframe \n
        #  Dataframe that contains data from SPARC catalog. 
        #  Contains data from DM_halo_models/pyfiles/data/SPARC_Lelli2016c.txt
        #  @var gal_init
        #  Pandas dataframe \n
        #  Dataframe that contains data for an initialized galaxy
        #  @var data
        #  dictionary
        #  Dictionary that contains data for the chosen galaxy (i.e. either 'name' or 'random') :
        #  - 'Radius' : ndarray[N] \n
        #    Numpy array of galactocentric radii 
        #    in units of \f$[\mbox{kpc}]\f$
        #  - 'Vobs' : ndarray[N] \n 
        #    Numpy array of observed circular velocities at given radii 
        #    in units of \f$[\mbox{km}\,\mbox{s}^{-1}]\f$
        #  - 'eVobs' : ndarray[N] \n 
        #    Numpy array of error in observed circular velocities 
        #    in units of \f$[\mbox{km}\,\mbox{s}^{-1}]\f$
        #  - 'Vgas' : ndarray[N] \n 
        #    Numpy array of contribution from gas to circular velocities 
        #    in units of \f$[\mbox{km}\,\mbox{s}^{-1}]\f$
        #  - 'Vdisk' : ndarray[N] \n
        #    Numpy array of contribution from the disk to circular velocities 
        #    in units of \f$[\mbox{km}\,\mbox{s}^{-1}]\f$
        #  - 'Vbulge' : ndarray[N] \n
        #    Numpy array of contribution from the bulge to circular velocities 
        #    in units of \f$[\mbox{km}\,\mbox{s}^{-1}]\f$
        #  - 'Luminosity' : float \n
        #    Total luminosity at \f$[3.6 \, \mu \mbox{m}]\f$ 
        #    in units of \f$[10^9 \, L_{\odot}]\f$
        #  - 'Mgas': float \n
        #    Total mass of HI gas 
        #    in units of \f$[10^9 \, M_{\odot}]\f$
        #  - 'Vf' : float \n 
        #    Maximum circular velocity 
        #    in units of \f$[\mbox{km}\,\mbox{s}^{-1}]\f$
        #  - 'Inclination' : float \n
        #    Inclination of galaxy
        #    in units of \f$[\mbox{degrees}]\f$
        #  - 'Quality' : int \n
        #    Quality flag of galaxy
        self.name=name
        self.df=pd.read_fwf('pyfiles/data/MassModels_Lelli2016c.txt',
                            skiprows=25, 
                            header=None, 
                            infer_nrows=500,
                            usecols=[0,2,3,4,5,6,7],
                            names=["galaxy","radius","vobs","evobs","vgas","vdisk","vbul"])
        self.df1=pd.read_fwf('pyfiles/data/SPARC_Lelli2016c.txt',
                             skiprows=98, 
                             header=None, 
                             infer_nrows=500,
                             usecols=[0,5,7,13,15,17],
                             names=["galaxy","inclination","luminosity","mgas","vf","quality"])
        if (self.name=='random'):
            self.gal_init=self.df.groupby('galaxy').get_group(random.sample(list(self.df.groupby('galaxy').indices),1)[0])
        else:
            self.gal_init=self.df.groupby('galaxy').get_group(self.name)
            
        self.name=self.gal_init.apply(lambda x: x.value_counts().index[0])['galaxy']
        mgas=self.df1.groupby('galaxy').get_group(self.name)['mgas']
        lum=self.df1.groupby('galaxy').get_group(self.name)['luminosity']
        vf=self.df1.groupby('galaxy').get_group(self.name)['vf']
        inclin=self.df1.groupby('galaxy').get_group(self.name)['inclination']
        quality=self.df1.groupby('galaxy').get_group(self.name)['quality']
        self.data={'Radius':self.gal_init['radius'].to_numpy(),
                    'Vobs':self.gal_init['vobs'].to_numpy(),
                    'eVobs':self.gal_init['evobs'].to_numpy(),
                    'Vgas':self.gal_init['vgas'].to_numpy(),
                    'Vdisk':self.gal_init['vdisk'].to_numpy(),
                    'Vbulge':self.gal_init['vbul'].to_numpy(),
                    'Luminosity':lum.to_numpy()[0],
                    'Mgas':mgas.to_numpy()[0],
                    'Vf':vf.to_numpy()[0],
                    'Inclination':inclin.to_numpy()[0],
                    'Quality':quality.to_numpy()[0]}