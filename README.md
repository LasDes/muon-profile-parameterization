We provide a simple python script implementing the parameterization of muon production profiles from the paper "Profiles of energetic muons in the atmosphere" by Thomas K. Gaisser and Stef Verpoest submitted to Astroparticle Physics.

The function "production_profile" returns the expected production of muons above the energy threshold E_mu, differential in slanth depth X along the shower axis with zenith angle theta, for a cosmic ray with primary energy E0 and mass A. The atmospheric temperature T at depth X can be single-valued for an isothermal atmosphere, or an array of temperatures corresponding to an array of depths can be given to describe a realistic atmosphere. Parameters are given optimized for two different muon energy ranges, ~300GeV-1TeV and ~50GeV, respectively corresponding to Table 1 and 2 in the paper, and accesible in the script through the keys 'IceCube' and 'NOvA' in the parameters dictionary.

A simple usage example is given in the '\_\_main\_\_' scope of the script.

For more information, please email: Stef Verpoest (stef.verpoest@ugent.be)
