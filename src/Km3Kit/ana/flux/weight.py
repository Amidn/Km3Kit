import numpy as np
import pandas as pd
# Got from Juan PalaciosGonzalez work on GRB : https://git.km3net.de/jpalaciosgonzalez/binned-methods/-/blob/main/libraries/lib_Weights.py
# Modified to use w5 instead of weight


def atmospheric_muons_weight(df, time_window,  full_livetime_runlist, weight="w_5", verbose=True):

	if verbose: print("\n Computing atmospheric muons weight... ")
	# Here we use 'weight' inside T branch 'sum_mc_evt'
	# This is 'livetime_DAQ'/'livetime_sim' for muatm, you can check in:
	# https://git.km3net.de/common/aanet/-/blob/v2.2.13/ana/dst/summarize_mc_evts.hh#L56
	# This is the way I did with ARCA8:
	##df.loc[df.mc_pdgid==0.0, 'wmuon'] = df.loc[df.mc_pdgid==0.0, 'weight']*time_window/full_livetime_runlist
	###                 |-> i.e. is muatm
	# This is the way I do it with ARCA21 grb production:
	#df['wmuon'] = df['weight']*time_window/full_livetime_runlist
	# After conversation with Francesco (visit CPPM November 2022):
	df['wmuon'] = np.array(df[weight])*time_window/full_livetime_runlist
	#df['wmuon'] = df['wmuon'].fillna(0.0)
	if verbose: print(" Muatm wmuon computed for time window "+"{:.1f}".format(time_window)+" s and full_livetime_runlist "+"{:.1f}".format(full_livetime_runlist)+" s.")
	return df

def cosmic_neutrino_weight(df, time_window, full_livetime_runlist=1.0,  phi0=1.2*10.0**-8, gamma=-2.0, rbr=True, verbose=True):

	if verbose: print("\n Computing cosmic neutrino weight... ")
	# Here we use 'n_gen' and 'livetime_DAQ' inside T branch 'sum_mc_evt',
	# which are prepared inside the DST file to provide the correct wcosmic.
	df.loc[df.mc_pdgid!=0.0, 'wcosmic'] = df.loc[df.mc_pdgid!=0.0, 'mc_energy']
	#                  |-> i.e. is not muatm
	df['wcosmic']=np.array(df['wcosmic'])**gamma
	df['wcosmic']=np.array(df['wcosmic'])*np.array(df['w2'])*np.array(df['livetime_DAQ'])/np.array(df['n_gen'])
	df['wcosmic']=np.array(df['wcosmic'])*0.5*phi0*10**4
	if rbr:
		df['wcosmic']=np.array(df['wcosmic'])*time_window/full_livetime_runlist
	else:
		df['wcosmic']=np.array(df['wcosmic'])*time_window
	df['wcosmic']=df['wcosmic'].fillna(0.0)
	# NOTE. 'livetime_DAQ' is the correct livetime from DB (this is not the case of the offline files)
	# n_gen is not genvol.numberOfEvents of the header. Actually: header.genvol.numberOfEvents = number_runs_in_file * n_gen
	if verbose: print(" Signal wcosmic computed for phi0 "+str(phi0)+" and gamma "+str(gamma))
	return df

def atmospheric_neutrino_weight(df, label, time_window, full_livetime_runlist, verbose=True):

	if verbose: print("\n Computing atmospheric neutrino weight... ")
	# Here we also use 'n_gen' and 'livetime_DAQ' inside T branch 'sum_mc_evt',
	# as in cosmic_neutrino_weight()
	df['w3']=np.array(df[label])*np.array(df['w2'])/np.array(df['n_gen'])*np.array(df['livetime_DAQ'])
	df['w3']=np.array(df['w3'])*time_window/full_livetime_runlist
	# For label you can use 'w3_honda', 'w3_prompt' or 'w3_atmospheric'
	# I compute them in Info_extractor_from_dst_MCrbr.py
	df['w3']=df['w3'].fillna(0.0)
	if verbose: print(" Nuatm w3 computed using "+label+". ")
	return df
