import sys
assert('neuron' not in sys.modules)
import os
nrn_options = "-nogui -NSTACK 100000 -NFRAME 20000"
os.environ["NEURON_MODULE_OPTIONS"] = nrn_options
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.animation as animation
import time
from neuron_model_serial import NeuronSim
from elec_field import ICMS
from pulse_train import PulseTrain_square
import math
from helper import fibonacci_sphere, plot_electrode_and_neuron

import os, psutil

def pointElec_simulation(num_electrode, amplitude, pulse_width, period, total_time, cell_type, plot_neuron_with_electrodes):    
    process = psutil.Process(os.getpid())
    ##################################################################################
    ################## ICMS Monopolar Experimental Setup #############################
    ##################################################################################
    #SEED = 1234 
    #np.random.seed(SEED)
    #print("Setting Random Seed as %s"%(str(round(SEED,3))))
    cwd = os.getcwd()
    #### Defining Variables for Setting up Simulation
    cell_id_pyr_lst = [6,7,8,9,10] ## Different Morphology for L23 Pyr Cells
    cell_id_pv_lst = [32,33,34,35,36] ## Different Morphology for L23 LBC Cells
    cell_id_pyr = 6 ## Default Pyr Cell ID
    cell_id_pv = 36 ## Default PV Cell ID
    human_or_mice = 0 ## 1->mice, 0-> human
    temp = 37 ## Celsius, temparature at which neurons are simulated
    dt = 0.025 ## ms, discretization time step
    dist = 1 ## mm, distance from the origin for the ICMS electrode
    elec_location_ICMS = fibonacci_sphere(num_electrode) ## Sampling approximately uniformly spaced electrode locations from a unit sphere
    elec_location_ICMS = elec_location_ICMS*dist ## Scaling the radius of sphere to the dist variable
    
    angle_pv = np.array([0,0]) ## parameter used for specifying rotation of PV morphology
    angle_pyr = np.array([0,0]) ## parameter used for specifying rotation of Pyr morphology

    loc_pyr = np.array([0,0,0]) ## parameter used for specifying location of Pyr morphology
    loc_pv = np.array([0,0,0]) ## parameter used for specifying location of PV morphology

    
    #### Plotting Electrodes and Neurons
    ###################################################################################
    ###################################################################################

    PLOT_WAVEFORM = False
    
    if plot_neuron_with_electrodes:     
        ## Get Neuron Coordinates
        if cell_type in [6,7,35,36]:
            neuron = NeuronSim(human_or_mice=human_or_mice, cell_id=cell_type, temp=temp, dt=dt)
            coord = neuron._translate_rotate_neuron(pos_neuron=loc_pyr, angle=angle_pyr)
            
            ## Plot ICMS Electrode With Neuron 
            plot_electrode_and_neuron(coord_elec=elec_location_ICMS*10**3, coord=coord, savepath=savepath_curr)
                
            del neuron   
        
        else:
            print("Invalid Neuron Type Chosen!")
        
        ############################################################################################################
    
    #### Monopolar Stimulation
    ###################################################################################
    ###################################################################################
    
    ## Generating Waveforms
    save_state_show = False
    start_time, time_taken_round = time.time(), 0

    pulse_train_square = PulseTrain_square()
    sampling_rate = 1e6 ## ms, Hz
    amp_array, time_array = pulse_train_square.amp_train(amp=amplitude,pulse_width=pulse_width,period = period,total_time = total_time,sampling_rate = sampling_rate)
    amp_array2 = np.zeros(time_array.shape)
    if PLOT_WAVEFORM == True:
        pulse_train_square.plot_waveform()
    soma_recordings =[]
    t_filtered = []
    for l in range(len(elec_location_ICMS)):
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<")
        print("Starting Simulation for Electrode Location %d"%(l))
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<")
        ## Defining DATA Saving Directories
        #########################################################################################
   
        ## Generate Electric Field Simulator
        start_time = time.time()
        print("Loading Electric Field Simulator...")
        elec_field = ICMS(x=elec_location_ICMS[l, 0], 
                y=elec_location_ICMS[l, 1], 
                z=elec_location_ICMS[l, 2], 
                conductivity=0.33)
        elec_field2 = None
        
        print("Electric Field Simulator Loaded! Time Taken %s s"% (str(round(time.time()-start_time,3))))
       
        if cell_type in [6,7,35,36]:
            ### Run Pyr Stimulation
            ######################################################################################
            neuron = NeuronSim(human_or_mice=human_or_mice, cell_id=cell_type, temp=temp, dt=dt, elec_field=elec_field, elec_field2=elec_field2) ## Initializing neuron model
            print("Cell Id chosen %d."%(int((cell_type))))
            neuron._set_xtra_param(angle=angle_pyr, pos_neuron=loc_pyr)  ## Setting Extracellular Stim Paramaters
            delay_init, delay_final = 2000,5 ## ms, delay added to the stimulation before and after applying stimulation

            ## Checking if Save State for Pyramidal neuron model Exists and if not then creating one
            save_state = os.path.join(cwd,'cells/SaveState/human_or_mice'+str(human_or_mice)+'cell-'+str(cell_type)+'_Temp-'+str(temp)+'C_dt-'+str(dt*10**3)+'us_delay-'+str(delay_init)+'ms.bin')
            if not os.path.exists(save_state):
                start_time = time.time()
                print("Generating Save State...")
                neuron.stimulate(time_array=time_array, amp_array=amp_array, amp_array2=amp_array2, sampling_rate=sampling_rate, delay_init=delay_init, delay_final=delay_final, save_state_show=save_state_show)
                print("Save State Generated! Time Taken %s s"%(str(round(time.time()-start_time,3))))
    
            start_time = time.time()
            print("Simulation for Neuron Started...")
            results = neuron.stimulate(time_array=time_array, amp_array=amp_array, scale1=1, sampling_rate=sampling_rate, delay_init=delay_init, delay_final=delay_final) 
            print("Simulation Finished! Time Taken %s s"%(str(round(time.time()-start_time,3))))

            soma_recording_filtered,t_filtered = neuron.save_soma_recording(delay_init=delay_init)   
            
            soma_recordings.append(soma_recording_filtered) 
            del neuron
        else:
            print("Invalid Neuron Type Chosen!")
            break
    return np.array(soma_recordings), np.array(t_filtered)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run point electrode stimulation.")
    parser.add_argument("--num_electrode", type=int, default=1, help="Number of electrodes")
    parser.add_argument("--amplitude", type=float, default=10000, help="Amplitude of stimulation")
    parser.add_argument("--pulse_width", type=float, default=100, help="Pulse width in ms")
    parser.add_argument("--period", type=float, default=200, help="Pulse period in ms")
    parser.add_argument("--total_time", type=float, default=1000, help="Total stim duration in ms")
    parser.add_argument("--cell_type", type=int, default = 6, help="Cell type to stimulate")
    parser.add_argument("--plot_neuron_with_electrodes", type=bool, default = False, help="Show neuron with electrodes")
    args = parser.parse_args()
    pointElec_simulation(
        num_electrode = args.num_electrode,
        amplitude = args.amplitude,
        pulse_width = args.pulse_width,
        period = args.period,
        total_time = args.total_time,
        cell_type = args.cell_type,
        plot_neuron_with_electrodes = args.plot_neuron_with_electrodes
    )

if __name__ == "__main__":
    main()
