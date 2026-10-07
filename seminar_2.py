import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt

'''-------------- IMPORT DATA ------------'''
data_file = '/Users/alextumyalis/mne_data/MNE-sample-data/MEG/sample/sample_audvis_filt-0-40_raw.fif'
raw = mne.io.read_raw_fif(data_file , preload=True)
raw.info['bads'] = []

raw_eeg = raw.copy().pick_types(eeg=True, eog=True, stim=True).filter(0.1, 20).resample(sfreq=100)
raw_eeg.plot()



'''-------------- CHANNEL TYPES ------------'''
raw_eeg.set_channel_types({'EEG 001': 'emg'})


'''-------------- DROP CHANNEL -------------'''
raw_eeg.drop_channels([])
raw_eeg.plot()


''' ------------ CROP EEG ------------------'''
raw_eeg_crop = raw_eeg.copy().crop(tmin=1, tmax=10)
raw_eeg_crop.plot()


'''-------------- CHANNELS MONTAGE ------------------'''
# EEG channels
ch_names = raw_eeg.ch_names
print(ch_names)
raw_eeg.plot_sensors(kind='topomap', ch_type='eeg', show_names=True)

# standard montage
mne.channels.get_builtin_montages()
montage = mne.channels.make_standard_montage('mgh60')
montage.plot()
cn = montage.ch_names

# rename channels
rename_chan = dict(zip(ch_names[9:69], cn))
mne.rename_channels(raw_eeg.info, rename_chan)

# set montage
ten_twenty_montage = mne.channels.make_standard_montage('mgh60', head_size=0.095)
ten_twenty_montage.plot(kind='topomap', show_names=True, sphere=(0, 0.011, 0, 0.095))
raw_eeg = raw_eeg.set_montage(ten_twenty_montage)


'''------------- EVENTS -----------------'''
events = mne.find_events(raw_eeg, stim_channel='STI 014')
raw_eeg.events = events

print(events[:20])


event_dict = {'audial/left':  1, 
              'audial/right': 2, 
              'visual/left':    3,
              'visual/right':   4, 
              'smiley':         5, 
              'buttonpress':    32}

mne.viz.plot_events(events, event_id=event_dict, sfreq=raw_eeg.info['sfreq'])


event_dict = {'audial/left':  1, 
              'audial/right': 2, 
              'visual/left':  3,
              'visual/right': 4}


'''------------- EPOCHS -------------------'''
'if epochs overlap than incorrect visualisation'
epochs = mne.Epochs(raw_eeg, 
                    picks=['eeg', 'eog'],
                    events=raw_eeg.events, 
                    event_id=event_dict, 
                    tmin=-0.1, 
                    tmax=0.5,
                    preload=True)
epochs.info

epochs_vis = epochs['visual']
epochs_aud = epochs['audial']

epochs.plot(n_epochs=10,
            n_channels=20,
            events=events,
            event_id=event_dict,
            picks=['eeg', 'eog'])

bad_epochs = [16, 38, 101, 184, 205, 230, 268]
bad_ind = []
for ind, b in enumerate(epochs.selection):
    if b in bad_epochs:
        bad_ind.append(ind)

epochs.drop(bad_ind)

epochs.drop_bad(dict(eeg=100e-6))


eeg_epochs_all = mne.concatenate_epochs([epochs_vis, epochs_aud])

len(eeg_epochs_all_cond)

eeg_epochs_all_cond.plot(n_epochs=20,
                         n_channels=20,
                         events=eeg_epochs_all_cond.events,
                         event_id=event_dict)


epochs.info
''' ---------------- ICA -----------------'''
ica = mne.preprocessing.ICA(n_components=40,
                            random_state=97,
                            max_iter=800,
                            method='infomax',
                            fit_params=dict(extended=True))
ica.fit(epochs, picks=['eeg'])





def eeg_ica(eeg, ic_number = 64):
    ica = mne.preprocessing.ICA(n_components=ic_number, 
                                noise_cov=None, 
                                random_state=42, 
                                method='infomax', 
                                fit_params=dict(extended=True))
    ica.fit(eeg, picks=['eeg'])
    ica.apply(eeg)
    return ica


ica.plot_sources(epochs, start=0, stop=10, show_scrollbars=True)
epochs.info

ica.exclude = [1, 2]

ica.plot_properties(epochs, picks=ica.exclude)


epochs_ica = ica.apply(epochs)
epochs_ica.plot(n_channels=40)



'''----------------- REJECT EPOCHS ------------------'''
reject_criteria = dict(eeg=100e-6)  # 100 µV
epochs_ica.drop_bad(reject=reject_criteria)
epochs_ica.plot_drop_log()



'''----------------- METADATA -----------------------'''
ep_number = len(epochs_ica)

md = pd.DataFrame([np.random.randint(0, 10, size=ep_number),
                   np.random.randint(0, 10, size=ep_number)]).T

epochs_ica.metadata = md
epochs_ica.metadata


'''---------------- BASELINE CORRECTION --------------'''
epochs_ica = epochs_ica.apply_baseline((-0.1, 0.0))



'''----------------- AVERAGING ----------------------'''
epochs_vis = epochs_ica['visual']
epochs_aud = epochs_ica['audial']

epochs_vis.plot()
epochs_aud.plot()

epochs_vis['left'].average().plot()
epochs_aud['right'].average().plot()


'''--------------- VISUALIZATION ---------------------''' 
epochs_vis['left'].average().plot_topomap(times=np.linspace(0.0, 0.50, 5),
                                          average=0.05,
                                          nrows=1,
                                          #sphere=(0.00, 0.025, 0.015, 0.11)),
                                          cmap='jet',
                                          #contours=None,
                                          time_unit='ms')

epochs_aud.average().plot_joint(times=[0.090, 0.180, 0.250]) # times=0.275


condition_diff = [epochs_vis['left'].average(), epochs_vis['right'].average()]
evoked_contrast = mne.combine_evoked(condition_diff, weights=[1, -1])
mne.viz.plot_compare_evokeds(condition_diff, picks='EEG059')

mne.viz.plot_evoked_topo(condition_diff)



vis_df = epochs_vis.average().to_data_frame()



peak = epochs_vis.average().pick_channels(['EEG040']).get_peak(ch_type='eeg', tmin=0.080, tmax=0.120,
                                                              mode='pos', return_amplitude=True)


fig, ax = plt.subplots(1,2)
mne.viz.plot_compare_evokeds(condition_diff, picks='EEG059', axes=ax[0])
mne.viz.plot_compare_evokeds(condition_diff, picks='EEG053', axes=ax[1])
