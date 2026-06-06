from __future__ import annotations
import numpy as np
import mne


def preprocess_raw(raw, fmin: float = 8.0, fmax: float = 30.0):
    raw = raw.copy()
    raw.filter(fmin, fmax, fir_design='firwin', verbose=False)
    return raw


def create_left_right_epochs(raw, tmin: float = 1.0, tmax: float = 4.0):
    """Create left/right motor imagery epochs from EEGBCI T1/T2 annotations."""
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    selected_event_id = {}
    if 'T1' in event_id:
        selected_event_id['left_or_class_1'] = event_id['T1']
    if 'T2' in event_id:
        selected_event_id['right_or_class_2'] = event_id['T2']
    if len(selected_event_id) != 2:
        raise ValueError(f'Could not find T1/T2 annotations. Found event_id={event_id}')
    picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, stim=False, exclude='bads')
    epochs = mne.Epochs(raw, events, event_id=selected_event_id, tmin=tmin, tmax=tmax, picks=picks, baseline=None, preload=True, verbose=False)
    X = epochs.get_data()
    y = epochs.events[:, -1]
    unique_codes = sorted(np.unique(y))
    mapping = {code: idx for idx, code in enumerate(unique_codes)}
    y = np.array([mapping[v] for v in y])
    return X, y, epochs
