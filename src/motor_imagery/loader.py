from __future__ import annotations
from typing import Sequence
import mne
from mne.datasets import eegbci


def load_physionet_eegbci(subjects: Sequence[int], runs: Sequence[int]):
    """Load PhysioNet EEG Motor Movement/Imagery data using MNE."""
    raw_list = []
    for subject in subjects:
        file_paths = eegbci.load_data(subject, runs)
        for path in file_paths:
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            eegbci.standardize(raw)
            raw.set_montage('standard_1005', on_missing='ignore')
            raw_list.append(raw)
    if not raw_list:
        raise ValueError('No EEGBCI files loaded. Check subject/run values.')
    return mne.concatenate_raws(raw_list)
