import numpy as np
from tqdm import tqdm

BPS_TO_MBPS = 1e-6

class Histories:
    def __init__(self, files_tms, htype, num_nodes, tm_length_func=lambda: 5, max_steps=60):
        self._tms = []
        self._opts = []
        self._tm_times = []
        self._tm_ind = 0
        self._type = htype
        self._max_steps = max_steps
        self._tm_mask = np.ones((num_nodes, num_nodes), dtype=bool).flatten()
        self._tm_mask[np.eye(num_nodes).flatten() == 1] = False

        for fname in files_tms:
            print('[+] Populating TMS.')
            self._populate_tms(fname, tm_length_func)
            self._read_opt(fname)

    def _read_opt(self, fname):
        try:
            with open(fname.replace('hist', 'opt')) as f:
                lines = f.readlines()
                self._opts += [np.float64(_) for _ in lines]
        except:
            return None
        
    def _populate_tms(self, fname, tm_length_func):
        with open(fname) as f:
            for line in tqdm(f.readlines()):
                try:
                    tm = self._parse_tm_line(line)
                except:
                    import pdb;
                    pdb.set_trace()

                self._tms.append(tm * BPS_TO_MBPS)
                self._tm_times.append(tm_length_func())

    def __len__(self):
        return len(self._tms)
    
    def _parse_tm_line(self, line):
        tm = np.array([np.float64(_) for _ in line.split(" ") if _], dtype=np.float64)
        num_nodes = int(np.sqrt(tm.shape[0]))
        tm = tm.reshape((num_nodes, num_nodes))
        tm = (tm - tm * np.eye(num_nodes))
        return tm.flatten()[self._tm_mask]
    
    def get_next(self):
        tm = self._tms[self._tm_ind]
        tm_time = self._tm_times[self._tm_ind]
        opt_val = self._opts[self._tm_ind]
        return tm, tm_time, opt_val
    
    def num_tms(self):
        return len(self._tms)
    
    def num_histories(self):
        return 1
    
    def reset(self):
        self._tm_ind = 0