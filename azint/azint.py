import numpy as np
from multiprocessing import shared_memory
from _azint import generate_matrix, spmv


class Poni():
    def __init__(self, filename):
        config = {}
        with open(filename) as opened_file:
            for line in opened_file:
                if line.startswith("#") or (":" not in line):
                    continue
                words = line.split(":", 1)
                key = words[0].strip().lower()
                value = words[1].strip()
                config[key] = value
                
        self.dist = float(config['distance'])
        self.poni1 = float(config['poni1'])
        self.poni2 = float(config['poni2'])
        self.rot1 = float(config['rot1'])
        self.rot2 = float(config['rot2'])
        self.rot3 = float(config['rot3'])
        self.wavelength = float(config['wavelength'])


class AzimuthalIntegrator():
    def __init__(self, poni_file, shape, pixel_size, n_splitting, mask, bins, solid_angle=True, create=True):
        self._shms = []
        self.create = create
        self.output_shape = [len(axis)-1 for axis in bins[::-1]]
        poni = Poni(poni_file)
        if self.create:
            self.sparse_matrix = generate_matrix(poni, shape, pixel_size, n_splitting, mask, bins)
            self._make_shm('azint_col_idx', self.sparse_matrix[0])
            self._make_shm('azint_row_ptr', self.sparse_matrix[1])
            self._make_shm('azint_values', self.sparse_matrix[2])
        
        else:
            col_idx = self._load_shm('azint_col_idx', np.int32)
            row_ptr = self._load_shm('azint_row_ptr', np.int32)
            values = self._load_shm('azint_values', np.float32)
            self.sparse_matrix = (col_idx, row_ptr, values)
            
        if solid_angle:
            d1 = (np.arange(shape[0], dtype=np.float32) + 0.5) * pixel_size - poni.poni1
            d2 = (np.arange(shape[1], dtype=np.float32) + 0.5) * pixel_size - poni.poni2
            p1, p2 = np.meshgrid(d2, d1)
            solid_angle = poni.dist / np.sqrt(poni.dist**2 + p1*p1 + p2*p2)
            self.norm = spmv(*self.sparse_matrix, solid_angle**3)
        else:
            self.norm = spmv(*self.sparse_matrix, np.ones(shape[0]*shape[1], dtype=np.float32))
            
    def close(self):
        for shm in self._shms:
            shm.close()
            if self.create:
                shm.unlink()
                pass
            
    def integrate(self, img):
        signal = spmv(*self.sparse_matrix, img)
        result = np.divide(signal, self.norm, out=np.zeros_like(signal), where=self.norm!=0.0)
        return result.reshape(self.output_shape)

    def _make_shm(self, name, a):
        shm = shared_memory.SharedMemory(name, create=True, size=a.nbytes)
        self._shms.append(shm)
        shm_a = np.ndarray(shape=a.shape, dtype=a.dtype, buffer=shm.buf)
        shm_a[:] = a[:]
    
    def _load_shm(self, name, dtype):
        shm = shared_memory.SharedMemory(name=name)
        self._shms.append(shm)
        size = shm.size // 4
        a = np.frombuffer(shm.buf, dtype=dtype)
        return a
