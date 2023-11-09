import inspect
try:
    import h5py
except ImportError:
    h5py = None
import numpy as np

'''
The detector system has to be compatible with pyFAI since
we read the detectors from the pyFAI generated poni Files
So all the detector class names have to match their pyFAI
equivalent
'''

class Detector():
    def __init__(self, pixel1, pixel2, max_shape, pixel_corners=None):
        self.pixel1 = pixel1
        self.pixel2 = pixel2
        self.shape = max_shape
        if pixel_corners is None:
            pixel_corners = self.get_pixel_corners(pixel1, pixel2, max_shape)
        self.pixel_corners = pixel_corners
     
     
    @classmethod
    def factory(cls, name, config = {}):
        cls = globals().get(name)
        if inspect.isclass(cls) and issubclass(cls, Detector):
            return cls(**config)
        else:
            raise RuntimeError(f'Detector {name} not supported')
    
    
    def get_pixel_corners(self, pixel1, pixel2, shape):
        '''
        h, w, corner index [A, B, C, D], coordinates [z, y, x]
        A D
        B C
        '''
        pixel_corners = np.zeros((shape[0], shape[1], 4, 3), dtype=np.float32)

        d1 = np.arange(shape[0] + 1, dtype=np.float32) * pixel1
        d2 = np.arange(shape[1] + 1, dtype=np.float32) * pixel2
        p2, p1 = np.meshgrid(d2, d1)
        
        pixel_corners[:,:, 0, 1] = p1[:-1, :-1]
        pixel_corners[:,:, 0, 2] = p2[:-1, :-1]
        pixel_corners[:,:, 1, 1] = p1[1:, :-1]
        pixel_corners[:,:, 1, 2] = p2[1:, :-1]
        pixel_corners[:,:, 2, 1] = p1[1:, 1:]
        pixel_corners[:,:, 2, 2] = p2[1:, 1:]
        pixel_corners[:,:, 3, 1] = p1[:-1, 1:]
        pixel_corners[:,:, 3, 2] = p2[:-1, 1:]
        return pixel_corners


class NexusDetector(Detector):
    def __init__(self, filename):
        if not h5py:
            raise RuntimeError('h5py module missing: NexusDetector is not supported') 
        
        def find_det(name, item):
            if item.attrs.get('NX_class') == 'NXdetector':
                return item
            
        with h5py.File(filename, 'r') as fh:
            det = fh.visititems(find_det)
            if det is None:
                raise RuntimeError(f'No NXdetector found in file: {filename}')
        
            pixel1, pixel2 = det['pixel_size'][()]
            shape = det['shape'][()]
            pixel_corners = det['pixel_corners'][()]
        super().__init__(pixel1, pixel2, shape, pixel_corners)
   

# Eiger detectors
class Eiger(Detector):
    max_shape = (10, 10)
    def __init__(self):
        super().__init__(75e-6, 75e-6, self.max_shape)
 
 
# Eiger1 detectors
class Eiger500k(Eiger):
    max_shape = (514, 1030)
    

class Eiger1M(Eiger):
    max_shape = (1065, 1030)


class Eiger4M(Eiger):
    max_shape = (2167, 2070)
    

class Eiger9M(Eiger):
    max_shape = (3269, 3110)


class Eiger16M(Eiger):
    max_shape = (4371, 4150)
    

# Eiger2 detectors
class Eiger2_250k(Eiger):
    max_shape = (512, 512)
    

class Eiger2_500k(Eiger):
    max_shape = (512, 1028)
    

class Eiger2_1M(Eiger):
    max_shape = (1062, 1028)
    
    
class Eiger2_4M(Eiger):
    max_shape = (2162, 2068)
    
    
class Eiger2_9M(Eiger):
    max_shape = (3262, 3108)
    
    
class Eiger2_16M(Eiger):
    max_shape = (4362, 4148)
  
  
# Eiger2 CdTe detectors
class Eiger2CdTe_500k(Eiger):
    max_shape = (512, 1028)
    

class Eiger2CdTe_1M(Eiger):
    max_shape = (1062, 1028)
    

class Eiger2CdTe_4M(Eiger):
    max_shape = (2162, 2068)
    
    
class Eiger2CdTe_9M(Eiger):
    max_shape = (3262, 3108)
    
    
class Eiger2CdTe_16M(Eiger):
    max_shape = (4362, 4148)
    

# Pilatus detectors
class Pilatus(Detector):
    max_shape = (10, 10)
    def __init__(self):
        super().__init__(172e-6, 172e-6, self.max_shape)
        
        
class Pilatus1M(Pilatus):
    max_shape = (1043, 981)
    

class Pilatus2M(Pilatus):
    max_shape = (1679, 1475)
    
    
class Pilatus6M(Pilatus):
    max_shape = (2527, 2463)
    

class PilatusCdTe1M(Pilatus):
     max_shape = (1043, 981)


class PilatusCdTe2M(Pilatus):
    max_shape = (1679, 1475)
    

class Pilatus4(Detector):
    max_shape = (10, 10)
    def __init__(self):
        super().__init__(150e-6, 150e-6, self.max_shape)
        
        
class Pilatus4_1M(Pilatus4):
    max_shape = (1080, 1033)
    
    
class Pilatus4_2M(Pilatus4):
    max_shape = (1630, 1553)
    
    
class Pilatus4_4M(Pilatus4):
    max_shape = (2180, 2073)
    
    
class Pilatus4_CdTe_1M(Pilatus4):
    max_shape = (1080, 1033)
    

class Pilatus4_CdTe_2M(Pilatus4):
    max_shape = (1630, 1553)
    
    
class Pilatus4_CdTe_4M(Pilatus4):
    max_shape = (2180, 2073)
