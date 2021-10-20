import os
import time
import psutil
import numpy as np
from multiprocessing import Process
from azint import AzimuthalIntegrator

def worker(ai, nrep):
    img = np.ones((2167, 2070), dtype=np.uint32)
    for i in range(nrep):
        ai.integrate(img)

def benchmark():
    start = time.perf_counter()
    base = os.path.dirname(__file__)
    poni = os.path.join(base, 'bench.poni')
    ai = AzimuthalIntegrator(poni,
                            (2167, 2070),
                            75.0e-6, 4,
                            2000,
                            solid_angle=True)
    end = time.perf_counter()
    print('Initialization time %.1fs' %(end - start))
    nworkers = psutil.cpu_count(logical=False)
    nrep = 200
    procs = []
    for i in range(nworkers):
        p = Process(target=worker, args=(ai, nrep))
        procs.append(p)
    
    start = time.perf_counter()
    for i in range(nworkers):
        procs[i].start()
    
    for i in range(nworkers):
        procs[i].join()
    end = time.perf_counter()
    duration = end - start
    print('Eiger 4M: %.2f frames/s with %d worker processes' %(nworkers * nrep / duration, nworkers))
