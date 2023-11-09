import os
import time
import numpy as np
from multiprocessing import Process, cpu_count
from azint import AzimuthalIntegrator

def worker(ai, nrep):
    img = np.ones((2167, 2070), dtype=np.uint32)
    for i in range(nrep):
        ai.integrate(img)

def benchmark():
    start = time.perf_counter()
    base = os.path.dirname(__file__)
    poni = os.path.join(base, 'bench.poni')
    ai = AzimuthalIntegrator(poni, 4, 2000)
    end = time.perf_counter()
    print('Initialization time %.1fs' %(end - start))
    nworkers = cpu_count()
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
