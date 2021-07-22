from multiprocessing import Process
import multiprocessing
import numpy as np 
from scipy.sparse import spdiags
from scipy.signal import medfilt
from scipy.ndimage import median_filter,gaussian_filter
import scipy.sparse as sparse
####################################
def parallel_tasks(du,Ix2,npixels,pp_d):
    tmp=pp_d*np.reshape(Ix2,(npixels,1),'F')
    du.append( spdiags(tmp.T, 0, npixels, npixels))
