import numpy as np
from numpy.core.fromnumeric import reshape
from numpy.core.numeric import zeros_like 
import scipy.ndimage
import cv2
from math import ceil,floor
import math
from scipy.ndimage.filters import convolve as filter2
from scipy.sparse import spdiags
from scipy.signal import medfilt
from scipy.ndimage import median_filter
import scipy.sparse as sparse
import denoise_LO as lo
import time
from cupyx.scipy.sparse.linalg import gmres
import cupyx
import cupy as cp
import mpi4py as mpi 
from mpi4py import MPI
from multiprocessing import Process
import multiprocessing
import multi_pro as mp
###########################################################
def parallel_tasks(du,i,Ix2,npixels,pp_d):
    tmp=pp_d*np.reshape(Ix2,(npixels,1),'F')
    du[i]= spdiags(tmp.T, 0, npixels, npixels)
###########################################################

###########################################################
def warp_image2(Image,XI,YI,h):
 
    ''' We add the flow estimated to the second image coordinates, remap them towards the ogriginal image  and finally  calculate the derivatives of the warped image
    h: derivative kernel
    '''

    Image=np.array(Image,np.float32)
    XI=np.array(XI,np.float32)
    YI=np.array(YI,np.float32)
    WImage=cv2.remap(Image,XI,YI,interpolation=cv2.INTER_CUBIC)
    Ix=filter2(Image, h)
    Iy=filter2(Image, h.T)
    
    Iy=cv2.remap(Iy,XI,YI,interpolation=cv2.INTER_CUBIC)   
    Ix=cv2.remap(Ix,XI,YI,interpolation=cv2.INTER_CUBIC)

    return [WImage,Ix,Iy]
    
############################################
def derivatives(Image1,Image2,u,v,h,b):
    '''This function compute the derivatives of the second warped image
    u: horizontal displacement 
    v: vertical displacement
    Image1, Image2: images sequence 
    h: derivative kernel 
    b: weight used for averaging 
    '''
    N,M=Image1.shape
    y=np.linspace(0,N-1,N)
    x=np.linspace(0,M-1,M)
    x,y=np.meshgrid(x,y)
    Ix=np.zeros((N,M))
    Iy=np.zeros((N,M))
    x=x+u; y=y+v
    WImage,I2x,I2y=warp_image2(Image2,x,y,h)  # Derivatives of the second image 

    It= WImage-Image1 # Temporal deriv
    
    Ix=filter2(Image1, h) # spatial derivatives for the first image 
    Iy=filter2(Image1, h.T)

    Ix  = b*I2x+(1-b)*Ix           # Averaging 
    Iy  = b*I2y+(1-b)*Iy


    It=np.nan_to_num(It) #Remove Nan values on the derivatives 
    Ix=np.nan_to_num(Ix)
    Iy=np.nan_to_num(Iy)
    out_bound= np.where((y > N-1) | (y<0) | (x> M-1) | (x<0))
    Ix[out_bound]=0 # setting derivatives value on out of bound pixels to 0  
    Iy[out_bound]=0
    It[out_bound]=0
    return [Ix,Iy,It]
############################################################
'''def conv_matrix(F,sz):
     #Construction of Laplacien Matrix 
    M=sparse.lil_matrix( ( sz[0]*sz[1],sz[0]*sz[1] ),dtype=np.float32)
    if( F.shape==(1,2)):
        for i in range(sz[0],sz[0]*sz[1]):      
            M[i,i-sz[0]]=-1     
        for i in range(sz[0],sz[0]*sz[1]): 
            M[i,i]=1
    elif(F.shape==(2,1)):
        for i in range(sz[0]*sz[1]):
            if(i==0):
                M[i,i]=0
            else:
                if(i%sz[0]!=0):
                    M[i,i]=1
                    M[i,i-1]=-1
    return M'''

'''def conv_matrix(F,sz):
     #Construction of Laplacien Matrix 
    M=sparse.lil_matrix( ( sz[0]*sz[1],sz[0]*sz[1] ),dtype=np.float32)
    for i in range(sz[0]*sz[1]):
        if( F.shape==(1,2)):
            if i>=sz[0]:
                M[i,i-sz[0]]=-1
                M[i,i]=1 
        elif(F.shape==(2,1)):
            if(i==0):
                M[i,i]=0
            else:
                if(i%sz[0]!=0):
                    M[i,i]=1
                    M[i,i-1]=-1
           
    return M'''
'''def conv_matrix(F,sz):
    # Construction of Laplacien Matrix 
    M=sparse.lil_matrix( ( sz[0]*sz[1],sz[0]*sz[1] ),dtype=np.float32)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    print('my raank is:', rank)
    size = comm.Get_size()
    block_size=sz[0]*sz[1]//size
    print('block s',block_size)
    #M=np.zeros( ( sz[0]*sz[1],sz[0]*sz[1] ),dtype=np.float32)

    for i in range(rank*block_size,(rank+1)*block_size):
        if( F.shape==(1,2)):
            if i>= sz[0]:
                M[i,i-sz[0]]=-1     
                M[i,i]=1
        elif(F.shape==(2,1)):
            if(i==0):
                M[i,i]=0
            else:
                if(i%sz[0]!=0):
                    M[i,i]=1
                    M[i,i-1]=-1
    
    comm.Barrier()
    

    return M'''
def conv_matrix(F,sz):
    '''Construction of Laplacien Matrix
    F: spacial filter it can be [-1,1] or [[-1],[1]] 
    sz: size of the used image
    I: rows of non zeros elements
    J: columns of non zeros elements
    K: The values of the matrix M 
    (IE: M(I,J)=K)
    We distinguish horizontal and vertical filters '''
    if(F.shape==(1,2)): 
        I=np.hstack((np.arange(sz[0],(sz[0]*(sz[1]))),np.arange(sz[0],(sz[0]*(sz[1])))))                                                        
        J=np.hstack((np.arange(sz[0],(sz[0]*(sz[1]))),np.arange(sz[0],(sz[0]*(sz[1]))))) 
        K=np.zeros(2*(sz[0]*(sz[1]-1)))
        K[:sz[0]*(sz[1]-1)]=1
        J[sz[0]*(sz[1]-1):2*(sz[0]*sz[1])]=J[sz[0]*(sz[1]-1):2*sz[0]*sz[1]]-sz[0]
        K[sz[0]*(sz[1]-1):2*(sz[0]*sz[1])]=-1
    
    if(F.shape==(2,1)):
        lI=[]
        for i in range(1,sz[0]*sz[1]):
            if(i%sz[0]!=0):
                lI.append(i)
        I=np.array(lI)
        nnzl=I.shape[0]
        I=np.hstack((I,I))
        
        
        J=np.hstack((np.array(lI),np.array(lI)-1))
        K=np.ones((2*nnzl))
        K[nnzl:2*nnzl]=-1
    M=sparse.lil_matrix( ( sz[0]*sz[1],sz[0]*sz[1] ),dtype=np.float32)
    M[I,J]=K

        
    
    return M
########################################################
def deriv_charbonnier_over_x(x,sigma,a):
    ''' Derivatives of the penality over x
     '''
    #y =2*a*(sigma**2 + x**2)**(a-1);
    y = 2 / (sigma**2)
    return y
def deriv_quadra_over_x(x,sigma):
    ''' Derivatives of the quadratique penality  penality over x '''
    y = 2 / (sigma**2)
    return y

# These lines make the 2 previous functions work on arrays (Define a vectorized function) 
charbonnier_over_x=np.vectorize(deriv_charbonnier_over_x)
quadr_ov_x=np.vectorize(deriv_quadra_over_x)
########################################################
def matrix_construct_loop(S,i,M1,M2,u,du,v,dv,npixels,eps,a):
    '''This is just a task that we use in flow_operator function 
    Read the comments of oprical_flow function for more details '''
    if(S[i].shape==(1,2)):
        M=M1
    elif(S[i].shape==(2,1)):
        M=M2
    u_=sparse.lil_matrix.dot(M,np.reshape((u+du),(npixels,1),'F'))
    v_=sparse.lil_matrix.dot(M,np.reshape((v+dv),(npixels,1),'F'))

    pp_su=charbonnier_over_x(u_,eps,a)
    pp_sv=charbonnier_over_x(v_,eps,a)
    dic_FU =sparse.lil_matrix.dot(M.T,sparse.lil_matrix.dot(spdiags(pp_su.T, 0, npixels, npixels),M))
    dic_FV=sparse.lil_matrix.dot(M.T,sparse.lil_matrix.dot(spdiags(pp_sv.T, 0, npixels, npixels),M))
    return[dic_FU,dic_FV]
########################################################
def flow_operator(u,v, du,dv, It, Ix, Iy,S,lmbda,eps,a,M1,M2,ret_A,ret_b):
    ''' Returns a linear flow operator (equation) of the form A * x = b using the a penality already choosen in .  
    The flow equation is linearized around UV with the initialization INIT
    (e.g. from a previous pyramid level).  Using Charbonnier function deriv_charbonnier_over_x function 
    u,v:  horizontal and vertical displacement 
    du,dv:  horizontal and vertical increment steps
    It,Ix,Iy: temporal and spatial derivatives 
    S: contains the spatial filters used for Computing the term related to Laplacien S=[ [[-1,1]], [[] [-1],[1] ]]
    lmbda: regularization parameter 
    eps,a: are the parameter of the penality used 
    M1,M2: Matrix of convolution used to compute laplacien term 
    ret_Aand ret_b: shared variables between some threads where we store the matrix and the second term computed 
    '''
    #sz=np.shape(Ix)
    npixels=Ix.shape[1]*Ix.shape[0]
    #charbonnier_over_x=np.vectorize(deriv_charbonnier_over_x)
    FU=sparse.lil_matrix((npixels,npixels),dtype=np.float32)
    FV=sparse.lil_matrix((npixels,npixels),dtype=np.float32)
    t1=time.time()
    
    for i in range(len(S)):
        if(S[i].shape==(1,2)):
            M=M1
        elif(S[i].shape==(2,1)):
            M=M2
        '''u_=sparse.lil_matrix.dot(M,np.reshape((u+du),(npixels,1),'F'))
        v_=sparse.lil_matrix.dot(M,np.reshape((v+dv),(npixels,1),'F'))

        pp_su=charbonnier_over_x(u_,eps,a)
        pp_sv=charbonnier_over_x(v_,eps,a)
        FU        = FU+ sparse.lil_matrix.dot(M.T,sparse.lil_matrix.dot(spdiags(pp_su.T, 0, npixels, npixels),M))
        FV        = FV+ sparse.lil_matrix.dot(M.T,sparse.lil_matrix.dot(spdiags(pp_sv.T, 0, npixels, npixels),M))'''

        U=sparse.lil_matrix.dot(M,np.hstack( (np.reshape((u+du),(npixels,1),'F'),np.reshape((v+dv),(npixels,1),'F'))) )
        pp_u=charbonnier_over_x(U,eps,a)
        FU        = FU+ sparse.lil_matrix.dot(M.T,sparse.lil_matrix.dot(spdiags(pp_u[:,0].T, 0, npixels, npixels),M))
        FV        = FV+ sparse.lil_matrix.dot(M.T,sparse.lil_matrix.dot(spdiags(pp_u[:,1].T, 0, npixels, npixels),M))

    
    t2=time.time()
    #print("tache 1", t2-t1)
    
    t1=time.time()
    MM = sparse.vstack( (sparse.hstack ( ( -FU, sparse.lil_matrix((npixels,npixels)) ) )  ,  sparse.hstack( ( sparse.lil_matrix((npixels,npixels)) , -FV ) )  ))  
    t2=time.time()
    #print("tache 2", t2-t1)
    del(FU); del(FV); del(M)
    t1=time.time()

    Ix2 = Ix*Ix
    Iy2 = Iy*Iy
    Ixy = Ix*Iy
    Itx = It*Ix
    Ity = It*Iy

    It = It + Ix*du+ Iy*dv 
    t2=time.time()
    #print("tache 3", t2-t1)
    t1=time.time()
    
    pp_d=charbonnier_over_x(np.reshape(It,(npixels,1),'F'),eps,a)
    tmp=pp_d*np.reshape(Ix2,(npixels,1),'F')
    duu = spdiags(tmp.T, 0, npixels, npixels)
    
    tmp = pp_d*np.reshape(Iy2,(npixels,1),'F')
    
    dvv = spdiags(tmp.T, 0, npixels, npixels)
    
    tmp = pp_d*np.reshape(Ixy,(npixels,1),'F')
    
    dduv = spdiags(tmp.T, 0, npixels, npixels)
    t2=time.time()
    #print("tache 4", t2-t1)
    '''manager = multiprocessing.Manager()
    ret_u = manager.dict()

    pu1 = multiprocessing.Process(target=parallel_tasks, args=(ret_u,0,Ix2,npixels,pp_d))
    pu2 = multiprocessing.Process(target=parallel_tasks, args=(ret_u,1,Iy2,npixels,pp_d))
    pu3 = multiprocessing.Process(target=parallel_tasks, args=(ret_u,2,Ixy,npixels,pp_d))
    pu1.start()
    pu2.start()
    pu3.start()
    pu1.join()
    pu2.join()
    pu3.join()
    duu=ret_u[0];dvv=ret_u[1];dduv=ret_u[2];'''
    t1=time.time()
    A = sparse.vstack( (sparse.hstack ( ( duu, dduv ) )  ,  sparse.hstack( ( dduv , dvv ) )  )) - lmbda*MM
    b=sparse.lil_matrix.dot(lmbda*MM, np.vstack((np.reshape(u, (npixels,1) ,'F'), np.reshape(v, (npixels,1) ,'F') ) )) - np.vstack( (pp_d*np.reshape(Itx,(npixels,1),'F'),  pp_d*np.reshape(Ity,(npixels,1),'F') ) )
    t2=time.time()
    #print("tache 4", t2-t1)
    '''if( ((np.max(pp_su)-np.max(pp_su)<1E-06)   ) and ((np.max(pp_sv)-np.max(pp_sv)<1E-06)   ) and ((np.max(pp_d)-np.max(pp_d)<1E-06)   ) ):
        iterative=False
    else:
        iterative=True
    #return [A,b,iterative]'''
    ret_A[0]=A; ret_b[0]=b
#############################################################
def flow_operator_quadr(u,v, du,dv, It, Ix, Iy,S,lmbda,sigma_qua,M1,M2,ret_A,ret_b):
    ''' Returns a linear flow operator (equation) of the form A * x = b using the a Quadratic penality  .  
    The flow equation is linearized around UV with the initialization INIT
    (e.g. from a previous pyramid level).  Using Charbonnier function deriv_charbonnier_over_x function 
    u,v:  horizontal and vertical displacement 
    du,dv:  horizontal and vertical increment steps
    It,Ix,Iy: temporal and spatial derivatives 
    S: contains the spatial filters used for Computing the term related to Laplacien S=[ [[-1,1]], [[] [-1],[1] ]]
    lmbda: regularization parameter 
    sigma_qua: is a parameter related to the quadratic penality  
    M1,M2: Matrix of convolution used to compute laplacien term 
    ret_Aand ret_b: shared variables between some threads where we store the matrix and the second term computed 
    PS: This function is similar to flow_operator function, the only difference is that we use a quadratic penality here and a flexible penality in the other function 
    ''' 
    npixels=Ix.shape[1]*Ix.shape[0]

    FU=sparse.lil_matrix((npixels,npixels),dtype=np.float32)
    FV=sparse.lil_matrix((npixels,npixels),dtype=np.float32)
    for i in range(len(S)):
        #t1=time.time()
        #M=conv_matrix(S[i],sz)
        if(S[i].shape==(1,2)):
            M=M1
        elif(S[i].shape==(2,1)):
            M=M2
        #t2=time.time()
        #print("time matrix:",t2-t1)

        '''u_=sparse.lil_matrix.dot(M,np.reshape((u+du),(npixels,1),'F'))
        v_=sparse.lil_matrix.dot(M,np.reshape((v+dv),(npixels,1),'F'))'''
        U=sparse.lil_matrix.dot(M,np.hstack( (np.reshape((u+du),(npixels,1),'F'),np.reshape((v+dv),(npixels,1),'F'))) )

        '''pp_su=quadr_ov_x(u_,sigma_qua)
        pp_sv=quadr_ov_x(v_,sigma_qua)'''
        pp_u=quadr_ov_x(U,sigma_qua)
        
        '''FU        = FU+ sparse.lil_matrix.dot(M.T,sparse.lil_matrix.dot(spdiags(pp_su.T, 0, npixels, npixels),M))
        FV        = FV+ sparse.lil_matrix.dot(M.T,sparse.lil_matrix.dot(spdiags(pp_sv.T, 0, npixels, npixels),M))'''
        FU        = FU+ sparse.lil_matrix.dot(M.T,sparse.lil_matrix.dot(spdiags(pp_u[:,0].T, 0, npixels, npixels),M))
        FV        = FV+ sparse.lil_matrix.dot(M.T,sparse.lil_matrix.dot(spdiags(pp_u[:,1].T, 0, npixels, npixels),M))

    MM = sparse.vstack( (sparse.hstack ( ( -FU, sparse.lil_matrix((npixels,npixels)) ) )  ,  sparse.hstack( ( sparse.lil_matrix((npixels,npixels)) , -FV ) )  ))  
    del(FU)
    del(FV)
    del(M)
    Ix2 = Ix*Ix #Ix^2
    Iy2 = Iy*Iy #Iy^2
    Ixy = Ix*Iy #Ix*Iy
    Itx = It*Ix #It*Ix
    Ity = It*Iy #It*Iy

    It = It + Ix*du+ Iy*dv
    
    pp_d=quadr_ov_x(np.reshape(It,(npixels,1),'F'),sigma_qua)
    
    tmp=pp_d*np.reshape(Ix2,(npixels,1),'F')
    duu = spdiags(tmp.T, 0, npixels, npixels)
    
    tmp = pp_d*np.reshape(Iy2,(npixels,1),'F')
    
    dvv = spdiags(tmp.T, 0, npixels, npixels)
    
    tmp = pp_d*np.reshape(Ixy,(npixels,1),'F')
    
    dduv = spdiags(tmp.T, 0, npixels, npixels)
    '''manager = multiprocessing.Manager()
    ret_u = manager.dict()

    pu1 = multiprocessing.Process(target=parallel_tasks, args=(ret_u,0,Ix2,npixels,pp_d))
    pu2 = multiprocessing.Process(target=parallel_tasks, args=(ret_u,1,Iy2,npixels,pp_d))
    pu3 = multiprocessing.Process(target=parallel_tasks, args=(ret_u,2,Ixy,npixels,pp_d))
    pu1.start()
    pu2.start()
    pu3.start()
    pu1.join()
    pu2.join()
    pu3.join()
    duu=ret_u[0];dvv=ret_u[1];dduv=ret_u[2];'''
    A = sparse.vstack( (sparse.hstack ( ( duu, dduv ) )  ,  sparse.hstack( ( dduv , dvv ) )  ))
    A=A-lmbda*MM
    
    b=sparse.lil_matrix.dot(lmbda*MM, np.vstack((np.reshape(u, (npixels,1) ,'F'), np.reshape(v, (npixels,1) ,'F') ) )) - np.vstack( (pp_d*np.reshape(Itx,(npixels,1),'F'),  pp_d*np.reshape(Ity,(npixels,1),'F') ) ) 

    '''if( ((np.max(pp_su)-np.max(pp_su)<1E-06)   ) and ((np.max(pp_sv)-np.max(pp_sv)<1E-06)   ) and ((np.max(pp_d)-np.max(pp_d)<1E-06)   ) ):
        iterative=False
    else:
        iterative=True
    #return [A,b,iterative]'''
    ret_A[1]=A; ret_b[1]=b
#############################################################
def  compute_flow_base(Image1,Image2,max_iter,max_linear_iter,u,v,alpha,lmbda,S,size_median_filter,h,coef,uhat,vhat,itersLO,lambda2,lambda3,remplacement,eps,a,sigma_qua,M1,M2):
    '''COMPUTE_FLOW_BASE   Base function for computing flow field using u,v displacements as an initialization
   - Image1,Image2: Image sequence
    -max_iter: warping iteration 
    -max_linear_iter:  maximum number of linearization performed per warping
    -alpha: a parameter tused to get a weighted energy: Ec=alpha*E_quadratic+(1-alpha)E_penality 
    -S: contains the spatial filters used for Computing the term related to Laplacien S=[ [[-1,1]], [[] [-1],[1] ]]
    -size_median_filter: is the size of the used median filter or the size of the neighbors used during LO optimization(The new median formula)
    -h: spatial derivative kernel 
    -coef: factor to average the derivatives of the second warped image and the first (used on derivatives functions to get Ix,Iy and It )
    -uhat,vhat: auxiliar displacement fields 
    -itersLO: iterations for LO formulation
    -lambda2: are the parameters 
    -lmbda: regularization parameter 
    -sigma_qua: is a parameter related to the quadratic penality   
    -lambda2: weight for coupling term 
    -lambda3: weight for non local term term
    remplacement: binary variable telling us to remplace the fileds by auixilary fields or not 
    M1,M2: Matrices of convolution used to compute laplacien term 
    '''
    npixels=u.shape[0]*u.shape[1]
    #u0=np.zeros((u.shape)); v0=np.zeros((u.shape));
    charbonnier_over_x=np.vectorize(deriv_charbonnier_over_x)
    Lambdas=np.logspace(math.log(1e-4), math.log(lambda2),max_iter)
    lambda2_tmp=Lambdas[0] 
    for i in range(max_iter):
        du=np.zeros((u.shape)); dv=np.zeros((v.shape))
       
        [Ix,Iy,It]=derivatives(Image1,Image2,u,v,h,coef)
        #print('iter',i,'shapes',Ix.shape)
        for j in range(max_linear_iter):
            t1=time.time()
            if (alpha==1):
                #[A,b,iterative]=flow_operator_quadr(u,v, du,dv, It, Ix, Iy,S,lmbda,sigma_qua,M1,M2,ret_A,ret_b)
                manager = multiprocessing.Manager()
                ret_A = manager.dict()
                managerb=multiprocessing.Manager()
                ret_b=managerb.dict()
                ret_A[0]=0; ret_b[0]=0
                p1 = multiprocessing.Process(target=flow_operator_quadr, args=(u,v, du,dv, It, Ix, Iy,S,lmbda,sigma_qua,M1,M2,ret_A,ret_b))
                p1.start()
                p1.join()

                A=ret_A[1]
                b=ret_b[1]
            elif(alpha>0 and alpha != 1):
                #[A,b,iterative]=flow_operator_quadr(u,v, du,dv, It, Ix, Iy,S,lmbda,sigma_qua,M1,M2,ret_A,ret_b)
                #[A1,b1,iterative1]=flow_operator(u,v, du,dv, It, Ix, Iy,S,lmbda,eps,a,M1,M2,ret_A,ret_b)

            
                manager = multiprocessing.Manager()
                ret_A = manager.dict()
                managerb=multiprocessing.Manager()
                ret_b=manager.dict()
                p1 = multiprocessing.Process(target=flow_operator_quadr, args=(u,v, du,dv, It, Ix, Iy,S,lmbda,sigma_qua,M1,M2,ret_A,ret_b))
                p2 = multiprocessing.Process(target=flow_operator, args=(u,v, du,dv, It, Ix, Iy,S,lmbda,eps,a,M1,M2,ret_A,ret_b))
                p2.start()
                p1.start()

                p1.join()
                p2.join()
                
                #A=alpha*A+(1-alpha)*A1
                #b=alpha*b+(1-alpha)*b1
                A=alpha*ret_A[1]+(1-alpha)*ret_A[0]
                b=alpha*ret_b[1]+(1-alpha)*ret_b[0]
            elif(alpha==0):
                #[A,b,iterative]=flow_operator(u,v, du,dv, It, Ix, Iy,S,lmbda,eps,a,M1,M2)
                manager = multiprocessing.Manager()
                ret_A = manager.dict()
                managerb=multiprocessing.Manager()
                ret_b=managerb.dict()
                p1 = multiprocessing.Process(target=flow_operator, args=(u,v, du,dv, It, Ix, Iy,S,lmbda,eps,a,M1,M2,ret_A,ret_b))
                p1.start()
                p1.join()

                A=ret_A[0]
                b=ret_b[0]
            tmp0=np.reshape( np.hstack((u-uhat,v-vhat)) , (1,2*u.shape[0]*u.shape[1]) ,'F')
            #tmp1=deriv_charbonnier_over_x(tmp0,eps,a)
            tmp1=charbonnier_over_x(tmp0,eps,a)
            
            tmpA=spdiags(tmp1,0,A.shape[0],A.shape[1])
            A= A + lambda2_tmp*tmpA
            b=b - lambda2_tmp*tmp1.T*tmp0.T
            t2=time.time()
            #print("temps pour construire la matrice:",(t2-t1))
            #x=scipy.sparse.linalg.spsolve(A,b)  #Direct solvers 
            #y=scipy.sparse.linalg.gmres(A,b)  #Gmres  solver 
            #y=scipy.sparse.linalg.bicg(A,b) #BICg Solver 
            #y=scipy.sparse.linalg.lgmres(A,b) # LGMRES 
            #diag=1/A.diagonal()
            t1=time.time()
            P=spdiags(A.diagonal(), 0, A.shape[0],A.shape[1] ) #Precond de Jacobi
            

            y=scipy.sparse.linalg.minres(A,b,M=P) #entre 1 et 2   650 s  #Minres Solver using P as preconditioner 
            #y=scipy.sparse.linalg.minres(A,b) #Minres Solver Without preconditioner
            #y=scipy.sparse.linalg.qmr(A,b) #Qmr solver 
            #y=gmres(cupyx.ndarray(A),cupyx.ndarray(b)) #testung cupyx solvers 
            #A_gpu=(cupyx.scipy.sparse.spmatrix.copy(A)).asfptype()
            #b_gpu=(cupyx.scipy.sparse.spmatrix.copy(scipy.sparse.csr_matrix(b))).asfptype()

            #y=gmres(A_gpu,b_gpu) #GMRES solver 
            x=y[0]
            t2=time.time()
            x[x>1]=1
            x[x<-1]=-1
            du=np.reshape(x[0:npixels], (u.shape[0],u.shape[1]),'F' )
            dv=np.reshape(x[npixels:2*npixels], (u.shape[0],u.shape[1]),'F' )

            
        print('it warping',i)
        u = u + du
        v = v + dv
        t1=time.time()
        '''uhat=lo.denoise_LO (u, size_median_filter, lambda2_tmp/lambda3, itersLO) # Denoising LO new formula of optimization  
        vhat=lo.denoise_LO (v, size_median_filter, lambda2_tmp/lambda3, itersLO)'''
        #[uhat,vhat]=lo.denoise_LO (u,v, size_median_filter, lambda2_tmp/lambda3, itersLO)
        uhat=median_filter(u,size=size_median_filter) # Denoising using a normal median filter 
        vhat=median_filter(v,size=size_median_filter)
        t2=time.time()
        #print("Temps optimisation mediane",(t2-t1))
        if remplacement==True:
            u=uhat
            v=vhat
        if i!=max_iter-1:

            lambda2_tmp=Lambdas[i+1] # Increment Lambda2

        
    return [u,v,uhat,vhat]

