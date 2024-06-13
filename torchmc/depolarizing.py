import numpy as np

class depolarizing():
    def __init__(self,Mueller_matrix,re):
        # hw, eh, ew = Mueller_matrix.size()
        self.Mueller_matrix = Mueller_matrix
        
        self.hw = re
        # self.w = w
        self.eh = 4
        self.ew = 4
        # [i,j,0-4,0-4]
        #  i,j for the size of image
        # 0-4, 0-4, up to 16 is the dimension of mueller matrix
    
    def depolarization_index(self):
        """calculate the Depolarization index
            Albert's tesis, page 35, eq 2.32
        Args:
            self.mueller_matrix ((N, 4, 4) numpy.ndarray): Numpy array containing N( H*W,could bemore ) mueller matrices.

        Returns:
            (N, 1) numpy.array: Numpy array containing degree of polarimetric purity
        """
        M_square = self.Mueller_matrix[:,:,:]**2
        M_square_sum = np.sum(self.Mueller_matrix.reshape(self.hw,self.eh*self.ew),-1)
        P_deta = np.sqrt((M_square_sum[:] -M_square[:,0,0])/(3*M_square[:,0,0]))
        return P_deta
    
    def cal_convariance(self,m):
        """Transform a mueller matrix into a hermitian covariance matrix
            using the relation 1/4 ∑ m_ij * (sigma_i ⊗ sigma_j),
            where sigma_i are the Pauli spin matrices.

        Args:
            m_m ((N, 4, 4) numpy.ndarray): Numpy array containing N mueller matrices.

        Returns:
            (N, 4, 4) numpy.array: Numpy array containing the respective covariance matrix for each
                                mueller matrix.
        """
        
        cov_matrix = np.zeros(m.shape, dtype='complex64')
        cov_matrix[:, 0, 0] = m[:, 0, 0] + m[:, 0, 1]    + m[:, 1, 0]    + m[:, 1, 1]
        cov_matrix[:, 0, 1] = m[:, 0, 2] + 1j*m[:, 0, 3] + m[:, 1, 2]    + 1j*m[:, 1, 3]
        cov_matrix[:, 0, 2] = m[:, 2, 0] + m[:, 2, 1]    - 1j*m[:, 3, 0] - 1j*m[:, 3, 1]
        cov_matrix[:, 0, 3] = m[:, 2, 2] + 1j*m[:, 2, 3] - 1j*m[:, 3, 2] + m[:, 3, 3]

        cov_matrix[:, 1, 1] = m[:, 0, 0] - m[:, 0, 1]    + m[:, 1, 0]    - m[:, 1, 1]
        cov_matrix[:, 1, 2] = m[:, 2, 2] - 1j*m[:, 2, 3] - 1j*m[:, 3, 2] - m[:, 3, 3]
        cov_matrix[:, 1, 3] = m[:, 2, 0] - m[:, 2, 1]    - 1j*m[:, 3, 0] + 1j*m[:, 3, 1]

        cov_matrix[:, 2, 2] = m[:, 0, 0] + m[:, 0, 1]    - m[:, 1, 0]    - m[:, 1, 1]
        cov_matrix[:, 2, 3] = m[:, 0, 2] - m[:, 1, 2]    + 1j*m[:, 0, 3] - 1j*m[:, 1, 3]
        cov_matrix[:, 3, 3] = m[:, 0, 0] - m[:, 0, 1]    - m[:, 1, 0]    + m[:, 1, 1]

        cov_matrix[:, 1, 0] = np.conjugate(cov_matrix[:, 0, 1])
        cov_matrix[:, 2, 0] = np.conjugate(cov_matrix[:, 0, 2])
        cov_matrix[:, 3, 0] = np.conjugate(cov_matrix[:, 0, 3])
        cov_matrix[:, 2, 1] = np.conjugate(cov_matrix[:, 1, 2])
        cov_matrix[:, 3, 1] = np.conjugate(cov_matrix[:, 1, 3])
        cov_matrix[:, 3, 2] = np.conjugate(cov_matrix[:, 2, 3])

        cov_matrix = np.divide(cov_matrix, 4.0)

        return cov_matrix

    def ipp(self):
        """
        calculate the depolarizing content of Mueller matrices, 
        the so-called Indices of Polarimetric Purity (IPP) 

        Albert's tesis page 36, eg.2.34

        Args:
            self.Mueller_matrix:
            ((N, 4, 4) numpy.ndarray): Numpy array containing N mueller matrices.

        Returns:
            P = [P1,P2,P3]
            (N, 3) numpy.array: Numpy array containing IPP indices
        """
        # Convert mueller matrix into covariance matrix and calculate eigenvalues/-vectors
        cov_matrix = self.cal_convariance(self.Mueller_matrix)
        eig_val_sorted, eig_vec_sorted = self.sorted_eigh(cov_matrix)
        # [N*4]

        P =np.zeros((self.hw, 3))
        p =np.zeros((self.hw, 4))
        for i in range(1,4):
            p[:,i] = i*(eig_val_sorted[:,i-1]- eig_val_sorted[:,i])
            # P[:,2] = (eig_val_sorted[i-1]- eig_val_sorted[i])

        P [:,0] = p[:,1]/self.Mueller_matrix[:,0,0]
        P [:,1] = (p[:,1]+p[:,2])/self.Mueller_matrix[:,0,0]
        P [:,2] = (p[:,1]+p[:,2]+p[:,3])/self.Mueller_matrix[:,0,0]

        #  P_deta = np.sqrt(2/3*P1**2+2/9*P2**2+1/9*P3**2)
        return P 

    def cp(self):
        """
        calculate the components of Purity 
        the so-called Indices of Polarimetric Purity (IPP) 

        Albert's tesis page 36, eg.2.34

        Args:
            self.Mueller_matrix:
            ((N, 4, 4) numpy.ndarray): Numpy array containing N mueller matrices.

        Returns:
            P  (N, 3) numpy.array: Numpy array containing m10,m20,m30, called P vectors, based on Lu-Chipman product decomposition
            D  (N, 3) numpy.array: Numpy array containing m01,m02,m03, called D vectors, based on Lu-Chipman product decomposition
            Ps (N, 1) numpy.array: Numpy array containing IPP indices
        """
        d = np.zeros((self.hw,3))
        d[:,0] = self.Mueller_matrix[:,0,1]/self.Mueller_matrix[:,0,0]
        d[:,1] = self.Mueller_matrix[:,0,2]/self.Mueller_matrix[:,0,0]
        d[:,2] = self.Mueller_matrix[:,0,3]/self.Mueller_matrix[:,0,0]
        D = np.sqrt(d[:,0]**2+d[:,1]**2+d[:,2]**2)
        
        p = np.zeros((self.hw,3))
        p[:,0] = self.Mueller_matrix[:,1,0]/self.Mueller_matrix[:,0,0]
        p[:,1] = self.Mueller_matrix[:,2,0]/self.Mueller_matrix[:,0,0]
        p[:,2] = self.Mueller_matrix[:,3,0]/self.Mueller_matrix[:,0,0]
        P = np.sqrt(p[:,0]**2+p[:,1]**2+p[:,2]**2)

        M_square = self.Mueller_matrix[:,:,:]**2
        sum_matrix = np.zeros((self.hw,1))
        for i in range(1,4):
            for j in range(1,4):
                sum_matrix[:,0] = sum_matrix[:,0] + M_square[:,i,j]
        Ps = np.sqrt(sum_matrix[:,0]/(3*self.Mueller_matrix[:,0,0]**2))

        return P,D,Ps
    
    def sorted_eigh(self,matrix):
        """Calculate the sorted eigenvalues and eigenvectors
        of a hermitian matrix

        Args:
            matrix ((N, 4, 4) numpy.ndarray): An array of 4x4 matrixes.

        Returns:
            ((N, 4) numpy.ndarray, (N, 4, 4) numpy.ndarray):
                Contains the sorted eigenvalues (first tuple pos)
                and eigenvectors of the matrix. The values are sorted by eigenvalues
                on descending order.
        """
        eig_val, eig_vec = np.linalg.eigh(matrix)

        # Sort eigenvalues and -vectors descending by eigenvalue
        idx = (-eig_val).argsort(axis=1)
        idx_vec = np.transpose(np.repeat(idx, 4, axis=1).reshape((*idx.shape, 4)),
                            (0, 2, 1))

        eig_val_sorted = np.take_along_axis(eig_val, idx, axis=1)
        eig_vec_sorted = np.take_along_axis(eig_vec, idx_vec, axis=2)

        return eig_val_sorted, eig_vec_sorted