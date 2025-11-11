from numpy.linalg import matrix_rank

def check_inverse_rank(matrix):
    rank = matrix_rank(matrix, tol=1e-12) # tol: Helps avoid counting very small numbers 
                                          #      due to floating-point errors as non-zero.
    print("matrix rank is : "+ str(rank))
    print("matrix size is : "+ str(matrix.shape))

    if matrix.shape[0] == matrix.shape[1]:
       if rank == matrix.shape[0]:
           print("matrix is invertible")
           return True
       else:
           print("matrix is not invertible")
           return False
    else:
       print("matrix is not square, hence not invertible")
       return False
