import torch
import numpy as np

a = torch.tensor([1, 2, 3])  # creating a 2d array which will execute with numbers 1,2,3
b = torch.zeros(2, 3)  # array with 2 rows and 3 columns and filled with 0
c = torch.ones(4)  # a 1d array with 4 columns and filed with 1
d = torch.rand(2, 2)  # generating random numbers at 2x2 array that too between 0 to 1
e = torch.randn(2, 2)  # generating random numbers at 2x2 array with random numbers
f = torch.eye(3)  # creating an identity matrix of size 3x3 with 1 at diagonals
g = torch.arange( 0, 10, 2)  # creating a 1d array with numbers
# #starting from 0 to 10 with step size of 2
h = torch.linspace(0, 1, 5)#

print("Tensor a:", a)
print("Zeros (b):\n", b)
print("Ones (c):", c)
print("Random Uniform (d):\n", d)
print("Random Normal (e):\n", e)
print("Identity Matrix (f):\n", f)
print("Arange (g):", g)
print("Linspace (h):", h)
#reshaping and viewing
print("\nReshaping and Viewing:")
x = torch.rand(2, 3)#random numbers of 2x3 matrix
print("Original x:\n", x)# representing the original matrix
print("Reshape (3, 2):\n", x.view(3, 2))#reshaping the matrix to 3x2
print("Transpose:\n", x.T)#creating transpose of matrix
print("Unsqueeze (add dim):\n", x.unsqueeze(0))#adding another bracket to minimise the dimension
#actual definition:adds a new dimension to the tensor at the specified position
print("Squeeze (remove dim):\n", x.unsqueeze(0).squeeze())#removes the added dimension
# # 3. Arithmetic Operations
a = torch.tensor([1.0, 2.0])
b = torch.tensor([3.0, 4.0])
print("Add:", a + b)#adds elements from a same column
print("Multiply:", a * b)#multiply elements from a same column
print("Divide:", a / b)#divides elements from the same column

print("Dot Product:", a @ b)#mulprod
print("Mean:", a.mean())
print("Sum:", a.sum())
print("Max:", a.max())
#Indexing and slicing
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("x[0] (first row):", x[0])
print("x[:,1] (second column):", x[:, 1])
print("x[1,2] (row 1, col 2):", x[1, 2])
print("x[0:2,1:] (slice):\n", x[0:2, 1:])
#tensor info
print("Shape:", x.shape)
print("Dimensions:", x.ndim)
print("Data type:", x.dtype)
print("Size:", x.size())
print("Device:", x.device)

#move to gpu
print("move to gpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.rand(2, 2).to(device)
print("Tensor on device:", x)

#numpy conversion
print("--------------------------")
print("numpy conversion")
a_np = np.array([1, 2, 3])
b_torch = torch.from_numpy(a_np)
c_np_back = b_torch.numpy()
print("NumPy to Tensor:", b_torch)
print("Tensor to NumPy:", c_np_back)


#random seed
print("-------------------------")
print("Random seed")
torch.manual_seed(42)
print("Random (with seed):", torch.rand(2, 2))
