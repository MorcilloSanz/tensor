# Talg (Tensor algebra library)

Tensor algebra library written in a `single` header file in C. The goal of this project is to create a tensor algebra library designed for `graphics programming`, that can also be useful in other areas such as `deep learning` or `physics`.
> It is written in C11 (-std=c11)

![](img/tensor.png)

## Table of contents

- [Create tensor](#create-tensor)
- [Destroy tensor](#destroy-tensor)
- [Create a copy of a tensor](#create-a-copy-of-a-tensor)
- [Create identity matrix](#create-identity-matrix)
- [Print tensor](#print-tensor)
- [Set and get values](#set-and-get-values)
- [Addition and subtraction](#addition-and-subtraction)
- [Product](#product)
- [Hadamard product](#hadamard-product)
- [Tensor product](#tensor-product)
- [Dot product](#dot-product)
- [Cross product](#cross-product)
- [Transpose](#transpose)
- [Cofactor matrix](#cofactor-matrix)
- [Adjugate matrix](#adjugate-matrix)
- [Determinant](#determinant)
- [Inverse](#inverse)
- [Transform vector](#transform-vector)
- [**TODO** Subtensors]()
- [**TODO** Singular value decomposition (SVD)]()
- [**TODO** Eigenvalues and eigenvectors ]()
- [**TODO** Solve linear system ]()
- [**TODO** Jacobi method ]()
- [**TODO** Jacobi eigenvalue algorithm]()


## Create tensor
Talg allows you to create vectors, matrices and tesors (row-major order):

**Create rank 1 tensor (vector):** creates a vector of dim *n*.
```c
Vector* create_vector(uint8_t dim);
```
**Create rank 2 tensor (matrix):** creates a matrix of *n* cols and *m* rows.
```c
Matrix* create_matrix(uint8_t cols, uint8_t rows);
```
**Create rank 3 tensor:** creates a rank 3 tensor of *n* cols, *m* rows and *d* depth.
```c
Tensor* create_tensor_rank3(uint8_t cols, uint8_t rows, uint8_t depth);
```
**Create rank 4 tensor:** creates a rank 3 tensor of *n* cols, *m* rows, *d* depth and *t* time.
```c
Tensor* create_tensor_rank4(uint8_t cols, uint8_t rows, uint8_t depth, uint8_t time);
```

## Destroy tensor
> **Important:** as these create methods allocates memory in the heap. It is important to free them once they are not useful anymore:

```c
void destroy_tensor(Tensor* tensor);
```

## Create a copy of a tensor
Creates a copy of a tensor:
```c
Tensor* create_copy(Tensor* tensor);
```
```c
Matrix* matrix = create_identity(4);
Matrix* matrix_copy = create_copy(matrix);
```

## Create identity matrix
Creates a identity squared matrix of *n* cols and *n* rows.

```c
Matrix* create_identity(uint8_t cols);
```

```c
Matrix* matrix = create_identity(4);
```

## Print tensor
Prints a tensor of any rank:

```c
void print_tensor(Tensor* tensor);
```

## Set and get values
Set and get values from any algebraic structure:
```c
void set(Tensor* tensor, double value, ...);
```
```c
double get(Tensor* tensor, ...);
```
**Vector:** *set(vector, value, j)* and *get(vector, j)* as a vector is a matrix of one column, *j* is the corresponding index in the column.
```c
Vector* vector = create_vector(3);

set(vector, 1.0, 0); 
set(vector, 2.0, 1); 
set(vector, 3.0, 2);

double value = get(vector, 1);
```
**Matrices:** *set(matrix, value, i, j)* and *get(matrix, i, j)* where *i* is the column and *j* is the row, like x and y in cartesian coordinates.
```c
Matrix* matrix = create_matrix(3, 2);

set(matrix, 4.0, 0, 0); set(matrix, 5.0, 1, 0); set(matrix, 6.0, 2, 0);
set(matrix, 7.0, 0, 1); set(matrix, 8.0, 1, 1); set(matrix, 9.0, 2, 1);

double value = get(matrix, 0, 1);
```

**Rank 3 tensors:** *set(tensor, value, i, j, k)* and *get(matrix, i, j, k)* where *i* is the column, *j* is the row and *k* is the depth.
```c
Tensor* tensor = create_tensor_rank3(3, 3, 3);

set(tensor, 2.50, 0, 0, 0);
set(tensor, 3.25, 2, 0, 1);
```
**Rank 4 tensors:** *set(tensor, value, i, j, k, t)* and *get(matrix, i, j, k, t)* where *i* is the column, *j* is the row, *k* is the depth and *t* is the time.

```c
Tensor* tensor = create_tensor_rank4(3, 2, 3, 4);

set(tensor, 4.0, 0, 0, 0, 0);
set(tensor, 5.5, 2, 0, 1, 2);

double value = get(tensor2, 2, 0, 1, 2);
```

## Addition and subtraction
It is possible to sum and subtract tensors to tensors and scalars to tensors:

```c
void sum(Tensor* lhs, Tensor* rhs);
```
```c
void subtract(Tensor* lhs, Tensor* rhs);
```
```c
void sum_scalar(Tensor* lhs, double rhs);
```

**Sum two tensors:** add the rhs tensor to the lhs tensor.
```c
Tensor* tensor1 = create_tensor_rank3(3, 3, 3);

set(tensor1, 2.5, 0, 0, 0);
set(tensor1, 3.25, 2, 0, 1);

Tensor* tensor2 = create_tensor_rank3(3, 3, 3);

set(tensor2, 3.25, 0, 0, 0);
set(tensor2, 2.5, 2, 0, 1);

sum(tensor1, tensor2);

print_tensor(tensor1);

destroy_tensor(tensor1);
destroy_tensor(tensor2);
```

**Subtract two tensors:** subtracts the rhs tensor to the lhs tensor.
```c
Tensor* tensor1 = create_tensor_rank3(3, 3, 3);

set(tensor1, 2.5, 0, 0, 0);
set(tensor1, 3.25, 2, 0, 1);

Tensor* tensor2 = create_tensor_rank3(3, 3, 3);

set(tensor2, 3.25, 0, 0, 0);
set(tensor2, 2.5, 2, 0, 1);

subtract(tensor1, tensor2);

print_tensor(tensor1);

destroy_tensor(tensor1);
destroy_tensor(tensor2);
```

**Sum scalar to tensor:** sums a scalar to a tensor.
```c
Matrix* matrix = create_identity(3);

sum_scalar(matrix, 2.0);
```

## Product

> **Info:** *product* is only available for matrices for the moment.

Multiplies the rhs matrix to the lhs matrix:

```c
void product(Matrix* lhs, Matrix* rhs);
```
```c
void product_scalar(Tensor* lhs, double rhs);
```
**Product:** multiplies two matrices.
```c
Matrix* matrix1 = create_matrix(2, 3);

set(matrix1, 1.0, 0, 0); set(matrix1, 2.0, 1, 0);
set(matrix1, 3.0, 0, 1); set(matrix1, 4.0, 1, 1);
set(matrix1, 5.0, 0, 2); set(matrix1, 6.0, 1, 2);

Matrix* matrix2 = create_matrix(3, 2);

set(matrix2, 1.0, 0, 0); set(matrix2, 2.0, 1, 0); set(matrix2, 3.0, 2, 0);
set(matrix2, 4.0, 0, 1); set(matrix2, 5.0, 1, 1); set(matrix2, 6.0, 2, 1);

product(matrix1, matrix2);

print_tensor(matrix1);
printf("\n");
```
**Multiply scalar to tensor:** multiplies a scalar to a tensor.
```c
Tensor* tensor = create_tensor_rank3(4, 4, 4);

product_scalar(tensor, 1.5);
```

# Hadamard product
Computes the hadamard product of the rhs tensor to the lhs tensor:

```c
void hadamard_product(Tensor* lhs, Tensor* rhs);
```

```c
Vector* vector1 = create_vector(3);

set(vector1, 1.0, 0);
set(vector1, 2.0, 1);
set(vector1, 3.0, 2);

Vector* vector2 = create_vector(3);

set(vector2, 3.0, 0);
set(vector2, 2.0, 1);
set(vector2, 1.0, 2);

hadamard_product(vector1, vector2);

print_tensor(vector1);

```

# Tensor product
Computes the tensor product of the rhs tensor to the lhs tensor:
```c
Tensor* tensor_product(Tensor* lhs, Tensor* rhs);
```

```c
Matrix* matrix1 = create_matrix(2, 2);

set(matrix1, 1.0, 0, 0); set(matrix1, 2.0, 1, 0);
set(matrix1, 3.0, 0, 1); set(matrix1, 4.0, 1, 1);

printf("Matrix1:\n");
print_tensor(matrix1);
printf("\n");

Matrix* matrix2 = create_matrix(2, 2);

set(matrix2, 5.0, 0, 0); set(matrix2, 6.0, 1, 0);
set(matrix2, 7.0, 0, 1); set(matrix2, 8.0, 1, 1);

printf("Matrix2:\n");
print_tensor(matrix2);
printf("\n");

Tensor* tensor = tensor_product(matrix1, matrix2);

printf("Rank 4 tensor:\n");
print_tensor(tensor);

destroy_tensor(matrix1);
destroy_tensor(matrix2);
destroy_tensor(tensor);
```

# Dot product
Computes the dot product of the rhs tensor to the lhs tensor.

```c
double dot_product(Tensor* lhs, Tensor* rhs);
```

```c
Vector* vector1 = create_vector(3);

set(vector1, 1.0, 0);
set(vector1, 2.0, 1);
set(vector1, 3.0, 2);

Vector* vector2 = create_vector(3);

set(vector2, 3.0, 0);
set(vector2, 2.0, 1);
set(vector2, 1.0, 2);

double dot = dot_product(vector1, vector2);
```

# Cross product
> **Info:** cross product is only defined for R^3 vectors.

Computes the cross product of two vectors:
```c
Vector* cross_product(Vector* lhs, Vector* rhs);
```
```c
Vector* vector1 = create_vector(3);

set(vector1, 1.0, 0);
set(vector1, 2.0, 1);
set(vector1, 3.0, 2);

Vector* vector2 = create_vector(3);

set(vector2, 3.0, 0);
set(vector2, 2.0, 1);
set(vector2, 1.0, 2);

Vector* cross = cross_product(vector1, vector2);
```


# Transpose
>**Info:** transpose is only available for matrices for the moment.

Computes the transpose of a matrix:

```c
void transpose(Matrix* matrix);
```

```c
Matrix* matrix = create_matrix(3, 2);

set(matrix, 4.0, 0, 0); set(matrix, 5.0, 1, 0); set(matrix, 6.0, 2, 0);
set(matrix, 7.0, 0, 1); set(matrix, 8.0, 1, 1); set(matrix, 9.0, 2, 1);

transpose(matrix);

print_tensor(matrix);
printf("\n");
```

# Cofactor matrix
Computes the cofactor matrix of a matrix:

```c
Matrix* cofactor_matrix(Matrix* matrix);
```

# Adjugate matrix
Computes the adjugate matrix of a matrix:

```c
Matrix* adjugate_matrix(Matrix* matrix);
```

# Determinant
Computes the determinant of a matrix:

```c
double determinant(Matrix* matrix);
```

```c
Matrix* matrix = create_matrix(3, 3);

set(matrix, 1.0, 0, 0); set(matrix, 2.0, 1, 0); set(matrix,  3.0, 2, 0);
set(matrix, 0.0, 0, 1); set(matrix, 3.0, 1, 1); set(matrix, -2.0, 2, 1);
set(matrix, 7.0, 0, 2); set(matrix, 1.0, 1, 2); set(matrix,  4.0, 2, 2);

double det = determinant(matrix);
```

# Inverse
>**Info:** inverse is only available for matrices for the moment.

Computes the inverse of a matrix:

```c
Matrix* inverse(Matrix* matrix);
```

```c
Matrix* matrix = create_matrix(3, 3);

set(matrix, 1.0, 0, 0); set(matrix, 2.0, 1, 0); set(matrix,  3.0, 2, 0);
set(matrix, 0.0, 0, 1); set(matrix, 3.0, 1, 1); set(matrix, -2.0, 2, 1);
set(matrix, 7.0, 0, 2); set(matrix, 1.0, 1, 2); set(matrix,  4.0, 2, 2);

Matrix* inv = inverse(matrix);

print_tensor(inv);
printf("\n");

destroy_tensor(inv);
destroy_tensor(matrix);
```

# Transform vector
Transforms a vector by a matrix:

```c
void transform(Vector* vector, Matrix* matrix);
```

```c
Vector* vector = create_vector(4);

set(vector, 1.0, 0);
set(vector, 2.0, 1);
set(vector, 3.0, 2);
set(vector, 1.0, 3);

Matrix* matrix = create_identity(4);
product_scalar(matrix, 2.0);
set(matrix, 1.0, 3, 3);

transform(vector, matrix);

print_tensor(vector);
```