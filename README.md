# Talg (Tensor algebra library)

Tensor algebra library written in a single header file in C (-std=c11).

## Table of contents

- [Create algebraic structures](#create-algebraic-structures)
- [Create a copy of a tensor](#create-a-copy-of-a-tensor)
- [Create identity matrix](#create-identity-matrix)
- [Set and get values](#set-and-get-values)
- [Addition and subtraction](#addition-and-subtraction)
- [Product](#product)
- [Hadamard product](#hadamard-product)
- [Tensor product](#tensor-product)
- [Dot product](#dot-product)
- [Cross product](#cross-product)
- [Transpose](#transpose)
- [Inverse](#inverse)
- [Determinant](#determinant)
- [Transform vector](#transform-vector)

## Create algebraic structures
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

> **Important:** as these create methods allocates memory in the heap. It is important to free them once they are not useful anymore:

```c
void destroy_tensor(Tensor* tensor);
```

Example:

```c
Matrix* matrix = create_matrix(3, 3);

set(matrix, 1.0, 0, 0); set(matrix, 2.0, 1, 0); set(matrix,  3.0, 2, 0);
set(matrix, 0.0, 0, 1); set(matrix, 3.0, 1, 1); set(matrix, -2.0, 2, 1);
set(matrix, 7.0, 0, 2); set(matrix, 1.0, 1, 2); set(matrix,  4.0, 2, 2);

Matrix* inv = inverse(matrix);

print_matrix(inv);
printf("\n");

destroy_tensor(inv);
destroy_tensor(matrix);
```

*As vector and matrices are tensors of rank 1 and 2, it is possible to write Tensor instead of Vector or Matrix.*

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
Tensor* tensor1 = create_tensor_rank3(3, 3, 3);

set(tensor1, 2.50, 0, 0, 0);
set(tensor1, 3.25, 2, 0, 1);

Tensor* tensor2 = create_tensor_rank4(3, 2, 3, 4);

set(tensor2, 4.0, 0, 0, 0, 0);
set(tensor2, 5.5, 2, 0, 1, 2);

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
Matrix* matrix1 = create_matrix(3, 3);
...

Matrix* matrix2 = create_matrix(3, 3);
...

sum(matrix1, matrix2);
```

**Subtract two tensors:** subtracts the rhs tensor to the lhs tensor.
```c
Matrix* matrix1 = create_matrix(3, 3);
...

Matrix* matrix2 = create_matrix(3, 3);
...

subtract(matrix1, matrix2);
```

**Sum scalar to tensor:** sums a scalar to a tensor.
```c
Matrix* matrix1 = create_matrix(3, 3);
...

sum_scalar(matrix1, 2.0);
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

print_matrix(matrix1);
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
...

Vector* vector2 = create_vector(3);
...

hadamard_product(vector1, vector2);

printf("(%f,%f,%f)\n", get(vector1, 0), get(vector1, 1), get(vector1, 2));

```

# Tensor product
Computes the tensor product of the rhs tensor to the lhs tensor:
```c
Tensor* tensor_product(Tensor* lhs, Tensor* rhs);
```

```c
Vector* vector1 = create_vector(3);
...

Vector* vector2 = create_vector(4);
...

Tensor* result = tensor_product(vector1, vector2);

print_matrix(result);
```

# Dot product
Computes the dot product of the rhs tensor to the lhs tensor.

```c
double dot_product(Tensor* lhs, Tensor* rhs);
```

```c
Vector* vector1 = create_vector(3);
...

Vector* vector2 = create_vector(3);
...

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
...

Vector* vector2 = create_vector(3);
...

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

print_matrix(matrix);
printf("\n");
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

print_matrix(inv);
printf("\n");
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

# Transform vector
Transforms a vector by a matrix:

```c
void transform(Vector* vector, Matrix* matrix);
```

```c
Vector* vector = create_vector(4);
...

Matrix* matrix = create_matrix(4, 4);

transform(vector, matrix);
```