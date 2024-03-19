# tensor

Tensor algebra library in a single header file in C11 for `graphics programming`, `machine` and `deep learning` and `physics`.

* Vectors, matrices and rank 3 and higher rank tensors.
* Linear algebra and tensor algebra operations.

Take a look at [tensor wiki](https://github.com/MorcilloSanz/tensor/wiki) for reading the docs.

## Transform vector example
```c
Matrix* matrix = create_indentity(4);
set_value(matrix, 2.0, 3, 0);
set_value(matrix, 3.0, 3, 1);
set_value(matrix, 4.0, 3, 2);

printf("Matrix:\n");
print_tensor(matrix);

Vector* vector = create_vector(4);
set_value(vector, 1.0, 0);
set_value(vector, 1.0, 1);
set_value(vector, 1.0, 2);
set_value(vector, 1.0, 3);

transform(vector, matrix);

printf("Vector:\n");
print_tensor(vector);

destroy_tensor(matrix);
destroy_tensor(vector);
```

## Inverse of a matrix example
```c
Matrix* matrix = create_indentity(4);
set_value(matrix, 2.0, 3, 0);
set_value(matrix, 3.0, 3, 1);
set_value(matrix, 4.0, 3, 2);

Matrix* inv = inverse(matrix);

matmul(matrix, inv);
print_tensor(matrix);

destroy_tensor(matrix);
destroy_tensor(inv);
```

## Rank 3 tensor example
```c
Tensor* tensor = create_tensor_rank3(3, 3, 3);
set_value(tensor, 1.0, 0, 0, 0);
set_value(tensor, 2.0, 1, 1, 1);
set_value(tensor, 3.0, 2, 2, 2);

print_tensor(tensor);
```