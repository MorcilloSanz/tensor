# tensor

Tensor algebra library in a single header file in C11 for `graphics programming`, `machine` and `deep learning` and `physics`.
> Still under development.

<style>
td, th {
   border: none!important;
}
</style>

<table>
    <tr>
        <td>
            <ul>
                <li>Vectors, matrices, rank 3 and higher rank tensors.</li>
                <li>Linear algebra and tensor algebra operations.</li>
                <li>Pure C code, easy to integrate in C++ or other languajes using bindings.</li>
                <li>Only one header file.</li>
            </ul>
        <td>
        <td><img src="img/tensor.png"/></td>
    </tr>
</table>

Take a look at [tensor wiki](https://github.com/MorcilloSanz/tensor/wiki) for reading the docs.

## Transform vector example
```c
Matrix* matrix = create_indentity(4);
set_value(matrix, 2.0, 3, 0);
set_value(matrix, 3.0, 3, 1);
set_value(matrix, 4.0, 3, 2);

Vector* vector = create_vector(4);
set_value(vector, 1.0, 0);
set_value(vector, 1.0, 1);
set_value(vector, 1.0, 2);
set_value(vector, 1.0, 3);

transform(vector, matrix);
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
destroy_tensor(tensor);
```