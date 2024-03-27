/*
    MIT License

    Copyright (c) 2024 Alberto Morcillo Sanz

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

    tensor (tensor algebra library in a single header file in C) 
    - Use at least C11 (-std=c11)
*/

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>

struct Tensor {
    int* shape;
    double* data;
    uint8_t rank;
};

typedef struct Tensor Vector;
typedef struct Tensor Matrix;
typedef struct Tensor Tensor;

/**
 * @brief Creates a tensor.
 * 
 * @param rank rank of the tensor.
 * @param .... the shape of the tensor.
 * 
 * @return The tensor.
*/
Tensor* create_tensor(uint8_t rank, ...);

/**
 * @brief Creates a tensor.
 * 
 * @param rank rank of the tensor.
 * @param shape the shape of the tensor.
 * 
 * @return The tensor.
*/
Tensor* create_tensor_byptr(uint8_t rank, int* shape);

/**
 * @brief Creates a vector.
 * 
 * @param n n number of elements.
 * 
 * @return The vector.
*/
Vector* create_vector(int n);

/**
 * @brief Creates a matrix.
 * 
 * @param rows number of rows of the matrix.
 * @param cols number of cols of the matrix.
 * 
 * @return The matrix.
*/
Matrix* create_matrix(uint8_t rows, uint8_t cols);

/**
 * @brief Creates a identity matrix.
 * 
 * @param n number of rows and cols of the matrix.
 * 
 * @return The matrix.
*/
Matrix* create_indentity(uint8_t n);

/**
 * @brief Creates a rank 3 tensor.
 * 
 * @param rows number of rows of the tensor.
 * @param cols number of cols of the tensor.
 * @param depth depth of the tensor.
 * 
 * @return The tensor.
*/
Tensor* create_tensor_rank3(uint8_t rows, uint8_t cols, uint8_t depth);

/**
 * @brief Creates a copy of a tensor.
 * 
 * @param tensor the tensor.
 * 
 * @return The copy of the tensor.
*/
Tensor* create_copy(Tensor* tensor);

/**
 * @brief Sets a value at a given position of the tensor.
 * 
 * @param value the value.
 * @param .... the positions.
*/
void set_value(Tensor* tensor, double value, ...);

/**
 * @brief Returns the value at a given position of the tensor.
 * 
 * @param .... the positions.
 * 
 * @return The value.
*/
double get_value(Tensor* tensor, ...);

/**
 * @brief Sets a value at a given position of the tensor.
 * 
 * @param value the value.
 * @param positions the positions.
*/
void set_byptr(Tensor* tensor, double value, int* positions);

/**
 * @brief Returns the value at a given position of the tensor.
 * 
 * @param positions the positions.
 * 
 * @return The value.
*/
double get_byptr(Tensor* tensor, int* positions);

/**
 * @brief Returns the length of the data of the tensor.
 * 
 * @param tensor the tensor.
 * 
 * @return The length;
*/
unsigned int get_length(Tensor* tensor);

/**
 * @brief Computes the transpose of a tensor. The dimensions
 * dim0 and dim1 are swapped.
 * 
 * @param tensor the tensor.
 * @param dim0 dimension 0
 * @param dim1 dimension 1
*/
void transpose_tensor(Tensor* tensor, int dim0, int dim1);

/**
 * @brief Computes the transpose of a matrix.
 * 
 * @param matrix the matrix.
 */
void transpose(Matrix* matrix);

/**
 * @brief Sums a scalar to a tensor.
 * 
 * @param tensor the tensor.
 * @param scalar the scalar.
*/
void sum_scalar(Tensor* tensor, double scalar);

/**
 * @brief Multiplies a scalar to a tensor.
 * 
 * @param tensor the tensor.
 * @param scalar the scalar.
*/
void product_scalar(Tensor* tensor, double scalar);

/**
 * @brief Sums two tensors.
 * @warning The result will be updated in the left matrix.
 * 
 * @param lhs left tensor.
 * @param rhs right tensor.
*/
void sum_tensors(Tensor* lhs, Tensor* rhs);

/**
 * @brief Multiplies the rhs matrix to the lhs matrix.
 * @warning The result will be updated in the left matrix.
 * 
 * @param lhs left matrix.
 * @param rhs right matrix.
*/
void matmul(Matrix* lhs, Matrix* rhs);

/**
 * @brief Multiplies (Hadamard) the rhs tensor to the lhs tensor.
 * @warning The result will be updated in the left tensor.
 * 
 * @param lhs left tensor.
 * @param rhs right tensor.
 * 
 * @pre lhs and rhs must be the same rank.
*/
void hadamard_product(Tensor* lhs, Tensor* rhs);

/**
 * @brief Computes the tensor product between lhs tensor and rhs tensor.
 * @warning The result will be updated in the left tensor.
 * 
 * @param lhs left tensor.
 * @param rhs right tensor.
 * 
 * @return The tensor product.
*/
Tensor* tensor_product(Tensor* lhs, Tensor* rhs);

/**
 * @brief Computes the dyadic product between two vectors.
 * @warning The result will be updated in the left vector.
 * 
 * @param lhs left vector.
 * @param rhs right vector.
 * 
 * @return The dyadic product.
*/
Tensor* dyadic_product(Vector* lhs, Vector* rhs);

/**
 * @brief Calculates the dot product between lhs and rhs tensors.
 * 
 * @param lhs left tensor.
 * @param rhs right tensor.
 * 
 * @return The dot product.
*/
double dot_product(Tensor* lhs, Tensor* rhs);

/**
 * @brief Computes the cross product between lhs and rhs vectors.
 * 
 * @param lhs left vector.
 * @param rhs right vector.
 * 
 * @pre lhs and rhs must be R^3 vectors.
 * 
 * @return The cross product.
*/
Vector* cross_product(Vector* lhs, Vector* rhs);

/**
 * @brief Transforms a vector by a transformation matrix.
 * 
 * @param vector vector.
 * @param matrix transformation matrix.
*/
void transform(Vector* vector, Matrix* matrix);

/**
 * @brief Calculates the minor of a matrix in a position (row,col) of the matrix.
 * 
 * @param matrix the matrix.
 * @param row the row index.
 * @param col the column index.
 * 
 * @return The minor.
 */
double minor(Matrix* matrix, uint8_t row, uint8_t col);

/**
 * @brief Calculates the cofactor of a matrix in a position (row,col) of the matrix.
 * 
 * @param matrix the matrix.
 * @param row the row index.
 * @param col the column index.
 * 
 * @return The cofactor.
 */
double cofactor(Matrix* matrix, uint8_t row, uint8_t col);

/**
 * @brief Computes the cofactor matrix of a matrix.
 * 
 * @param matrix the matrix.
 * 
 * @return The cofactor matrix.
*/
Matrix* cofactor_matrix(Matrix* matrix);

/**
 * @brief Computes the adjugate matrix of a matrix.
 * 
 * @param matrix the matrix.
 * 
 * @return The adjugate matrix
*/
Matrix* adjugate_matrix(Matrix* matrix);

/**
 * @brief Calculates the determinant of a matrix.
 * 
 * @param matrix the matrix.
 * 
 * @return The determinant.
*/
double determinant(Matrix* matrix);

/**
 * @brief Computes the inverse of a matrix.
 * 
 * @param matrix the matrix.
 * 
 * @return The inverse.
*/
Matrix* inverse(Matrix* matrix);

/**
 * @brief Prints a vector.
 * 
 * @param vector the vector.
*/
void print_vector(Vector* vector);

/**
 * @brief Prints a matrix.
 * 
 * @param matrix the matrix.
*/
void print_matrix(Matrix* matrix);

/**
 * @brief Prints a rank 3 tensor.
 * 
 * @param tensor the tensor.
*/
void print_tensor_rank3(Tensor* tensor);

/**
 * @brief Prints a rank 4 tensor.
 * 
 * @param tensor the tensor.
*/
void print_tensor_rank4(Tensor* tensor);

/**
 * @brief Prints a tensor.
 * @deprecated
 * 
 * @param tensor the tensor.
*/
void print_tensor(Tensor* tensor);

/**
 * @brief Destroys a tensor.
 * 
 * @param tensor the tensor.
*/
void destroy_tensor(Tensor* tensor);

/*---------------------------------------*/

Tensor* create_tensor(uint8_t rank, ...) {

    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));

    va_list args;
    va_start(args, rank);

    size_t length = 1;
    int* shape = (int*)malloc(sizeof(int) * rank);

    for(int i = 0; i < rank; i ++) {
        int s = va_arg(args, int);
        length *= s;
        shape[i] = s;
    }

    va_end(args);

    tensor->rank = rank;
    tensor->shape = shape;
    tensor->data = (double*)malloc(sizeof(double) * length);
    for(size_t i = 0; i < length; i ++) tensor->data[i] = 0.0;

    return tensor;
}

Tensor* create_tensor_byptr(uint8_t rank, int* shape) {

    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));

    size_t length = 1;
    for(int i = 0; i < rank; i ++) length *= shape[i];

    tensor->rank = rank;
    tensor->shape = shape;
    tensor->data = (double*)malloc(sizeof(double) * length);
    for(size_t i = 0; i < length; i ++) tensor->data[i] = 0.0;

    return tensor;
}

Vector* create_vector(int n) {
    Vector* vector = create_tensor(1, n);
    return vector;
}

Matrix* create_matrix(uint8_t rows, uint8_t cols) {
    Matrix* matrix = create_tensor(2, rows, cols);
    return matrix;
}

Matrix* create_indentity(uint8_t n) {

    Matrix* matrix = create_matrix(n, n);

    for(int c = 0; c < n; c++) {
        for(int r = 0; r < n; r ++) {
            if(c == r)
                set_value(matrix, 1.0, c, r);
        }
    }

    return matrix;
}

Tensor* create_tensor_rank3(uint8_t rows, uint8_t cols, uint8_t depth) {
    Tensor* tensor = create_tensor(3, rows, cols, depth);
    return tensor;
}

Tensor* create_copy(Tensor* tensor) {

    Tensor* copy_tensor = (Tensor*)malloc(sizeof(Tensor));

    int* shape = (int*)malloc(sizeof(int) * tensor->rank);
    for(int i = 0; i < tensor->rank; i ++)
        shape[i] = tensor->shape[i];

    copy_tensor->rank = tensor->rank;
    copy_tensor->shape = shape;
    
    size_t length = 1;
    for(int i = 0; i < tensor->rank; i ++)
        length *= tensor->shape[i];

    copy_tensor->data = (double*)malloc(sizeof(double) * length);
    for(int i = 0; i < length; i ++)
        copy_tensor->data[i] = tensor->data[i];

    return copy_tensor;
}

void set_value(Tensor* tensor, double value, ...) {

    va_list args;
    va_start(args, value);

    int index = 0;
    for(int i = 0; i < tensor->rank; i ++) {

        int subIndex = va_arg(args, int);
        for(int j = 0; j < i; j ++) {
            subIndex *= tensor->shape[j];
        }

        index += subIndex;
    }

    tensor->data[index] = value;
    va_end(args);
}

double get_value(Tensor* tensor, ...) {

    va_list args;
    va_start(args, tensor);

    int index = 0;
    for(int i = 0; i < tensor->rank; i ++) {

        int subIndex = va_arg(args, int);
        for(int j = 0; j < i; j ++) {
            subIndex *= tensor->shape[j];
        }

        index += subIndex;
    }

    double value = tensor->data[index];
    va_end(args);

    return value;
}

void set_byptr(Tensor* tensor, double value, int* positions) {

    int index = 0;
    for(int i = 0; i < tensor->rank; i ++) {

        int subIndex = positions[i];
        for(int j = 0; j < i; j ++) {
            subIndex *= tensor->shape[j];
        }

        index += subIndex;
    }

    tensor->data[index] = value;
}

double get_byptr(Tensor* tensor, int* positions) {

    int index = 0;
    for(int i = 0; i < tensor->rank; i ++) {

        int subIndex = positions[i];
        for(int j = 0; j < i; j ++) {
            subIndex *= tensor->shape[j];
        }

        index += subIndex;
    }

    return tensor->data[index];
}

unsigned int get_length(Tensor* tensor) {

    int length = 1;
    for(int i = 0; i < tensor->rank; i ++) 
        length *= tensor->shape[i];

    return length;
}

// Generated by ChatGPT :D
int** get_positions(int *shape, int dimensions) {
    // Calcular el número total de posiciones
    int total_positions = 1;
    for (int i = 0; i < dimensions; i++) {
        total_positions *= shape[i];
    }

    // Crear un array 2D para almacenar las posiciones
    int **positions = (int **)malloc(total_positions * sizeof(int *));
    for (int i = 0; i < total_positions; i++) {
        positions[i] = (int *)malloc(dimensions * sizeof(int));
    }

    // Generar las posiciones
    int *multipliers = (int *)malloc(dimensions * sizeof(int));
    multipliers[dimensions - 1] = 1;
    for (int i = dimensions - 2; i >= 0; i--) {
        multipliers[i] = multipliers[i + 1] * shape[i + 1];
    }

    for (int i = 0; i < total_positions; i++) {
        int remainder = i;
        for (int j = 0; j < dimensions; j++) {
            positions[i][j] = remainder / multipliers[j];
            remainder %= multipliers[j];
        }
    }

    free(multipliers);
    return positions;
}

void transpose_tensor(Tensor* tensor, int dim0, int dim1) {

    size_t length = 1;
    int* shape = (int*)(malloc(sizeof(int) * tensor->rank));

    for(int i = 0; i < tensor->rank; i ++) {
        if(i == dim0)       shape[dim0] = tensor->shape[dim1];
        else if(i == dim1)  shape[dim1] = tensor->shape[dim0];
        else                shape[i] = tensor->shape[i];
        length *= tensor->shape[i];
    }

    Tensor* transpose_tensor = create_tensor_byptr(tensor->rank, shape);

    int** positions = get_positions(shape, tensor->rank);

    for(int i = 0; i < length; i ++) {

        int* pos = positions[i];
        int* initial_pos = (int*)malloc(sizeof(int) * tensor->rank);

        for(int i = tensor->rank - 1; i >= 0; i --) {

            if(i == dim0)       initial_pos[dim0] = pos[dim1];
            else if(i == dim1)  initial_pos[dim1] = pos[dim0];
            else                initial_pos[i] = pos[i];
        }
            
        double value = get_byptr(tensor, initial_pos);
        set_byptr(transpose_tensor, value, pos);

        free(pos);
        free(initial_pos);
    }

    free(positions);

    for(int i = 0; i < tensor->rank; i ++)
        tensor->shape[i] = transpose_tensor->shape[i];

    for(int i = 0; i < length; i ++)
        tensor->data[i] = transpose_tensor->data[i];

    destroy_tensor(transpose_tensor);
}

void transpose(Matrix* matrix) {
    transpose_tensor(matrix, 0, 1);
}

void sum_scalar(Tensor* tensor, double scalar) {

    size_t length = get_length(tensor);

    for(int i = 0; i < length; i ++)
        tensor->data[i] += scalar;
}

void product_scalar(Tensor* tensor, double scalar) {

    size_t length = get_length(tensor);

    for(int i = 0; i < length; i ++)
        tensor->data[i] *= scalar;
}

void sum_tensors(Tensor* lhs, Tensor* rhs) {

    if(lhs->rank != rhs->rank)
        return;

    size_t length = 1;
    for(int i = 0; i < lhs->rank; i ++) {

        if(lhs->shape[i] != rhs->shape[i])
            return;

        length *= lhs->shape[i];
    }

    for(int i = 0; i < length; i ++)
        lhs->data[i] += rhs->data[i];
}

void matmul(Matrix* lhs, Matrix* rhs) {

    if(lhs->shape[1] != rhs->shape[0])
        return;

    Matrix* result = create_matrix(rhs->shape[1], lhs->shape[0]);

    for(uint8_t r = 0; r < lhs->shape[0]; r ++) {
        for(uint8_t c = 0; c < rhs->shape[1]; c ++) {

            double mir = 0.0;
            for(int k = 0; k < lhs->shape[1]; k ++)
                mir += get_value(rhs, k, r) * get_value(lhs, c, k);

            set_value(result, mir, r, c);
        }
    }

    free(lhs->data);
    lhs->shape[1] = rhs->shape[1];
    lhs->data = (double*)malloc(sizeof(double) * lhs->shape[0] * lhs->shape[1]);

    for(uint8_t i = 0; i < lhs->shape[0] * lhs->shape[1]; i ++)
        lhs->data[i] = result->data[i];

    destroy_tensor(result);
}

void hadamard_product(Tensor* lhs, Tensor* rhs) {

    if(lhs->rank != rhs->rank)
        return;

    size_t length = 1;
    for(int i = 0; i < lhs->rank; i ++) {

        if(lhs->shape[i] != rhs->shape[i])
            return;

        length *= lhs->shape[i];
    }

    for(size_t i = 0; i < length; i ++)
        lhs->data[i] *= rhs->data[i];
}

Tensor* tensor_product(Tensor* lhs, Tensor* rhs) {

    Tensor* temp = lhs;
    lhs = rhs;
    rhs = temp;

    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    
    tensor->rank = lhs->rank + rhs->rank;
    tensor->shape = (int*)malloc(sizeof(int) * tensor->rank);

    for(int i = 0; i < tensor->rank; i ++) {
        if(i < lhs->rank) tensor->shape[i] = lhs->shape[i];
        else tensor->shape[i] = rhs->shape[i - lhs->rank];
    }

    int length_lhs = get_length(lhs);
    int length_rhs = get_length(rhs);

    tensor->data = (double*)malloc(sizeof(double) * length_lhs * length_rhs);

    int** lhs_positions = get_positions(lhs->shape, lhs->rank);
    for(int i = 0; i < length_lhs; i ++) {

        int* lhs_position = lhs_positions[i];

        Tensor* subtensor = create_copy(rhs);
        product_scalar(subtensor, get_byptr(lhs, lhs_position));

        int** rhs_positions = get_positions(rhs->shape, rhs->rank);
        for(int j = 0; j < length_rhs; j ++) {

            int* rhs_position = rhs_positions[j];

            int* pos = (int*)malloc(sizeof(int) * (lhs->rank + rhs->rank));
            for(int k = 0; k < lhs->rank + rhs->rank; k ++) {
                if(k < lhs->rank) pos[k] = lhs_position[k];
                else pos[k] = rhs_position[k - lhs->rank];
            }

            double value = get_byptr(subtensor, rhs_position);
            set_byptr(tensor, value, pos);

            free(rhs_position);
            free(pos);
        }

        destroy_tensor(subtensor);

        free(lhs_position);
        free(rhs_positions);
    }

    free(lhs_positions);
    return tensor;
}

Tensor* dyadic_product(Vector* lhs, Vector* rhs) {
    
    Tensor* product = tensor_product(lhs, rhs);
    transpose(product);

    return product;
}

double dot_product(Tensor* lhs, Tensor* rhs) {

    if(lhs->rank != rhs->rank)
        return 0.0;

    size_t length = 1;
    for(int i = 0; i < lhs->rank; i ++) {

        if(lhs->shape[i] != rhs->shape[i])
            return 0.0;

        length *= lhs->shape[i];
    }

    double sum = 0.0;
    for(size_t i = 0; i < length; i ++)
        sum += lhs->data[i] * rhs->data[i];

    return sum;
}

Vector* cross_product(Vector* lhs, Vector* rhs) {

    Vector* cross = create_vector(3);

    if(lhs->shape[0] != 3 && rhs->shape[0] != 3)
        return cross;

    cross->data[0] =  lhs->data[1] * rhs->data[2] - lhs->data[2] * rhs->data[1];
    cross->data[1] = -lhs->data[0] * rhs->data[2] + lhs->data[2] * rhs->data[0];
    cross->data[2] =  lhs->data[0] * rhs->data[1] - lhs->data[1] * rhs->data[0];

    return cross;
}

void transform(Vector* vector, Matrix* matrix) {

    if(matrix->shape[1] != vector->shape[0])
        return;

    double* values = (double*)malloc(sizeof(double) * matrix->shape[0]);

    for(int r = 0; r < matrix->shape[0]; r ++) {

        Vector* row_vector = create_vector(matrix->shape[1]);

        for(int c = 0; c < matrix->shape[1]; c ++) {
            double value = get_value(matrix, r, c);
            set_value(row_vector, value, c);
        }
        
        double value = dot_product(row_vector, vector);
        values[r] = value;

        destroy_tensor(row_vector);
    }

    vector->shape[0] = matrix->shape[0];
    vector->data = (double*)realloc(vector->data, vector->shape[0]);

    for(int c = 0; c < vector->shape[0]; c++) 
        set_value(vector, values[c], c);

    free(values);
}

double minor(Matrix* matrix, uint8_t row, uint8_t col) {

    Matrix* sub_matrix = create_matrix(matrix->shape[1] - 1, matrix->shape[1] - 1);

    uint8_t index = 0;
    for(uint8_t r = 0; r < matrix->shape[0]; r ++) {
        for(uint8_t c = 0; c < matrix->shape[1]; c ++) {

            if(row != r && col != c) {
                double value = get_value(matrix, r, c);
                sub_matrix->data[index] = value;
                index ++;
            }
        }
    }

    double det = determinant(sub_matrix);
    destroy_tensor(sub_matrix);

    return det;
}

double cofactor(Matrix* matrix, uint8_t row, uint8_t col) {

    int sign = pow(-1.0, row + col);
    double result =  sign * minor(matrix, row, col);
    return result;
}

Matrix* cofactor_matrix(Matrix* matrix) {

    Matrix* cof_matrix = create_matrix(matrix->shape[0], matrix->shape[1]);

    for(uint8_t r = 0; r < matrix->shape[0]; r ++) {
        for(uint8_t c = 0; c < matrix->shape[1]; c ++)
            set_value(cof_matrix, cofactor(matrix, r, c), r, c);
    }

    return cof_matrix;
}

Matrix* adjugate_matrix(Matrix* matrix) {

    Matrix* cof_matrix = cofactor_matrix(matrix);
    transpose(cof_matrix);

    return cof_matrix;
}

double determinant(Matrix* matrix) {

    if(matrix->shape[0] != matrix->shape[1])
        return 0.0;

    if(matrix->shape[1] == 2)
        return get_value(matrix, 0, 0) * get_value(matrix, 1, 1) - get_value(matrix, 1, 0) * get_value(matrix, 0, 1);

    double result = 0.0;

    for(uint8_t r = 0; r < matrix->shape[0]; r ++) {
        double cof = cofactor(matrix, r, 0);
        result += get_value(matrix, r, 0) * cof; 
    }

    return result;
}

Matrix* inverse(Matrix* matrix) {

    double det = determinant(matrix);
    Matrix* inv = adjugate_matrix(matrix);
    double inv_determinant = 1 / det;
    
    product_scalar(inv, inv_determinant);

    return inv;
}

void print_vector(Vector* vector) {

    if(vector->rank != 1) {
        printf("Vector rank %d. Expected rank 1.\n", vector->rank);
        return;
    }

    printf("(");
    for(int r = 0; r < vector->shape[0]; r ++) {
        if(r < vector->shape[0] - 1) printf("%f,", get_value(vector, r));
        else printf("%f)\n", get_value(vector, r));
    }
}

void print_matrix(Matrix* matrix) {

    if(matrix->rank != 2) {
        printf("Matrix rank %d. Expected rank 2.\n", matrix->rank);
        return;
    }

    for(int r = 0; r < matrix->shape[0]; r ++) {
        for(int c = 0; c < matrix->shape[1]; c ++) {
            printf("%f ", get_value(matrix, r, c));
        }
        printf("\n");
    }
}

void print_tensor_rank3(Tensor* tensor) {

    if(tensor->rank != 3) {
        printf("Tensor rank %d. Expected rank 3.\n", tensor->rank);
        return;
    }

    for(int d = 0; d < tensor->shape[2]; d ++) {
        for(int r = 0; r < tensor->shape[0]; r ++) {
            for(int c = 0; c < tensor->shape[1]; c ++) {
                printf("%f ", get_value(tensor, r, c, d));
            }
            printf("\n");
        }
        printf("\n");
    }
}

void print_tensor_rank4(Tensor* tensor) {

    if(tensor->rank != 4) {
        printf("Tensor rank %d. Expected rank 4.\n", tensor->rank);
        return;
    }

    for(uint8_t t = 0; t < tensor->shape[3]; t ++) {
        for(uint8_t r = 0; r < tensor->shape[0]; r ++) {

            for(uint8_t d = 0; d < tensor->shape[2]; d ++) {
                for(uint8_t c = 0; c < tensor->shape[1]; c ++) {

                    printf("%f ", get_value(tensor, c, r, d, t));
                }
                printf("\t");
            }
            printf("\n");
        }
        printf("\n");
    }
}

void print_tensor(Tensor* tensor) {

    int length = 1;

    printf("Tensor shape = (");
    for(int i = 0; i < tensor->rank; i ++) {
        length *= tensor->shape[i];

        if(i < tensor->rank - 1) printf("%d,", tensor->shape[i]);
        else printf("%d)\n", tensor->shape[i]);
    }

    printf("Data (column-major order) = [");
    for(int i = 0; i < length; i ++) {
        if(i < length - 1) printf("%f,", tensor->data[i]);
        else printf("%f]\n", tensor->data[i]);
    }
}

void destroy_tensor(Tensor* tensor) {
    free(tensor->shape);
    free(tensor->data);
    free(tensor);
}