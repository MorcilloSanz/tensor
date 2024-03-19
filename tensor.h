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
 * @param cols number of cols of the matrix.
 * @param rows number of rows of the matrix.
 * 
 * @return The matrix.
*/
Matrix* create_matrix(uint8_t cols, uint8_t rows);

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
 * @param cols number of cols of the tensor.
 * @param rows number of rows of the tensor.
 * @param depth depth of the tensor.
 * 
 * @return The tensor.
*/
Tensor* create_tensor_rank3(uint8_t cols, uint8_t rows, uint8_t depth);

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
 * @param lhs left tensor
 * @param rhs right tensor
 * 
 * @pre lhs and rhs must be the same rank.
 * @pre only rank 1 and 2 tensors allowed.
 * 
 * @return The tensor product.
*/
Tensor* tensor_product(Tensor* lhs, Tensor* rhs);

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
 * @brief Calculates the minor of a matrix in a position (i,j) of the matrix.
 * 
 * @param matrix the matrix.
 * @param i position i in (i,j).
 * @param j position j in (i,j).
 * 
 * @return The minor.
 */
double minor(Matrix* matrix, uint8_t i, uint8_t j);

/**
 * @brief Calculates the cofactor of a matrix in a position (i,j) of the matrix.
 * 
 * @param matrix the matrix.
 * @param i position i in (i,j).
 * @param j position j in (i,j).
 * 
 * @return The cofactor.
 */
double cofactor(Matrix* matrix, uint8_t i, uint8_t j);

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
 * @brief Prints a tensor.
 * @warning Rank 4 or less.
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

Vector* create_vector(int n) {
    Vector* vector = create_tensor(1, n);
    return vector;
}

Matrix* create_matrix(uint8_t cols, uint8_t rows) {
    Matrix* matrix = create_tensor(2, cols, rows);
    return matrix;
}

Matrix* create_indentity(uint8_t n) {

    Matrix* matrix = create_matrix(n, n);

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j ++) {
            if(i == j)
                set_value(matrix, 1.0, i, j);
        }
    }

    return matrix;
}

Tensor* create_tensor_rank3(uint8_t cols, uint8_t rows, uint8_t depth) {
    Tensor* tensor = create_tensor(3, cols, rows, depth);
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

void transpose(Matrix* matrix) {

    Matrix* transposeMatrix = create_matrix(matrix->shape[0], matrix->shape[1]);

    for(uint8_t i = 0; i < transposeMatrix->shape[1]; i ++) {
        for(uint8_t j = 0; j < transposeMatrix->shape[0]; j ++)
            set_value(transposeMatrix, get_value(matrix, j, i), i, j);
    }

    matrix->shape[0] = transposeMatrix->shape[0];
    matrix->shape[1] = transposeMatrix->shape[1];

    for(uint8_t i = 0; i < matrix->shape[0] * matrix->shape[1]; i ++)
        matrix->data[i] = transposeMatrix->data[i];

    destroy_tensor(transposeMatrix);
}

void sum_scalar(Tensor* tensor, double scalar) {

    size_t length = 1;
    for(int i = 0; i < tensor->rank; i ++)
        length *= tensor->shape[i];

    for(int i = 0; i < length; i ++)
        tensor->data[i] += scalar;
}

void product_scalar(Tensor* tensor, double scalar) {

    size_t length = 1;
    for(int i = 0; i < tensor->rank; i ++)
        length *= tensor->shape[i];

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

    for(uint8_t i = 0; i < rhs->shape[1]; i ++) {
        for(uint8_t j = 0; j < lhs->shape[0]; j ++) {

            double mij = 0.0;
            for(int k = 0; k < lhs->shape[1]; k ++)
                mij += get_value(rhs, i, k) * get_value(lhs, k, j);

            set_value(result, mij, i, j);
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

    if(lhs->rank == rhs->rank) {

        switch (lhs->rank) {
        case 1 :
        {
            Matrix* matrix = create_matrix(rhs->shape[0], lhs->shape[0]);

            for(uint8_t j = 0; j < matrix->shape[0]; j ++) {
                for(uint8_t i = 0; i < matrix->shape[1]; i ++) {
                
                    double value = get_value(lhs, j) * get_value(rhs, i);
                    set_value(matrix, value, i, j);
                }
            }

            return matrix;
        }    
        break;
        case 2:
        {
            Tensor* tensor = create_tensor(4, rhs->shape[1], rhs->shape[0], lhs->shape[1], lhs->shape[0]);

            for(uint8_t t = 0; t < tensor->shape[3]; t ++) {
                for(uint8_t k = 0; k < tensor->shape[2]; k ++) {

                    for(uint8_t j = 0; j < tensor->shape[0]; j ++) {
                        for(uint8_t i = 0; i < tensor->shape[1]; i ++) {
                        
                            double value = get_value(lhs, k, t) * get_value(rhs, i, j);
                            set_value(tensor, value, i, j, k, t);
                        }
                    }
                }
            }

            return tensor;
        }
        break;
        }
    }

    Vector* rank0 = create_vector(1);
    return rank0;
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

    Vector* vec_transformed = create_vector(vector->shape[0]);

    for(uint8_t j = 0; j < vector->shape[0]; j ++) {

        Vector* row_vector = create_vector(vector->shape[0]);

        for(uint8_t i = 0; i < matrix->shape[1]; i ++) {
            
            double value = get_value(matrix, i, j);
            set_value(row_vector, value, i);
        }

        vec_transformed->data[j] = dot_product(vector, row_vector);

        destroy_tensor(row_vector);
    }

    for(uint8_t j = 0; j < vector->shape[0]; j ++)
        vector->data[j] = vec_transformed->data[j];

    destroy_tensor(vec_transformed);
}

double minor(Matrix* matrix, uint8_t i, uint8_t j) {

    Matrix* sub_matrix = create_matrix(matrix->shape[1] - 1, matrix->shape[1] - 1);

    uint8_t index = 0;
    for(uint8_t row = 0; row < matrix->shape[0]; row ++) {
        for(uint8_t col = 0; col < matrix->shape[1]; col ++) {

            if(i != row && j != col) {
                double value = get_value(matrix, row, col);
                sub_matrix->data[index] = value;
                index ++;
            }
        }
    }

    double det = determinant(sub_matrix);
    destroy_tensor(sub_matrix);

    return det;
}

double cofactor(Matrix* matrix, uint8_t i, uint8_t j) {

    int sign = pow(-1.0, i + j);
    double result =  sign * minor(matrix, i, j);
    return result;
}

Matrix* cofactor_matrix(Matrix* matrix) {

    Matrix* cof_matrix = create_matrix(matrix->shape[1], matrix->shape[0]);

    for(uint8_t i = 0; i < matrix->shape[1]; i ++) {
        for(uint8_t j = 0; j < matrix->shape[0]; j ++)
            set_value(cof_matrix, cofactor(matrix, i, j), i, j);
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
        return get_value(matrix, 0, 0) * get_value(matrix, 1, 1) - get_value(matrix, 0, 1) * get_value(matrix, 1, 0);

    double result = 0.0;

    for(uint8_t j = 0; j < matrix->shape[1]; j ++) {
        double cof = cofactor(matrix, 0, j);
        result += get_value(matrix, 0, j) * cof; 
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

void print_tensor(Tensor* tensor) {

    int shape[4] = {1, 1, 1, 1};
    for(int i = 0; i < tensor->rank; i ++)
        shape[i] = tensor->shape[i];

    for(uint8_t t = 0; t < shape[3]; t ++) {
        for(uint8_t j = 0; j < shape[0]; j ++) {

            for(uint8_t k = 0; k < shape[2]; k ++) {
                for(uint8_t i = 0; i < shape[1]; i ++) {

                    if(tensor->rank == 1)
                        printf("%f ", get_value(tensor, j));
                    else
                        printf("%f ", get_value(tensor, i, j, k, t));
                }
                printf("\t");
            }
            printf("\n");
        }
        printf("\n");
    }
}

void destroy_tensor(Tensor* tensor) {
    free(tensor->shape);
    free(tensor->data);
    free(tensor);
}