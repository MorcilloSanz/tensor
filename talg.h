/*
    MIT License

    Copyright (c) 2023 Alberto Morcillo Sanz

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

    Talg (tensor algebra library in a single header file in C) 
    - Use at least C11 (-std=c11)
*/

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>

enum TensorRank {
    RANK_0 = 0,
    RANK_1 = 1,
    RANK_2 = 2,
    RANK_3 = 3,
    RANK_4 = 4
};

typedef enum TensorRank TensorRank;

struct Tensor {

    uint8_t rows;
    uint8_t cols;
    uint8_t depth;
    uint8_t time;

    double* data;

    TensorRank tensor_rank;
};

typedef struct Tensor Tensor;
typedef struct Tensor Matrix;
typedef struct Tensor Vector;

/*----------------------------------*/
/*      Function Declarations       */
/*----------------------------------*/

/**
 * @brief Creates a matrix of m rows and n columns
 * 
 * @param cols n cols
 * @param rows m rows
 * 
 * @return the matrix
*/
Matrix* create_matrix(uint8_t cols, uint8_t rows);

/**
 * @brief Creates a vector of dim components
 * 
 * @param dim dim of the vector
 * 
 * @return the vector
*/
Vector* create_vector(uint8_t dim);

/**
 * @brief Creates a rank 3 tensor
 * 
 * @param cols n cols
 * @param rows m rows
 * @param depth tensor depth
 * 
 * @return the tensor
*/
Tensor* create_tensor_rank3(uint8_t cols, uint8_t rows, uint8_t depth);

/**
 * @brief Creates a rank 4 tensor
 * 
 * @param cols n cols
 * @param rows m rows
 * @param depth tensor depth
 * @param time tensor time dimension
 * 
 * @return the tensor
*/
Tensor* create_tensor_rank4(uint8_t cols, uint8_t rows, uint8_t depth, uint8_t time);

/**
 * @brief Creates a identity matrix only
 * 
 * @param cols m and n cols
*/
Matrix* create_identity(uint8_t cols);

/**
 * @brief Creates a copy of a tensor
 * 
 * @param matrix Tensor pointer
 * 
 * @return the copy
*/
Tensor* create_copy(Tensor* tensor);

/**
 * @brief Sets a value in a position i of a vector
 * 
 * @param vector Vector pointer
 * @param value the value
 * @param i position i
*/
void set_vector(Vector* vector, double value, uint8_t i);

/**
 * @brief Sets a value in a position (i,j) of a matrix
 * 
 * @param matrix Matrix pointer
 * @param value the value
 * @param i position i in (i,j)
 * @param j position j in (i,j)
*/
void set_matrix(Matrix* matrix, double value, uint8_t i, uint8_t j);

/**
 * @brief Sets a value in a position (i,j,k) of a tensor
 * 
 * @param tensor tensor pointer
 * @param value the value
 * @param i position i in (i,j,k)
 * @param j position j in (i,j,k)
 * @param k position k in (i,j,k)
*/
void set_tensor_rank3(Tensor* tensor, double value, uint8_t i, uint8_t j, uint8_t k);

/**
 * @brief Sets a value in a position (i,j,k,t) of a tensor
 * 
 * @param tensor tensor pointer
 * @param value the value
 * @param i position i in (i,j,k,t)
 * @param j position j in (i,j,k,t)
 * @param k position k in (i,j,k,t)
 * @param t position t in (i,j,k,t)
*/
void set_tensor_rank4(Tensor* tensor, double value, uint8_t i, uint8_t j, uint8_t k, uint8_t t);

/**
 * @brief Sets a value in a position of a tensor
 * 
 * @param tensor Tensor pointer
 * @param value the value
 * @param ... position indices
*/
void set(Tensor* tensor, double value, ...);

/**
 * @brief Gets a value from a position i of a vector
 * 
 * @param vector Vector pointer
 * @param i position i
 * 
 * @return the value
*/
double get_vector(Vector* vector, uint8_t i);

/**
 * @brief Gets a value from a position (i,j) of a matrix
 * 
 * @param matrix Matrix pointer
 * @param i position i in (i,j)
 * @param j poistion j in (i,j)
 * 
 * @return the value
*/
double get_matrix(Matrix* matrix, uint8_t i, uint8_t j);

/**
 * @brief Gets a value from a position (i,j,k) of a tensor
 * 
 * @param tensor Tensor pointer
 * @param i position i in (i,j,k)
 * @param j position j in (i,j,k)
 * @param k position k in (i,j,k)
 * 
 * @return the value
*/
double get_tensor_rank3(Tensor* tensor, uint8_t i, uint8_t j, uint8_t k);

/**
 * @brief Gets a value from a position (i,j,k,t) of a tensor
 * 
 * @param tensor tensor pointer
 * @param i position i in (i,j,k,t)
 * @param j position j in (i,j,k,t)
 * @param k position k in (i,j,k,t)
 * @param t position t in (i,j,k,t)
 * 
 * @return the value
*/
double get_tensor_rank4(Tensor* tensor, uint8_t i, uint8_t j, uint8_t k, uint8_t t);

/**
 * @brief Returns a value from a position of a tensor
 * 
 * @param tensor Tensor pointer
 * @param ... position indices
 * 
 * @return the value
*/
double get(Tensor* tensor, ...);

/**
 * @brief Computes the transpose matrix of a matrix
 * 
 * @param matrix Matrix pointer
 */
void transpose(Matrix* matrix);

/**
 * @brief lhs + sign * rhs
 * 
 * @param lhs left tensor
 * @param rhs right tensor
 * 
 * @pre lhs and rhs must be the same rank
*/
void sum_sign(Tensor* lhs, Tensor* rhs, int sign);

/**
 * @brief Sums the rhs tensor to the lhs tensor
 * 
 * @param lhs left tensor
 * @param rhs right tensor
 * 
 * @pre lhs and rhs must be the same rank
*/
void sum(Tensor* lhs, Tensor* rhs);

/**
 * @brief Subtract the rhs tensor to the lhs tensor
 * 
 * @param lhs left tensor
 * @param rhs right tensor
 * 
 * @pre lhs and rhs must be the same rank
*/
void subtract(Tensor* lhs, Tensor* rhs);

/**
 * @brief Sums the rhs scalar to the lhs tensor
 * 
 * @param lhs left tensor
 * @param rhs right scalar
*/
void sum_scalar(Tensor* lhs, double rhs);

/**
 * @brief Multiplies the rhs matrix to the lhs matrix
 * 
 * @param lhs left matrix
 * @param rhs right matrix
*/
void product(Matrix* lhs, Matrix* rhs);

/**
 * @brief Multiplies the rhs value to the lhs tensor
 * 
 * @param lhs left tensor
 * @param rhs right scalar
*/
void product_scalar(Tensor* lhs, double rhs);

/**
 * @brief Multiplies (Hadamard) the rhs tensor to the lhs tensor
 * 
 * @param lhs left tensor
 * @param rhs right tensor
 * 
 * @pre lhs and rhs must be the same rank
*/
void hadamard_product(Tensor* lhs, Tensor* rhs);

/**
 * @brief Computes the tensor product between lhs tensor and rhs tensor
 * 
 * @param lhs left tensor
 * @param rhs right tensor
 * 
 * @pre lhs and rhs must be the same rank
 * @pre only rank 1 and 2 tensors allowed
 * 
 * @return the tensor product
*/
Tensor* tensor_product(Tensor* lhs, Tensor* rhs);

/**
 * @brief Calculates the dot product between lhs and rhs tensors
 * 
 * @param lhs left tensor
 * @param rhs right tensor
 * 
 * @return the dot product
*/
double dot_product(Tensor* lhs, Tensor* rhs);

/**
 * @brief Computes the cross product between lhs and rhs vectors
 * 
 * @param lhs left vector
 * @param rhs right vector
 * 
 * @pre lhs and rhs must be R^3 vectors
 * 
 * @return the cross product
*/
Vector* cross_product(Vector* lhs, Vector* rhs);

/**
 * @brief Transforms a vector by a transformation matrix
 * 
 * @param vector vector
 * @param matrix transformation matrix
*/
void transform(Vector* vector, Matrix* matrix);

/**
 * @brief Calculates the minor of a matrix in a position (i,j) of the matrix
 * 
 * @param matrix Matrix pointer
 * @param i position i in (i,j)
 * @param j position j in (i,j)
 * 
 * @return the minor
 */
double minor(Matrix* matrix, uint8_t i, uint8_t j);

/**
 * @brief Calculates the cofactor of a matrix in a position (i,j) of the matrix
 * 
 * @param matrix Matrix pointer
 * @param i position i in (i,j)
 * @param j position j in (i,j)
 * 
 * @return the cofactor
 */
double cofactor(Matrix* matrix, uint8_t i, uint8_t j);

/**
 * @brief Computes the cofactor matrix of a matrix
 * 
 * @param matrix Matrix pointer
 * 
 * @return the cofactor matrix
*/
Matrix* cofactor_matrix(Matrix* matrix);

/**
 * @brief Computes the adjugate matrix of a matrix
 * 
 * @param matrix Matrix pointer
 * 
 * @return the adjugate matrix
*/
Matrix* adjugate_matrix(Matrix* matrix);

/**
 * @brief Calculates the determinant of a matrix
 * 
 * @param matrix Matrix pointer
 * 
 * @return the determinant
*/
double determinant(Matrix* matrix);

/**
 * @brief Computes the inverse of a matrix
 * 
 * @param matrix Matrix pointer
 * 
 * @return the inverse
*/
Matrix* inverse(Matrix* matrix);

/**
 * @brief Prints a matrix
 * 
 * @param matrix Matrix pointer
*/
void print_matrix(Matrix* matrix);

/**
 * @brief Destroys a tensor
 * 
 * @param tensor Tensor pointer
*/
void destroy_tensor(Tensor* tensor);

/*----------------------------------*/
/*       Function Definitions       */
/*----------------------------------*/

/**
 * @brief Creates a matrix of m rows and n columns
 * 
 * @param cols n cols
 * @param rows m rows
 * 
 * @return the matrix
*/
Matrix* create_matrix(uint8_t cols, uint8_t rows) {

    Matrix* matrix = (Matrix*)malloc(sizeof(Matrix));

    matrix->rows = rows;
    matrix->cols = cols;
    matrix->depth = 1;
    matrix->time = 1;

    matrix->tensor_rank = RANK_2;

    size_t length = rows * cols;
    matrix->data = (double*)malloc(sizeof(double) * length);
    
    for(size_t i = 0; i < length; i ++)
        matrix->data[i] = 0.0;

    return matrix;
}

/**
 * @brief Creates a vector of dim components
 * 
 * @param dim dim of the vector
 * 
 * @return the vector
*/
Vector* create_vector(uint8_t dim) {

    Vector* vector = (Vector*)malloc(sizeof(Vector));

    vector->cols = 1;
    vector->rows = dim;
    vector->depth = 1;
    vector->time = 1;

    vector->tensor_rank = RANK_1;

    vector->data = (double*)malloc(sizeof(double) * dim);

    for(uint8_t j = 0; j < dim; j ++)
        vector->data[j] = 0.0;

    return vector;
}

/**
 * @brief Creates a rank 3 tensor
 * 
 * @param cols n cols
 * @param rows m rows
 * @param depth tensor depth
 * 
 * @return the tensor
*/
Tensor* create_tensor_rank3(uint8_t cols, uint8_t rows, uint8_t depth) {

    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));

    tensor->rows = rows;
    tensor->cols = cols;
    tensor->depth = depth;
    tensor->time = 1;

    tensor->tensor_rank = RANK_3;

    size_t length = rows * cols * depth;
    tensor->data = (double*)malloc(sizeof(double) * length);

    for(size_t i = 0; i < rows * cols * depth; i ++)
        tensor->data[i] = 0.0;

    return tensor;
}

/**
 * @brief Creates a rank 4 tensor
 * 
 * @param cols n cols
 * @param rows m rows
 * @param depth tensor depth
 * @param time tensor time dimension
 * 
 * @return the tensor
*/
Tensor* create_tensor_rank4(uint8_t cols, uint8_t rows, uint8_t depth, uint8_t time) {

    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));

    tensor->rows = rows;
    tensor->cols = cols;
    tensor->depth = depth;
    tensor->time = time;

    tensor->tensor_rank = RANK_4;

    size_t length = rows * cols * depth * time;
    tensor->data = (double*)malloc(sizeof(double) * length);

    for(size_t i = 0; i < length; i ++)
        tensor->data[i] = 0.0;

    return tensor;
}

/**
 * @brief Creates a identity matrix only
 * 
 * @param cols m and n cols
*/
Matrix* create_identity(uint8_t cols) {

    Matrix* matrix = create_matrix(cols, cols);

    for(uint8_t i = 0; i < cols; i ++)
        set(matrix, 1.0, i, i);

    return matrix;
}

/**
 * @brief Creates a copy of a tensor
 * 
 * @param tensor Tensor pointer
 * 
 * @return the copy
*/
Tensor* create_copy(Tensor* tensor) {

    Tensor* copy;

    switch (tensor->tensor_rank) {
    case RANK_1:
        copy = create_vector(tensor->rows);
    break;
    case RANK_2:
        copy = create_matrix(tensor->cols, tensor->rows);
    break;
    case RANK_3:
        copy = create_tensor_rank3(tensor->cols, tensor->rows, tensor->depth);
    break;
    case RANK_4:
        copy = create_tensor_rank4(tensor->cols, tensor->rows, tensor->depth, tensor->time);
    break;
    }

    size_t length = tensor->rows * tensor->cols * tensor->depth * tensor->time;
    for(size_t i = 0; i < length; i ++)
        copy->data[i] = tensor->data[i];

    return copy;
}

/**
 * @brief Sets a value in a position i of a vector
 * 
 * @param vector Vector pointer
 * @param value the value
 * @param i position i
*/
void set_vector(Vector* vector, double value, uint8_t i) {
    vector->data[i] = value;
}

/**
 * @brief Sets a value in a position (i,j) of a matrix
 * 
 * @param matrix Matrix pointer
 * @param value the value
 * @param i position i in (i,j)
 * @param j position j in (i,j)
*/
void set_matrix(Matrix* matrix, double value, uint8_t i, uint8_t j) {
    matrix->data[i + j * matrix->cols] = value;
}

/**
 * @brief Sets a value in a position (i,j,k) of a tensor
 * 
 * @param tensor tensor pointer
 * @param value the value
 * @param i position i in (i,j,k)
 * @param j position j in (i,j,k)
 * @param k position k in (i,j,k)
*/
void set_tensor_rank3(Tensor* tensor, double value, uint8_t i, uint8_t j, uint8_t k) {
    tensor->data[i + j * tensor->cols + k * tensor->rows * tensor->cols] = value;
}

/**
 * @brief Sets a value in a position (i,j,k,t) of a tensor
 * 
 * @param tensor tensor pointer
 * @param value the value
 * @param i position i in (i,j,k,t)
 * @param j position j in (i,j,k,t)
 * @param k position k in (i,j,k,t)
 * @param t position t in (i,j,k,t)
*/
void set_tensor_rank4(Tensor* tensor, double value, uint8_t i, uint8_t j, uint8_t k, uint8_t t) {
    tensor->data[i + j * tensor->cols + k * tensor->rows * tensor->cols + 
        t * tensor->rows * tensor->cols * tensor->depth] = value;
}

/**
 * @brief Sets a value in a position of a tensor
 * 
 * @param tensor Tensor pointer
 * @param value the value
 * @param ... position indices
*/
void set(Tensor* tensor, double value, ...) {

    va_list args;
    va_start(args, value);
    
    switch(tensor->tensor_rank) {
        case RANK_1:
        {
            uint8_t i = va_arg(args, int);
            set_vector(tensor, value, i);
        }
        break;
        case RANK_2:
        {
            uint8_t i = va_arg(args, int);
            uint8_t j = va_arg(args, int);
            set_matrix(tensor, value, i, j);
        }
        break;
        case RANK_3:
        {
            uint8_t i = va_arg(args, int);
            uint8_t j = va_arg(args, int);
            uint8_t k = va_arg(args, int);
            set_tensor_rank3(tensor, value, i, j, k);
        }
        break;
        case RANK_4:
        {
            uint8_t i = va_arg(args, int);
            uint8_t j = va_arg(args, int);
            uint8_t k = va_arg(args, int);
            uint8_t t = va_arg(args, int);
            set_tensor_rank4(tensor, value, i, j, k, t);
        }
        break;
    }

    va_end(args);
}

/**
 * @brief Gets a value from a position i of a vector
 * 
 * @param vector Vector pointer
 * @param i position i
 * 
 * @return the value
*/
double get_vector(Vector* vector, uint8_t i) {
    return vector->data[i];
}

/**
 * @brief Gets a value from a position (i,j) of a matrix
 * 
 * @param matrix Matrix pointer
 * @param i position i in (i,j)
 * @param j poistion j in (i,j)
 * 
 * @return the value
*/
double get_matrix(Matrix* matrix, uint8_t i, uint8_t j) {
    return matrix->data[i + j * matrix->cols];
}

/**
 * @brief Gets a value from a position (i,j,k) of a tensor
 * 
 * @param tensor Tensor pointer
 * @param i position i in (i,j,k)
 * @param j position j in (i,j,k)
 * @param k position k in (i,j,k)
 * 
 * @return the value
*/
double get_tensor_rank3(Tensor* tensor, uint8_t i, uint8_t j, uint8_t k) {
    return tensor->data[i + j * tensor->cols + k * tensor->rows * tensor->cols];
}

/**
 * @brief Gets a value from a position (i,j,k,t) of a tensor
 * 
 * @param tensor tensor pointer
 * @param i position i in (i,j,k,t)
 * @param j position j in (i,j,k,t)
 * @param k position k in (i,j,k,t)
 * @param t position t in (i,j,k,t)
 * 
 * @return the value
*/
double get_tensor_rank4(Tensor* tensor, uint8_t i, uint8_t j, uint8_t k, uint8_t t) {
    return tensor->data[i + j * tensor->cols + k * tensor->rows * tensor->cols + 
        t * tensor->rows * tensor->cols * tensor->depth];
}

/**
 * @brief Returns a value from a position of a tensor
 * 
 * @param tensor Tensor pointer
 * @param ... position indices
 * 
 * @return the value
*/
double get(Tensor* tensor, ...) {

    va_list args;
    va_start(args, tensor);
    
    switch(tensor->tensor_rank) {
        case RANK_1:
        {
            uint8_t i = va_arg(args, int);
            va_end(args);
            return get_vector(tensor, i);
        }
        break;
        case RANK_2:
        {
            uint8_t i = va_arg(args, int);
            uint8_t j = va_arg(args, int);
            va_end(args);
            return get_matrix(tensor, i, j);
        }
        break;
        case RANK_3:
        {
            uint8_t i = va_arg(args, int);
            uint8_t j = va_arg(args, int);
            uint8_t k = va_arg(args, int);
            va_end(args);
            return get_tensor_rank3(tensor, i, j, k);
        }
        break;
        case RANK_4:
        {
            uint8_t i = va_arg(args, int);
            uint8_t j = va_arg(args, int);
            uint8_t k = va_arg(args, int);
            uint8_t t = va_arg(args, int);
            va_end(args);
            return get_tensor_rank4(tensor, i, j, k, t);
        }
        break;
    }

    va_end(args);
    return 0;
}

/**
 * @brief Computes the transpose matrix of a matrix
 * 
 * @param matrix Matrix pointer
 */
void transpose(Matrix* matrix) {

    Matrix* transposeMatrix = create_matrix(matrix->rows, matrix->cols);

    for(uint8_t i = 0; i < transposeMatrix->cols; i ++) {
        for(uint8_t j = 0; j < transposeMatrix->rows; j ++)
            set(transposeMatrix, get(matrix, j, i), i, j);
    }

    matrix->rows = transposeMatrix->rows;
    matrix->cols = transposeMatrix->cols;

    for(uint8_t i = 0; i < matrix->rows * matrix->cols; i ++)
        matrix->data[i] = transposeMatrix->data[i];

    destroy_tensor(transposeMatrix);
}

/**
 * @brief lhs + sign * rhs
 * 
 * @param lhs left tensor
 * @param rhs right tensor
 * 
 * @pre lhs and rhs must be the same rank
*/
void sum_sign(Tensor* lhs, Tensor* rhs, int sign) {

    if(lhs->tensor_rank != rhs->tensor_rank || lhs->cols != rhs->cols || lhs->rows != rhs->rows 
        || lhs->depth != rhs->depth || lhs->time != rhs->time)
        return;

    size_t length = lhs->rows * lhs->cols * lhs->depth * lhs->time;

    for(size_t i = 0; i < length; i ++)
        lhs->data[i] += sign * rhs->data[i];
}

/**
 * @brief Sums the rhs tensor to the lhs tensor
 * 
 * @param lhs left tensor
 * @param rhs right tensor
 * 
 * @pre lhs and rhs must be the same rank
*/
void sum(Tensor* lhs, Tensor* rhs) {
    sum_sign(lhs, rhs, 1);
}

/**
 * @brief Subtract the rhs tensor to the lhs tensor
 * 
 * @param lhs left tensor
 * @param rhs right tensor
 * 
 * @pre lhs and rhs must be the same rank
*/
void subtract(Tensor* lhs, Tensor* rhs) {
    sum_sign(lhs, rhs, -1);
}

/**
 * @brief Sums the rhs scalar to the lhs tensor
 * 
 * @param lhs left tensor
 * @param rhs right scalar
*/
void sum_scalar(Tensor* lhs, double rhs) {

    for(uint8_t i = 0; i < lhs->rows * lhs->cols; i ++)
        lhs->data[i] += rhs;
}

/**
 * @brief Multiplies the rhs matrix to the lhs matrix
 * 
 * @param lhs left matrix
 * @param rhs right matrix
*/
void product(Matrix* lhs, Matrix* rhs) {

    if(lhs->cols != rhs->rows)
        return;

    Matrix* result = create_matrix(rhs->cols, lhs->rows);

    for(uint8_t i = 0; i < rhs->cols; i ++) {
        for(uint8_t j = 0; j < lhs->rows; j ++) {

            double mij = 0.0;
            for(int k = 0; k < lhs->cols; k ++)
                mij += get(rhs, i, k) * get(lhs, k, j);

            set(result, mij, i, j);
        }
    }

    free(lhs->data);
    lhs->cols = rhs->cols;
    lhs->data = (double*)malloc(sizeof(double) * lhs->rows * lhs->cols);

    for(uint8_t i = 0; i < lhs->rows * lhs->cols; i ++)
        lhs->data[i] = result->data[i];

    destroy_tensor(result);
}

/**
 * @brief Multiplies the rhs value to the lhs tensor
 * 
 * @param lhs left tensor
 * @param rhs right scalar
*/
void product_scalar(Tensor* lhs, double rhs) {

    for(uint8_t i = 0; i < lhs->rows * lhs->cols; i ++)
        lhs->data[i] *= rhs;
}

/**
 * @brief Multiplies (Hadamard) the rhs tensor to the lhs tensor
 * 
 * @param lhs left tensor
 * @param rhs right tensor
 * 
 * @pre lhs and rhs must be the same rank
*/
void hadamard_product(Tensor* lhs, Tensor* rhs) {

    if(lhs->tensor_rank != rhs->tensor_rank || lhs->cols != rhs->cols || lhs->rows != rhs->rows 
        || lhs->depth != rhs->depth || lhs->time != rhs->time)
        return;

    size_t length = lhs->rows * lhs->cols * lhs->depth * lhs->time;

    for(size_t i = 0; i < length; i ++)
        lhs->data[i] *= rhs->data[i];
}

/**
 * @brief Computes the tensor product between lhs tensor and rhs tensor
 * 
 * @param lhs left tensor
 * @param rhs right tensor
 * 
 * @pre lhs and rhs must be the same rank
 * @pre only rank 1 and 2 tensors allowed
 * 
 * @return the tensor product
*/
Tensor* tensor_product(Tensor* lhs, Tensor* rhs) {

    if(lhs->tensor_rank == rhs->tensor_rank) {

        switch (lhs->tensor_rank) {
        case RANK_1 :
        {
            Matrix* matrix = create_matrix(rhs->rows, lhs->rows);

            for(uint8_t j = 0; j < matrix->rows; j ++) {
                for(uint8_t i = 0; i < matrix->cols; i ++) {
                
                    double value = get(lhs, j) * get(rhs, i);
                    set(matrix, value, i, j);
                }
            }

            return matrix;
        }    
        break;
        case RANK_2:
        {
            Tensor* tensor = create_tensor_rank4(rhs->cols, rhs->rows, lhs->cols, lhs->rows);

            for(uint8_t t = 0; t < tensor->time; t ++) {
                for(uint8_t k = 0; k < tensor->depth; k ++) {

                    for(uint8_t j = 0; j < tensor->rows; j ++) {
                        for(uint8_t i = 0; i < tensor->cols; i ++) {
                        
                            double value = get(lhs, k, t) * get(rhs, i, j);
                            set(tensor, value, i, j, k, t);
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

/**
 * @brief Calculates the dot product between lhs and rhs tensors
 * 
 * @param lhs left tensor
 * @param rhs right tensor
 * 
 * @return the dot product
*/
double dot_product(Tensor* lhs, Tensor* rhs) {

    if(lhs->cols != rhs->cols || lhs->rows != rhs->rows || 
        lhs->depth != rhs->depth || lhs->time != rhs->time)
        return 0.0;

    double sum = 0.0;
    size_t length = lhs->rows * lhs->cols * lhs->depth * lhs->time;

    for(size_t i = 0; i < length; i ++)
        sum += lhs->data[i] * rhs->data[i];

    return sum;
}

/**
 * @brief Computes the cross product between lhs and rhs vectors
 * 
 * @param lhs left vector
 * @param rhs right vector
 * 
 * @pre lhs and rhs must be R^3 vectors
 * 
 * @return the cross product
*/
Vector* cross_product(Vector* lhs, Vector* rhs) {

    Vector* cross = create_vector(3);

    if(lhs->rows != 3 && rhs->rows != 3)
        return cross;

    cross->data[0] = lhs->data[1] * rhs->data[2] - lhs->data[2] * rhs->data[1];
    cross->data[1] = -lhs->data[0] * rhs->data[2] + lhs->data[2] * rhs->data[0];
    cross->data[2] = lhs->data[0] * rhs->data[1] - lhs->data[1] * rhs->data[0];

    return cross;
}

/**
 * @brief Transforms a vector by a transformation matrix
 * 
 * @param vector vector
 * @param matrix transformation matrix
*/
void transform(Vector* vector, Matrix* matrix) {

    if(matrix->cols != vector->rows)
        return;

    Vector* vec_transformed = create_vector(vector->rows);

    for(uint8_t j = 0; j < vector->rows; j ++) {

        Vector* row_vector = create_vector(vector->rows);

        for(uint8_t i = 0; i < matrix->cols; i ++) {
            
            double value = get(matrix, i, j);
            set(row_vector, value, i);
        }

        vec_transformed->data[j] = dot_product(vector, row_vector);

        destroy_tensor(row_vector);
    }

    for(uint8_t j = 0; j < vector->rows; j ++)
        vector->data[j] = vec_transformed->data[j];

    destroy_tensor(vec_transformed);
}

/**
 * @brief Calculates the minor of a matrix in a position (i,j) of the matrix
 * 
 * @param matrix Matrix pointer
 * @param i position i in (i,j)
 * @param j position j in (i,j)
 * 
 * @return the minor
 */
double minor(Matrix* matrix, uint8_t i, uint8_t j) {

    Matrix* sub_matrix = create_matrix(matrix->cols - 1, matrix->cols - 1);

    uint8_t index = 0;
    for(uint8_t row = 0; row < matrix->rows; row ++) {
        for(uint8_t col = 0; col < matrix->cols; col ++) {

            if(i != row && j != col) {
                double value = get(matrix, row, col);
                sub_matrix->data[index] = value;
                index ++;
            }
        }
    }

    double det = determinant(sub_matrix);
    destroy_tensor(sub_matrix);

    return det;
}

/**
 * @brief Calculates the cofactor of a matrix in a position (i,j) of the matrix
 * 
 * @param matrix Matrix pointer
 * @param i position i in (i,j)
 * @param j position j in (i,j)
 * 
 * @return the cofactor
 */
double cofactor(Matrix* matrix, uint8_t i, uint8_t j) {

    int sign = pow(-1.0, i + j);
    double result =  sign * minor(matrix, i, j);
    return result;
}

/**
 * @brief Computes the cofactor matrix of a matrix
 * 
 * @param matrix Matrix pointer
 * 
 * @return the cofactor matrix
*/
Matrix* cofactor_matrix(Matrix* matrix) {

    Matrix* cof_matrix = create_matrix(matrix->cols, matrix->rows);

    for(uint8_t i = 0; i < matrix->cols; i ++) {
        for(uint8_t j = 0; j < matrix->rows; j ++)
            set(cof_matrix, cofactor(matrix, i, j), i, j);
    }

    return cof_matrix;
}

/**
 * @brief Computes the adjugate matrix of a matrix
 * 
 * @param matrix Matrix pointer
 * 
 * @return the adjugate matrix
*/
Matrix* adjugate_matrix(Matrix* matrix) {

    Matrix* cof_matrix = cofactor_matrix(matrix);
    transpose(cof_matrix);

    return cof_matrix;
}

/**
 * @brief Calculates the determinant of a matrix
 * 
 * @param matrix Matrix pointer
 * 
 * @return the determinant
*/
double determinant(Matrix* matrix) {

    if(matrix->rows != matrix->cols)
        return 0.0;

    if(matrix->cols == 2)
        return get(matrix, 0, 0) * get(matrix, 1, 1) - get(matrix, 0, 1) * get(matrix, 1, 0);

    double result = 0.0;

    for(uint8_t j = 0; j < matrix->cols; j ++) {
        double cof = cofactor(matrix, 0, j);
        result += get(matrix, 0, j) * cof; 
    }

    return result;
}

/**
 * @brief Computes the inverse of a matrix
 * 
 * @param matrix Matrix pointer
 * 
 * @return the inverse
*/
Matrix* inverse(Matrix* matrix) {

    double det = determinant(matrix);
    Matrix* inv = adjugate_matrix(matrix);
    double inv_determinant = 1 / det;
    
    product_scalar(inv, inv_determinant);

    return inv;
}

/**
 * @brief Prints a matrix
 * 
 * @param matrix Matrix pointer
*/
void print_matrix(Matrix* matrix) {

    for(uint8_t j = 0; j < matrix->rows; j ++) {
        for(uint8_t i = 0; i < matrix->cols; i ++) {
            printf("%f ", get(matrix, i, j));
        }
        printf("\n");
    }
}

/**
 * @brief Destroys a tensor
 * 
 * @param tensor Tensor pointer
*/
void destroy_tensor(Tensor* tensor) {

    free(tensor->data);
    free(tensor);
}