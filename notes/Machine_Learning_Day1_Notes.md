# Machine Learning Notes -- Day 1

## NumPy Foundations and Introduction to Linear Regression

**Topics Covered** - Setting up Jupyter Notebook in VS Code - Why NumPy
is used in Machine Learning - Creating NumPy arrays - Array properties
(`shape`, `size`, `dtype`, `ndim`) - Array creation methods - Scalars,
Vectors, and Matrices - Features (`X`) and Labels (`Y`) - Linear
Regression intuition - Normal Equation - Matrix inverse vs pseudoinverse
(`pinv`) - Understanding the example code

------------------------------------------------------------------------

# 1. Jupyter Notebook Workflow

A Jupyter Notebook is divided into **cells**.

-   **Code cells** execute Python.
-   **Markdown cells** contain explanations, formulas, and notes.

Recommended notebook structure:

1.  Title
2.  Aim
3.  Theory
4.  Code
5.  Output
6.  Observation
7.  Conclusion

------------------------------------------------------------------------

# 2. Why NumPy?

Python lists are flexible but relatively slow for numerical computation.

NumPy provides:

-   Fast numerical computation
-   Efficient memory usage
-   Matrix operations
-   Statistical functions
-   Linear algebra

Import:

``` python
import numpy as np
```

------------------------------------------------------------------------

# 3. Creating Arrays

## Using `np.array()`

``` python
arr = np.array([10,20,30,40,50])
```

## Using `np.zeros()`

``` python
np.zeros(5)
np.zeros(5, dtype=int)
```

Creates arrays filled with zeros.

## Using `np.ones()`

``` python
np.ones(5)
```

Creates arrays filled with ones.

## Using `np.full()`

``` python
np.full(5,100)
```

Creates arrays filled with the same value.

## Using `np.empty()`

``` python
np.empty(5)
```

Allocates memory without initializing values.

## Using `np.arange()`

``` python
np.arange(0,10,2)
```

Generates values using **start, stop, step**.

## Using `np.linspace()`

``` python
np.linspace(0,100,11)
```

Generates a specified number of evenly spaced values.

## Random Arrays

``` python
np.random.rand(5)
np.random.randint(1,101,10)
```

------------------------------------------------------------------------

# 4. Array Properties

``` python
arr.shape
arr.size
arr.ndim
arr.dtype
```

Meaning:

-   **shape** → dimensions (rows, columns)
-   **size** → total elements
-   **ndim** → number of dimensions
-   **dtype** → data type

------------------------------------------------------------------------

# 5. Scalars, Vectors, and Matrices

## Scalar

A single value.

Example:

``` python
x = 5
```

## Vector

Represents one observation with multiple features.

``` python
student = np.array([8.2,5,2])
```

## Matrix

Represents multiple observations.

``` python
students = np.array([
    [8.2,5,2],
    [7.5,3,1],
    [9.1,8,4]
])
```

Rows represent **samples**.

Columns represent **features**.

------------------------------------------------------------------------

# 6. Features and Labels

Example dataset:

  CGPA   Projects   Placement
  ------ ---------- -----------
  8.2    5          1
  7.4    2          0
  9.0    7          1

Split:

``` python
X = data[:, :2]
Y = data[:, 2]
```

`X` contains input features.

`Y` contains the target (label).

------------------------------------------------------------------------

# 7. Linear Regression

Goal:

Learn a relationship between inputs and outputs.

Equation (single feature):

\[ y = mx + c \]

Multiple Linear Regression:

\[ y = A_0 + A_1x_1 + A_2x_2 \]

Where:

-   A0 = intercept
-   A1, A2 = feature weights

------------------------------------------------------------------------

# 8. Matrix Form

Input matrix:

\[ X \]

Target vector:

\[ Y \]

Parameter vector:

\[ `\beta`{=tex} \]

Prediction:

\[ Y = X`\beta`{=tex} \]

------------------------------------------------------------------------

# 9. Cost Function

Error:

    Actual − Predicted

Squared Error:

    (Actual − Predicted)^2

Cost:

\[ J = `\sum `{=tex}(y-`\hat `{=tex}y)\^2 \]

Objective:

Minimize the total squared error.

------------------------------------------------------------------------

# 10. Normal Equation

The optimal parameters are

\[ `\beta`{=tex}=(X^TX)^{-1}X\^TY \]

This directly computes the coefficients without iterative optimization.

------------------------------------------------------------------------

# 11. Why `pinv()`?

Your dataset contained dependent features:

    Age = 5 × Experience + 50

Therefore:

-   `X^TX` becomes singular.
-   `np.linalg.inv()` fails.
-   `np.linalg.pinv()` computes the Moore--Penrose pseudoinverse and
    returns the least-squares solution.

Recommended:

``` python
beta = np.linalg.pinv(X) @ Y
```

------------------------------------------------------------------------

# 12. Understanding the Code

``` python
beta = np.linalg.pinv(X) @ Y
```

-   `pinv(X)` computes the pseudoinverse.
-   `@` performs matrix multiplication.
-   `beta` contains the learned parameters.

Example output:

    A0 = -0.4886
    A1 = 4.9882
    A2 = 0.5109

Prediction:

``` python
prediction = A0 + A1*x1 + A2*x2
```

------------------------------------------------------------------------

# 13. Important Concepts Learned

-   Machine Learning learns from data.
-   NumPy is the numerical foundation of ML.
-   Arrays are the basic data structure.
-   Rows represent samples.
-   Columns represent features.
-   Linear Regression fits the best line.
-   The Normal Equation minimizes squared error.
-   `pinv()` handles singular matrices robustly.

------------------------------------------------------------------------

# Suggested Repository Structure

``` text
machine-learning-lab/
│
├── notebooks/
├── datasets/
├── notes/
├── exercises/
├── projects/
├── README.md
├── requirements.txt
└── .gitignore
```

Use one Git branch per lesson, for example:

-   `lesson-01-numpy`
-   `lesson-02-linear-regression`
-   `lesson-03-matrices`

Commit with meaningful messages after completing each lesson.

------------------------------------------------------------------------

# Next Topics

1.  Matrix multiplication
2.  Transpose
3.  Matrix rank
4.  Determinant
5.  Inverse
6.  Broadcasting
7.  NumPy indexing and slicing
8.  Linear Regression implementation from scratch
9.  Gradient Descent
