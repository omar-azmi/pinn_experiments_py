# PINN experiments in python

Trying out _Physics Informed Neural Networks_ (PINNs) in python.

# 1. Solving a PDE via Classical Neural Networks

## Problem statement

We are given the following partial differential equation and boundary conditions:

$$\begin{align}
\text{PDE:} \quad & \frac{\partial u}{\partial x} = 2 \frac{\partial u}{\partial t} + u \\
\text{Boundary Conditions:} \quad & u_{(x, 0)} = 6 \cdot e^{-3x} \\
& x \in \left[0, 2\right] \\
& t \in \left[0, 1\right]
\end{align}$$

## Exact solution

Lets try to solve the problem analytically to arrive at an exact solution, and perhaps gain some intuition on how our neural network should be designed to solve the PDE effectively.

Given the shape of the PDE, let's assume that $u_{(x, t)}$ carries the following form:

$$
u_{(x, t)} = A \cdot e^{\omega t + \gamma x} \quad , \quad \text{where} \quad A, \omega, \gamma \in \Bbb{R}
$$

Let's compute the partial differentials of $u$ with respect to $x$ and $t$, since they are used in the PDE:

$$
\frac{\partial u}{\partial x} = \gamma u \quad , \quad \frac{\partial u}{\partial t} = \omega u
$$

Substituting this form of $u$ back into the original PDE gives us:

$$\begin{align}
& \gamma u = 2 \omega u + u \\
\rightarrow \quad & \boxed{\gamma = 2 \omega + 1} \\
\therefore \quad & u_{(x, t)} = A \cdot e^{\omega t + (2 \omega + 1) x}
\end{align}$$

Applying the boundary condition now gives us:

$$\begin{align}
& \left. u_{(x, t)} \right|_{t = 0} = 6 \cdot e^{-3x} \\
\rightarrow \quad & A \cdot e^{(2 \omega + 1) x} = 6 \cdot e^{-3x} \\
\therefore \quad & \boxed{A = 6} \\
\text{and} \quad & 2 \omega + 1 = -3 \\
\rightarrow \quad & \boxed{\omega = -2} \\
\end{align}$$

Thus, we now have the exact solution to $u$ :

$$
\boxed{u_{(x, t)} = 6 \cdot e^{-2t - 3x}}
$$

## Breaking down the problem for Classical Neural Networks

### Where do we get our training data from?

We use the boundary condition $u_{(x, 0)}$ to generate any number of data points (for $x \in [0, 2]$ ).
Needless to say, these data points will be somewhat lacking, as they only occupy the span of one line, instead of the whole plane of possible values for $x$ and $t$ .

### What is the loss function here?

Suppose we run the neural network with input points $(x, t) \in \Gamma$ (in other words: $\Gamma \subset [0, 2] \times [0, 1]$ ).
Then, the resulting predicted output values of $u$ for each data point will be $\widetilde{u}_ {(x, y)}$ ,
with the actual "true" (training data) values being $u_{(x, t)}$ .

The loss function $\mathcal{L}_ {u}$ is then defined as the mean-squared-difference between the expectation $\widetilde{u}_ {(x, y)}$ , and the training value $u_ {(x, t)}$ .

$$\begin{gathered}
\mathcal{L}_ {u} = \frac{1}{\\#{\Gamma}} \sum_{\forall (x, t) \in \Gamma}{ \left( u_{(x, t)} - \widetilde{u}_{(x, y)} \right)^{2} } \\
\text{where } \\#{\Gamma} \text{ represents the number of elements in the set } \Gamma \text{ (i.e. cardinality)}
\end{gathered}$$

### What are the components of a Neural Network?

Any neural network with one or more hidden layers can be thought as a universal function approximator (citation needed).
A single layer in a neural network consists of nodes that are independent of each other, but depend on the output values of the previous layer(s)'s nodes (unless they are in the input layer).

There are 3 types of layers:

- `Input Layer`: The nodes of this layer is fed your input data (which would be the values of $x$ and $t$ in our scenario).
- `Hidden Layer`: The nodes of this layer take in the values from the previous layer, apply some _non-linear_ function (usually a linear combination, followed by a non-linear activation function), and then return their output computed values.
- `Output Layer`: The nodes in this layer are the end-points of your neural network.
  Unlike the hidden layer, these nodes typically perform a _linear_ operation on the output of the previous hidden layer.

### Significance of Non-Linearity

#### Assuming Linear Hidden Layers

Why is it essential for the hidden nodes of the neural network to be non-linear?

Well, suppose that we had a neural network with input $\vec{X}$ , outputs $\vec{U}$ , and $N$ _linear_ hidden layers $H_{0}, H_{1}, \dots, H_{N - 1}$ (representable as matrices), in addition to $H_{N}$ for transforming the output of $H_{N - 1}$.
Now, this effectively means that the output value $\vec{U}$ for a certain input $\vec{X}$ can be computed as the following matrix multiplication:

$$
\vec{U} = H_{N} H_{N - 1} \cdots H_{1} H_{0} \vec{X}
$$

But we can reduce (precompute) the matrix multiplications into just a single matrix $G =  H_{N} H_{N - 1} \cdots H_{1} H_{0}$ , simplifying our expression to:

$$
\vec{U} = G \vec{X}
$$

And since, all $H_{i \in \{0, 1, \dots, N\}}$ matrices were independent of the input nodes $\vec{X}$ , we can use $G$ to perform a computation of any $\vec{U}$ via a single matrix multiplication of $\vec{X}$ .

In essence, we have collapsed our $N$ hidden layers into precisely _zero_ hidden layers.
In fact, this is single-matrix-multiplication formulation is precisely what _linear regression_ is.
And we're not here to do linear regression, are we?

> Takeaway: A Neural Network with linear hidden layers is equivalent to zero hidden layers.
> You're just performing linear-regression in a much less efficient way.

#### Assuming Non-Linear Hidden Layers

So what does introducing non-linearity do to our system?

Let's step back and first transform our linear example's hidden layers to non-linear layers.
To do so, we simply pick a non-linear 1D function $s_{(v)}$ of our choice (such as a [_sigmoid_](https://en.wikipedia.org/wiki/Sigmoid_function) or a [_ReLU_](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))), and apply that non-linear function to each hidden layer's node's output value.

To simplify our declaration for matrix-wise operations, we need to introduce the following vector notation:

$$
s(\vec{Y}) \coloneqq \langle s(Y_{0}), s(Y_{1}), \dots, s(Y_{M - 1}) \rangle
$$

So now, in terms of layer-wise matrices, we get the following expression for computing $\vec{U}$ , given some input $\vec{X}$ :

$$
\vec{U} = H_{N} \cdot s(H_{N - 1} \cdots s(H_{1} s(H_{0} \vec{X})))
$$

And it would now be clear to you that this expression is irreducible.
Hence, we cannot pre-compute any part of this expression to make it smaller without having a concrete value of $\vec{X}$ .

The non-linearity portions are what make neural networks fundamentally different from statisctical methods.
Every layer must exhibit some form of non-linearity, otherwise it _will_ be reducible.

### Conventional Neural Networks

Typically, the hidden nodes of a neural network use simple matrix-based linear computation on their inputs, then add a constant bias value, and finally follow it up by applying a non-linear function onto each element of the output vector.

Given a hidden layer $H^{(k)}$ , with dimensionality $Dim(H^{(k)}) = \langle a, b \rangle$ ($b$ number of inputs and $a$ number of outputs (i.e. $a$ is also the number of nodes in that layer)):

- Each node $i \in \{1, \dots, a\}$ is a $(1 \times b)$ matrix, consisting of:
  - The matrix entries $W_{ij}$ (for $j \in \{1, \dots, b\}$ ), known as the `weights` of the $i^{th}{-}\text{node}$.
  - The scalar value $B_{i}$, known as the `bias` of the $i^{th}{-}\text{node}$.
  - A scalar non-linear function $S_{i (v)}$ , known as the `activation function` of the $i^{th}{-}\text{node}$.
- Thus an input $b{-}\text{dimensional}$ vector $\vec{V}$ will be transformed by the $i^{th}{-}\text{node}$ into the scalar value:

$$
S_{i}(B_{i} + W_{i:} \vec{V})
$$

- Therefore, the layer as a whole transforms some $b{-}\text{dimensional}$ input $\vec{V}$ to the $a{-}\text{dimensional}$ vector:

$$
H^{(k)}(\vec{V}) \coloneqq S(B + W_{::} \vec{V})
$$

- To make it clearer that each component of the expression is associated with the $k^{th}$ hidden layer, we use the $(k)$ superscript on each component:

$$
H^{(k)}(\vec{V}) \coloneqq S^{(k)}(B^{(k)} + W^{(k)} \vec{V})
$$

As a whole, our $N$ hidden-layered neural network can now be mathematically defined as:

$$\begin{align}
\vec{U} & = H^{(N)} \circ H^{(N - 1)} \circ \dots \circ H^{(1)} \circ H^{(0)} \vec{X} \\
& = H_{N} \circ S^{(N - 1)}(B^{(N - 1)} + W^{(N - 1)}( \dots S^{(1)}(B^{(1)} + W^{(1)}( S^{(0)}(B^{(0)} + W^{(0)}( \vec{X} )) )) ))
\end{align}$$

But it might be clearer to write the recursion iteratively:

$$\begin{align}
\vec{h}^{(-1)} & \coloneqq \vec{X} \\
\vec{h}^{(k)} & \coloneqq S^{(k)}(B^{(k)} + W^{(k)} \vec{h}^{(k - 1)}) \quad , \quad \text{for } k \in \{0, \dots, N\} \\
S^{(k)} & \hspace{0.5em} \text{is a non-linear function } \Bbb{R} \rightarrow \Bbb{R} \quad , \quad \text{for } k \in \{0, \dots, N - 1\} \\
S^{(N)} & \coloneqq \Bbb{1} \quad \text{(the identity function (linear))} \\
\vec{U}^{(k)} & \coloneqq \vec{h}^{(N)}
\end{align}$$

### What does the Neural Network look like?

If we were using a $2 \times 3$ neural network (2 hidden layers, with 3 nodes each) to solve our 2-input and 1-output problem, then, it can be illustrated on a graph as the following:

```mermaid
---
config:
  layout: elk
---
flowchart LR

subgraph inputs["Input Layer X"]
direction TB
	input0["x"]
	input1["t"]
	input0@{ shape: dbl-circ }
	input1@{ shape: dbl-circ }
end

subgraph net["Neural Network"]
direction LR

	subgraph layer0["Hidden Layer 0"]
	direction TB
		n00(("H[0,0]"))
		n01(("H[0,1]"))
		n02(("H[0,2]"))

		input0 & input1 --> n00 & n01 & n02
	end

	subgraph layer1["Hidden Layer 1"]
	direction TB
		n10(("H[1,0]"))
		n11(("H[1,1]"))
		n12(("H[1,2]"))

		n00 & n01 & n02 --> n10 & n11 & n12
	end

end

subgraph outputs["Output Layer U"]
direction TB
	output0["u"]
	output0@{ shape: dbl-circ }

	n10 & n11 & n12 --> output0
end

inputs ~~~ layer0
layer0 ~~~ layer1
layer1 ~~~ outputs
```

### Gradients of the Network Weights and Biases

To understand how the network parameters (weights and biases) need to be tweaked to minimize the loss $\mathcal{L}_ {u}$,
we will need to compute the partial derivatives of $\mathcal{L}_ {u}$ with respect to each trainable parameter (denoted as $\theta$ in the literature),
at a given set of training data (inputs and expected truthful outputs).

#### Automatically computing gradients with JAX

Now we talk about the specifics of the [jax](https://github.com/jax-ml/jax) library.
In jax, any simple ndarray compatible mathematical function is differentiable via `jax.grad`.
However, the return type of the derivative function generated by `jax.grad` is **always** a scalar (i.e. `float`).

For instance, lets suppose you have a 1-dimensional ( $\Bbb{R} \rightarrow \Bbb{R}$ ) function `f(x: float) -> float`,
which also supports `x: NDArray[float]` as an input (and will hence return an `NDArray[float]` during that case).
Then, when you differentiate `f` via jax using `df_dx = jax.grad(f)`, the signature of this function will strictly adhere to `df_dx(x: float) -> float`,
and it will absolutely **not** accept `x: NDArray[float]` as an input, and neither will it output an array.

The only way to _vectorize_ the generated scalar derivative function `df_dx` is by applying the `jax.vmap` function onto it, via `vectorized_df_dx = jax.vmap(df_dx)`.
And with this, the vectorized version will now strictly accept ndarrays as inputs and return ndarrays as outputs (i.e. `vectorized_df_dx(x: NDArray[float]) -> NDArray[float]`).
This means that using `vectorized_df_dx` on a scalar value (such as `float`) will throw an error.

Now you may wonder, what will happen if you compute the gradient of a _vectorized_ function.
For instance, what should the function signature of `df2_dx2 = jax.grad(vectorized_df_dx)` be?
The answer is that it won't work at all; `df2_dx2` will neither accept scalar inputs, nor ndarray inputs.
It will throw an error for any type of input that you provide it with.
This is because the function that you provide `jax.grad` **must** always support `float` as the return type (whether strictly or loosely).
And as we know, `vectorized_df_dx` only supports `NDArray[float]` as the return type (hence it is incompatible with `jax.grad` during runtime).

The lesson here is that you need to be very explicit about the type (scalar or ndarray) of your jax-bound functions.
And, you should never compute the gradient of a vector function, because `jax.grad` only works for functions that return a scalar value.

For a demonstration on how to work with jax functions, gradients, and vectorizations, take a look at the example below:

```py
from unittest import TestCase
import jax
import jax.numpy as jnp
t = TestCase()

# `f` is a loosely typed ndarray compatible function
f = jnp.sin

# `f` accepts scalars, arrays, matrices, and higher dimensional arrays, by applying the function element-wise
assert f(1.0).shape == tuple()
assert f(jnp.array([1.0, 1.0])).shape == (2,)
assert f(jnp.array([[1.0, 1.0], [1.0, 1.0]])).shape == (2, 2)

# the gradient of `f` (with respect to `x`) is `df_dx`, and it is a strictly typed `(x: float) -> float` function
df_dx = jax.grad(f)
# this is a vectorized version of `df_dx`, with the strictly typed signature `(x: Array1d[float]) -> Array1d[float]`
vectorized_df_dx = jax.vmap(df_dx)
# this is a matrix-vectorized version of `df_dx`, with the strictly typed signature `(x: Array2d[float]) -> Array2d[float]`
metricized_df_dx = jax.vmap(vectorized_df_dx)

# we can no longer use loosely typed inputs.
# we must use the appropriate element-wise vectorized version of our function based on the type of our input.
assert f(1.0).shape == tuple()
assert vectorized_df_dx(jnp.array([1.0, 1.0])).shape == (2,)
assert metricized_df_dx(jnp.array([[1.0, 1.0], [1.0, 1.0]])).shape == (2, 2)

# not using the correct input type will lead to an error being thrown
t.assertRaises(BaseException, vectorized_df_dx, 1.0)
t.assertRaises(BaseException, metricized_df_dx, 1.0)
t.assertRaises(BaseException, df_dx, jnp.array([1.0, 1.0]))
t.assertRaises(BaseException, metricized_df_dx, jnp.array([1.0, 1.0]))
t.assertRaises(BaseException, df_dx, jnp.array([[1.0, 1.0], [1.0, 1.0]]))
t.assertRaises(BaseException, vectorized_df_dx, jnp.array([[1.0, 1.0], [1.0, 1.0]]))

# computing the gradient of a vectorized function will lead to a runtime error
df2_dx2 = jax.grad(df_dx)                                # gradient of `(x: float) -> float` is permitted
wrongly_vectorized_df2_dx2 = jax.grad(vectorized_df_dx)  # gradient of `(x: NDArray[float]) -> NDArray[float]` is NOT permitted
vectorized_df2_dx2 = jax.vmap(df2_dx2)                   # this is the correct way to get the vectorized version of the gradient
# as it can be seen, `wrongly_vectorized_df2_dx2` neither supports `float`, nor `NDArray` as its input, because it was constructed incorrectly
t.assertRaises(BaseException, wrongly_vectorized_df2_dx2, 1.0)
t.assertRaises(BaseException, wrongly_vectorized_df2_dx2, jnp.array([1.0, 1.0]))
assert df2_dx2(1.0).shape == tuple()
assert vectorized_df2_dx2(jnp.array([1.0, 1.0])).shape == (2,)
```

With jax, you may even compute the partial-derivative of multidimensional functions $\Bbb{R}^{m} \rightarrow \Bbb{R}$ ,
so long as you specify the argument(s) with respect to which you want to compute the partial derivative.
By default, if you don't specify `argnums`, the partial derivative with respect to the first argument will be computed (i.e. it will behave as though `argnums = 0`).
You may even provide a sequence integers to `argnums` to compute the "vector gradient" $\nabla f$ in the form of tuple values.

TODO: investigate how the poorly documented `reduce_axes` kwarg of `jax.grad` works.

Since it is a little hard to grasp, it best to play around and test your gradient functions before using them in your code.
The following snippet might give you some idea as to how multidimensional functions work with `jax.grad`:

```py
from typing import Tuple, Sequence
import jax
import jax.numpy as jnp

def f(x: float, y: float, z: float) -> float:
	return (x ** 2.0 + y ** 2.0 + z ** 2.0)

def g(v: Tuple[float, float, float]) -> float:
	return (v[0] ** 2.0 + v[1] ** 2.0 + v[2] ** 2.0)

def h(v: Sequence[float]) -> float:
	return jnp.sum(jnp.array(v) ** 2)

def k(v: jax.Array) -> float:
	return jnp.sum(v ** 2)

df_dy = jax.grad(f, argnums=1)
df_dx_tuple = jax.grad(f, argnums=[0])
df_dxyz_tuple = jax.grad(f, argnums=[0,1,2])
dg_dxyz_tuple = jax.grad(g)
dh_dv = jax.grad(h)
dk_dv = jax.grad(k)

assert df_dy(1.0, 2.0, 4.0) == 4.0
assert df_dx_tuple(1.0, 2.0, 4.0) == (2.0,)
assert dg_dxyz_tuple((1.0, 2.0, 4.0,)) == (2.0, 4.0, 8.0,)
assert df_dxyz_tuple(1.0, 2.0, 4.0) == (2.0, 4.0, 8.0,)
assert dg_dxyz_tuple((1.0, 2.0, 4.0,)) == df_dxyz_tuple(1.0, 2.0, 4.0)

assert dh_dv(1.0) == 2.0
assert dk_dv(1.0) == 2.0
assert dh_dv((1.0,)) == (2.0,)                      # we cannot do the same with `dk_dv`, since it is not compatible with tuples
assert dh_dv((1.0, 2.0, 4.0,)) == (2.0, 4.0, 8.0,)  # we cannot do the same with `dk_dv`, since it is not compatible with tuples
assert all(dh_dv(jnp.array([1.0, 2.0, 4.0])) == jnp.array([2.0, 4.0, 8.0]))
assert all(dk_dv(jnp.array([1.0, 2.0, 4.0])) == jnp.array([2.0, 4.0, 8.0]))
```

#### Utilizing JAX in our Classical Neural Network

TODO: think of how to apply jax features to compute the gradient of the loss, and then perform gradient descent at a given learning rate.

<!--
$$\begin{align}
\vec{h}^{(-1)} & \coloneqq \vec{X} \\
\vec{h}^{(k)} & \coloneqq S^{(k)}(B^{(k)} + W^{(k)} \vec{h}^{(k - 1)}) \quad , \quad \text{for } k \in \{0, \dots, N\} \\
S^{(k)} & \hspace{0.5em} \text{is a non-linear function } \Bbb{R} \rightarrow \Bbb{R} \quad , \quad \text{for } k \in \{0, \dots, N - 1\} \\
S^{(N)} & \coloneqq \Bbb{1} \quad \text{(the identity function (linear))} \\
\vec{U}^{(k)} & \coloneqq \vec{h}^{(N)}
\end{align}$$
-->

## Further reading

- Explain measuring the goodness of our weights + biases via the total error
- Explain back propagation
- Explain gradient-descent, and how it is achieved via back propagation

For completeness, read the following excellent slides on mathematical introduction to neural networks: [University of Toronto, CSC411, Lecture 10](https://www.cs.toronto.edu/~jlucas/teaching/csc411/lectures/lec10_handout.pdf)

# 2. Solving a PDE via Physics Informed Neural Networks (PINN)

TODO: complete this section after you've established the classical neural network case.
