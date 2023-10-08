# Dev notes

## Math

<details>
<summary>Note: Math in GitHub</summary>

For my own reference, GitHub uses MathJax to render LaTeX math equations in Markdown
files. If you want to modify some of the equations, you can use this playground,
https://www.mathjax.org/#demo. I am using the delimiter syntax variant ([added in May
2023](https://github.blog/changelog/2023-05-08-new-delimiter-syntax-for-inline-mathematical-expressions/))
where you start the expression with <code>$\`</code> and end it with <code>\`$</code> in
order to avoid syntax conflicts with GitHub markdown.

</details>


In order to lay out the math equations we're going to be using, let's use this
ridiculously simple neural network that has just 3 nodes connected by 2 weights.

```mermaid
flowchart LR
    subgraph layer0[Input layer 0]
        subgraph groupLabel0[a<sub>0</sub>]
            l0_0(( ))
        end
    end

    subgraph layer1[Hidden layer 1]
        subgraph groupLabel1[a<sub>1</sub>]
            l1_0(( ))
        end
    end

    subgraph layer2[Output layer 2]
        subgraph groupLabel2[a<sub>2</sub>]
            l2_0(( ))
        end
    end

    l0_0 -- w1<br><br> --> l1_0 -- w2<br><br> --> l2_0


classDef layerGroup stroke:none,fill:none
classDef groupLabel stroke:none,fill:none

class layer0 layerGroup
class layer1 layerGroup
class layer2 layerGroup

class groupLabel0 layerGroup
class groupLabel1 layerGroup
class groupLabel2 layerGroup
```

#### Forward propagation:

This just means that we're going to feed the input through each layer in the network to
the next until we get the output in the final layer.

Some variable explanations:

 - $`a0`$: This is often labeled as $`x`$. Labeling the input as an "activation" is a bit strange since it hasn't been through our activation function but it just makes our notation a bit more consistent.
 - $`w1`$: The weight of the connection

Equations:

 - $`z_1 = a_0*w_1 + b_1`$: The weighted input to the 1st layer
 - $`a_1 = ActivationFunction(z_1)`$: activation 1
 - $`z_2 = a_1*w_2 + b_2`$: The weighted input to the 2nd layer
 - $`a_2 = ActivationFunction(z_2)`$ activation 2
 - $`c = CostFunction(a_2, \mathrm{expected\_output})`$: Cost (also known as loss) ($`\mathrm{expected\_output}`$ is often labeled as $`y`$)


#### Backward propagation:

With backwards propagation, our goal is to minimize the cost function which is achieved
by adjusting the weights and biases. In order to find which direction we should step in
order to adjust the weights and biases, we need to find the slope of the cost function
with respect to the weights and biases. The pure math way to find the slope of a
function is to take the derivative (same concepts that you learned in calculus class).
If we keep taking these steps downhill, we will eventually reach a local minimum of the cost
function (where the slope is 0) which is the goal. This process is called gradient descent.

If we keep these steps proportional to the slope, then when the slope is flattening out
approaching a local minimum, our steps get smaller and smaller which helps us from
overshooting. This is why our learn rate is some small number.

The partial derivative of cost with respect to the weight of the 2nd connection.
$`\begin{aligned}
\frac{\partial c}{\partial w_2} &= \frac{\partial z_2}{\partial w_2} &\times& \frac{\partial a_2}{\partial z_2} &\times& \frac{\partial c}{\partial a_2}
\\&= a_1 &\times& \verb|activation_function.derivative|(z_2) &\times& \verb|cost_function.derivative|(a_2)
\end{aligned}`$

The partial derivative of cost with respect to the weight of the 1st connection.
$`\begin{aligned}
\frac{\partial c}{\partial w_1} &= \frac{\partial z_1}{\partial w_1} &\times& \frac{\partial a_1}{\partial z_1} &\times& \frac{\partial z_2}{\partial a_1} &\times& \frac{\partial a_2}{\partial z_2} &\times& \frac{\partial c}{\partial a_2}
\\&= a_0 &\times& \verb|activation_function.derivative|(z_1) &\times& w_2 &\times& \verb|activation_function.derivative|(z_2)  &\times& \verb|cost_function.derivative|(a_2)
\end{aligned}`$

The partial derivative of cost with respect to bias of the 2nd node.
$`\begin{aligned}
\frac{\partial c}{\partial b_2} &= \frac{\partial z_2}{\partial b_2} &\times& \frac{\partial a_2}{\partial z_2} &\times& \frac{\partial c}{\partial a_2}
\\&= 1 &\times& \verb|activation_function.derivative|(z_2) &\times& \verb|cost_function.derivative|(a_2)
\end{aligned}`$

When expanding the network with more nodes per layer, TODO


## Zig

Libraries for linear algebra stuff (working with vectors, matrices).

 - [`zmath`](https://github.com/michal-z/zig-gamedev/tree/main/libs/zmath)
    - https://zig.news/michalz/fast-multi-platform-simd-math-library-in-zig-2adn
 - [`zalgebra`](https://github.com/kooparse/zalgebra)
 - [`zlm`](https://github.com/ziglibs/zlm)
