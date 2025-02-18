---
title: VAE基础
date: 2024-03-17T14:19:27+08:00
tags: [生成模型]
categories: [学习笔记]
math: true
---

## 主要框架 
考虑一个具有未知（潜在）变量 $z$，已知变量 $x$ 和固定参数 $\theta$ 的模型。
- $p\_{\theta} (x \mid z)$ 是似然函数
- $p\_{\theta} (z)$ 是先验分布
- $p\_{\theta}(z \mid x)$ 是后验分布

$$
p\_{\theta}(z \mid x) = p\_{\theta} (x , z) / p\_{\theta} (x) \text{, where } p\_{\theta} (x) =  \int p\_{\theta} (x, z) \, dz \text{ is intractable}
$$

- 近似方法

$$
q = \text{argmin}\_{q \in \mathcal{Q}} D \_{\text{KL}} (q(z) || p\_{\theta}(z \mid x))
$$

$$
D\_{KL}(q(z) || p\_{\theta}(z \mid x))
$$

- 因此，我们的最终最小化表达式为：

$$
\begin{aligned} 
 \mathbf{\psi}^*&=\underset{\mathbf{\psi}}{\text{argmin}} D\_{\mathbb{K} \mathbb{L}}(q\_{\mathbf{\psi}}(\mathbf{z}) \| p\_{\mathbf{\theta}}(\mathbf{z} \mid \mathbf{x})) 
\\\\
& =\underset{\mathbf{\psi}}{\text{argmin}} \mathbb{E}\_{q\_{\mathbf{\psi}}(\mathbf{z})}[\log q\_{\mathbf{\psi}}(\mathbf{z})-\log (\frac{p\_{\mathbf{\theta}}(\mathbf{x} \mid \mathbf{z}) p\_{\mathbf{\theta}}(\mathbf{z})}{p\_{\mathbf{\theta}}(\mathbf{x})})] 
\\\\
& =\underset{\mathbf{\psi}}{\text{argmin}} \underbrace{\mathbb{E}\_{q\_{\mathbf{\psi}}(\mathbf{z})}[\log q\_{\mathbf{\psi}}(\mathbf{z})-\log p\_{\mathbf{\theta}}(\mathbf{x} \mid \mathbf{z})-\log p\_{\mathbf{\theta}}(\mathbf{z})]}\_{\mathcal{L}(\mathbf{\theta}, \mathbf{\psi} \mid \mathbf{x})}+\log p_{\mathbf{\theta}}(\mathbf{x})  \end{aligned}
$$

- 可简化为：

{{< raw >}}
$$
\mathcal{L}(\mathbf{\theta}, \mathbf{\psi} \mid \mathbf{x})=\mathbb{E}_{q_{\mathbf{\psi}}(\mathbf{z})}[-\log p_{\mathbf{\theta}}(\mathbf{x}, \mathbf{z})+\log q_{\mathbf{\psi}}(\mathbf{z})]
$$
{{< /raw >}}

- 得到 evidence lower bound 或 **ELBO** 函数（由于两个分布的 KL 散度始终大于 0）：

{{< raw >}}
$$\mathrm{L}(\mathbf{\theta}, \mathbf{\psi} \mid \mathbf{x}) \triangleq \mathbb{E}_{q_{\mathbf{\psi}}(\mathbf{z})}[\log p_{\mathbf{\theta}}(\mathbf{x}, \mathbf{z})-\log q_{\mathbf{\psi}}(\mathbf{z})]=\mathrm{ELBO}$$
$$\mathrm{L}(\mathbf{\theta}, \mathbf{\psi} \mid \mathbf{x}) \le \log p_{\theta}(x)$$
{{< /raw >}}


- ELBO 的几种重写形式：

1. ELBO = 期望对数似然 - 从后验到先验的 KL 散度
2. ELBO = 期望对数联合分布 + 熵

{{< raw >}}
$$
 \mathrm{L}(\mathbf{\theta}, \mathbf{\psi} \mid \mathbf{x}) = \mathbb{E}_{q_{\mathbf{\psi}}(\mathbf{z})}[\log p_{\mathbf{\theta}}(\mathbf{x} \mid \mathbf{z})]-D_{\mathbb{K} \mathbb{L}}(q_{\mathbf{\psi}}(\mathbf{z}) \| p_{\mathbf{\theta}}(\mathbf{z})) \tag{1}
$$

$$
\mathrm{L}(\mathbf{\theta}, \mathbf{\psi} \mid \mathbf{x}) = \mathbb{E}_{q_{\mathbf{\psi}}(\mathbf{z})}[\log p_{\mathbf{\theta}}(\mathbf{x}, \mathbf{z})]+\mathbb{H}(q_{\mathbf{\psi}}(\mathbf{z})) \tag{2} 
$$
{{< /raw >}}


### Stochastic VI
使用以下形式替换原始表达式，主要用于大规模数据集，使用随机方式使数据更有效率：

{{< raw >}}
$$
\mathrm{L}(\mathbf{\theta}, \mathbf{\psi}_{1: N} \mid \mathcal{D})=\sum_{n=1}^N \mathrm{L}(\mathbf{\theta}, \mathbf{\psi}_n \mid \mathbf{x}_n) \approx \frac{N}{B} \sum_{\mathbf{x}_n \in \mathcal{B}}[\mathbb{E}_{q_{\psi_n}(\mathbf{z}_n)}[\log p_{\mathbf{\theta}}(\mathbf{x}_n \mid \mathbf{z}_n)+\log p_{\mathbf{\theta}}(\mathbf{z}_n)-\log q_{\mathbf{\psi}_n}(\mathbf{z}_n)]]
$$
{{< /raw >}}

## VAE

### Amortized VI
如果我们能够用 $f_{\phi}^{\text{inf}}(x_{n})$ 替换参数 $\psi_{n}$，则有：

{{< raw >}}
$$
q(\mathbf{z}_n \mid \mathbf{\psi}_n)=q(\mathbf{z}_n \mid f_{\mathbf{\phi}}^{\inf }(\mathbf{x}_n))=q_{\mathbf{\phi}}(\mathbf{z}_n \mid \mathbf{x}_n)
$$
{{< /raw >}}


最终得到 ELBO：

{{< raw >}}
$$
\mathrm{L}(\mathbf{\theta}, \mathbf{\psi} \mid \mathbf{x}) = \sum_{n=1}^N[\mathbb{E}_{q_{\mathbf{\phi}}(\mathbf{z}_n \mid \mathbf{x}_n)}[\log p_{\mathbf{\theta}}(\mathbf{x}_n, \mathbf{z}_n)-\log q_{\mathbf{\phi}}(\mathbf{z} \mid \mathbf{x}_n)]]
$$
上述的 $q_{\phi}(\cdot)$ 可以被称作一个推断网络或识别网络。

可以近似表示为（通过设置 mini-batch size = 1）：

$$
\mathbf{L}(\mathbf{\theta}, \mathbf{\psi} \mid \mathbf{x}) = N[\mathbb{E}_{q_{\mathbf{\phi}}(\mathbf{z}_n \mid \mathbf{x}_n)}[\log p_{\mathbf{\theta}}(\mathbf{x}_n, \mathbf{z}_n)-\log q_{\mathbf{\phi}}(\mathbf{z} \mid \mathbf{x}_n)]]
$$
{{< /raw >}}

### 一个简单的例子

{{< raw >}}
VAE 定义了生成模型：
$$
p_{\theta}(x, z) = p_{\theta}(z) p_{\theta}(x\mid z)
$$
其中 $p_{\theta} (z)$ 通常是高斯分布，$p_{\theta}(x \mid z)$ 通常是指数族分布的乘积（例如高斯分布或伯努利分布），其参数由神经网络解码器 $d_{\theta}(z)$ 计算。例如，对于二元观测值，我们可以使用
$$
p_{\theta}(x \mid z) = \prod_{d=1}^{D} \text{Ber}(x_{d} \mid \sigma(d_{\theta}(z)))
$$
我们可以拟合一个识别模型（类似于我们在 Atomized VI 中所做的）
$$
q_{\phi}(z \mid x) = q(z \mid e_{\phi}(x)) \approx p_{\theta}(z \mid x)
$$
和

$$
\begin{aligned}
q_{\mathbf{\phi}}(\mathbf{z} \mid \mathbf{x}) & =\mathcal{N}(\mathbf{z} \mid \mathbf{\mu}, \text{diag}(\exp (\mathbf{\ell})))
\\
(\mathbf{\mu}, \mathbf{\ell}) & =e_{\mathbf{\phi}}(\mathbf{x})
\end{aligned}
$$
{{< /raw >}}


您可以参考这个 [链接](https://danijar.com/building-variational-auto-encoders-in-tensorflow/) 进行实现。这主要使用 MNIST 数据集，注意：
> 这里，我们使用 Bernoulli 分布表示数据，将像素建模为二进制值。根据数据的类型和领域，您可能希望以不同的方式对其进行建模，例如再次将其建模为正态分布。

实现算法的[伪代码](https://arxiv.org/abs/1906.02691)（其中的Loss计算使用了对应的分布的似然）：

{{< figure src="https://p.sda1.dev/16/4b3aa4bee03c28a754449d1af6dc5b22/VAE_Ber_Gau_algo.png" title="算法伪代码" >}}

<!-- ![算法伪代码](/assets/VAE_Ber_Gau_algo.png) -->