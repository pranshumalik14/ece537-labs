### A Pluto.jl notebook ###
# v0.16.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ a32840fb-a432-408c-be05-d37f4f95b0e7
using Distributions, StatsBase, StatsPlots, LinearAlgebra, LaTeXStrings, PlutoUI, Measures

# ╔═╡ e0b06eb0-0cea-4491-8bad-1d0b19dc2ffc
PlutoUI.TableOfContents()

# ╔═╡ f974e162-359a-11ec-3dd3-9ff38609caa5
md"

# ECE537: Lab 2 Report
> _It is recommended to access this report by opening the `html` file on the browser or by clicking [here](https://pranshumalik14.github.io/ece537-labs/lab2/lab2.jl.html)_.

In the first part of the lab, we will be creating and analyzing joint Gaussian distributions as a part of which we will be extracting marginal densities of correlated joint random variables. In the second part, we will be empirically verifying the central limit theorem and the law of large numbers using uniform (univariate) random variables by testing for the relevant convergence cirtieria.

Throughout this lab, the [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) package in Julia has been utilized to be able to use the probabllity constructs in code.

"

# ╔═╡ cf6f5454-a62b-4d82-81c7-4b2c98d6fa22
md"

## 1.Simulating Bivariate Gaussian Distributions

The multivariate normal (or Gaussian) distribution is a multidimensional generalization of the normal distribution. The probability density function of a $n$-dimensional multivariate normal distribution with mean vector $\boldsymbol{\mu}$ and (symmetric, positive definite) covariance matrix $\boldsymbol{\Sigma}$ is:

$f_{\mathbf{X}}(\mathbf{x}; \boldsymbol{\mu},\boldsymbol{\Sigma}) = \frac{1}{\sqrt{(2\pi)^n \text{det}(\boldsymbol{\Sigma})}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\intercal\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right).$

We will proceed with using its implementation provided as the `MvNormal` distribution struct or type, with $n=2$ to make it a bivariate distribution.


For  with individual means $\mu_1, \mu_2$, standard deviations $\sigma_1, \sigma_2$, and correlation coefficient $\rho$, the bivariate normal distribution can be defined as:

$X \sim \mathcal{N}\left(\boldsymbol{\mu}=
\begin{bmatrix}
	\mu_1 \\
	\mu_2
\end{bmatrix},
\boldsymbol{\Sigma}=\begin{bmatrix}
	\sigma_1^2 & \rho\sigma_1\sigma_2 \\
	\rho\sigma_1\sigma_2 & \sigma_2^2
\end{bmatrix}\right).$

"

# ╔═╡ fb4e913f-eac0-45d5-9828-32cc4f0d9fbe
md"

### 1.1 Numerical Simulation

We will now define and sample various bivariate normal distributions and inspect their coverage and densities through scatter plots. Below is a slider for the number of samples we wish to take per distribution in this section.

𝑁₁ = $(@bind N₁ Slider(50:50:1600; show_value=true, default=800))

We will start by defining an uncorrelated bivariate normal distribution with zero mean and unit variance, $X \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$. After sampling the distribution $N_1$ times, we also show the scatter and contour plots for the variate.

"

# ╔═╡ e2b52309-9b20-4ee8-b22c-a10bbc01ce37
function matrixtotuple(x::AbstractMatrix{T}) where T
	r,c = size(x)	
	pts = Vector{Tuple}(undef,c)
	@time @inbounds for i in 1:c
		pts[i] = Tuple(x[:,i])
	end
	return pts
end

# ╔═╡ 11861633-309d-4a19-bab0-88cc752fb6a5
𝑋 = MvNormal(zeros(2), I(2));

# ╔═╡ 5f5998fc-8954-4738-ab9e-54e5db821eb2
𝑋samples = rand(𝑋, N₁) |> matrixtotuple;

# ╔═╡ fae5f520-cd44-4595-b5b8-e87fe215d15d
begin
	scatter(𝑋samples; alpha=450.0/N₁, legend=false, markerstrokewidth=0)
	xlabel!(L"X_1")
	ylabel!(L"X_2")
	title!(L"(X_1, X_2) \sim \mathcal{N}(\mathbf{0}, \mathbf{I})")
end

# ╔═╡ 1f7c1317-10e9-4e3f-ba24-9fa337a5a4da
begin
	marginal = marginalkde([x[1] for x ∈ 𝑋samples], [x[2] for x ∈ 𝑋samples]; levels=5)
	title = plot(;title="Contours and Marginal Densities of N(0,I)", 
		framestyle=nothing, showaxis=false, xticks=false, yticks=false, margin=0mm)
	plot(marginal, title; layout=@layout([A; B{0.01h}]))
end

# ╔═╡ 4caafd5b-f077-4238-b9a5-5b72d0b5f9ac
md"

On first look, and especially after visually adjusting the aspect ratios, the $X_1$ and $X_2$ marginal densities look nearly identical in trend and suggest that they are the same as the partitioned distributions of the bivariate. We will verify and formally present the this idea in the next section.

Now, we will define a collection of bivariate normal distributions with varying degree of correlation to see its effects qualitatively.

Since, the covariance matrix depends on the correlation coefficient, $\rho$ and the standrd devaiations $\sigma_1, \sigma_2$, we will define a generic function, $\boldsymbol{\Sigma}(\sigma_1, \sigma_2, \rho)$, for generating valid covariance matrices based on these parameters.

"

# ╔═╡ 9f1001fc-f7cb-47e6-abc2-3c46d05d4ade
function Σ(σ₁, σ₂, ρ)
	@assert abs(ρ) ≤ 1
	
	# add epsilon along diagonal for numerical stability during cholesky decomposition
	ϵ = 1e-6
	return ϵ*I(2) + [σ₁^2    ρ*σ₁*σ₂;
				 	 ρ*σ₁*σ₂ σ₂^2]
end

# ╔═╡ a647164f-8585-40bf-9036-729da288a0ca
md"

To see the effects of correlation in bivariate distributions, we will fix the mean across all distributions and only vary $\rho$. Therefore, we define $5$ distributions with mean $\boldsymbol{\mu} = \begin{bmatrix}1 & 2\end{bmatrix}^\intercal$, variance $\sigma_1^2 = 2$, $\sigma_2^2 = 1$, and correlation coefficient ranging from $\rho = -1.0$ to $\rho = 1.0$ in increments of $0.5$. These distributions are stored in the $\mathbf{X}_\rho$ vector.

"

# ╔═╡ bfc79861-8ca7-4aa0-b9a2-0e9e77c98be5
μ = [1; 2];

# ╔═╡ c7a54c8f-f68a-4f6f-ab33-0aff066f4542
𝑋ᵨ = [MvNormal(μ, Σ(√2, 1, ρ)) for ρ ∈ -1:0.5:1];

# ╔═╡ ae16d377-2f11-4b64-baa1-d1ad1f629b9b
𝑋ᵨsamples = [rand(𝑋, N₁) |> matrixtotuple for 𝑋 ∈ 𝑋ᵨ]

# ╔═╡ 06ba1279-2c12-45f9-88d8-cdcfd4000181
let
	p1 = scatter(𝑋ᵨsamples[1]; label=L"\rho=-1.0", alpha=250.0/N₁, markerstrokewidth=0)
	p2 = scatter(𝑋ᵨsamples[2]; label=L"\rho=-0.5", alpha=250.0/N₁, markerstrokewidth=0)
	p3 = scatter(𝑋ᵨsamples[3]; label=L"\rho=0.0", alpha=250.0/N₁, markerstrokewidth=0)
	p4 = scatter(𝑋ᵨsamples[4]; label=L"\rho=0.5", alpha=250.0/N₁, markerstrokewidth=0)
	p5 = scatter(𝑋ᵨsamples[5]; label=L"\rho=1.0", alpha=250.0/N₁, markerstrokewidth=0)
	p6 = plot(; framestyle=nothing, showaxis=false, xticks=false, yticks=false, 
		margin=0mm)
	p7 = plot(; title="Correlated Bivariate Normals, Xᵨ", framestyle=nothing, 
		showaxis=false, xticks=false, yticks=false, margin=0mm)
	plot(p1, p2, p3, p4, p5, p6, p7; layout=@layout([A B; C D; E F; G{-0.05h}]))
end

# ╔═╡ a4e41a9b-e396-42df-a9ef-04c9f7206cc9
md"

### 1.2 Summary of Results

Here we will test for a fixed number of samples, $N=100$, and observe the characteristics of the bivariate distributions and if they match our expectations.

"

# ╔═╡ 7b833bca-d15e-46e2-a0c4-00ab9b28f178
N = 100; # fixed number of samples

# ╔═╡ 3626b71b-e650-4bdc-af24-8523e91a2f4e
md"

The scatter plot for $X \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ is shown below.

"

# ╔═╡ 3132eb86-e219-4990-8f57-1be55667877a
𝑋fixedsamples = rand(𝑋, N) |> matrixtotuple;

# ╔═╡ 27d7df82-221c-4ff1-9e6e-996bd1470122
begin
	scatter(𝑋fixedsamples; alpha=0.65, legend=false, markerstrokewidth=0, 
		aspect_ratio=:equal)
	xlabel!(L"X_1")
	ylabel!(L"X_2")
	title!(L"(X_1, X_2) \sim \mathcal{N}(\mathbf{0}, \mathbf{I})")
end

# ╔═╡ bbc3149b-5c12-4fe3-8fb9-518703e9b733
md"

As can be seen, both $X_1$ and $X_2$ show equal spread around zero and will thus also have nearly identical marginal densities $f_{X_1}$ and $f_{X_2}$.

Similarly, we will sample the correlated collection of bivariate normals and display their scatter plots below.

"

# ╔═╡ b93ac3f0-dbe3-4608-8149-2191dc4441c5
𝑋ᵨfixedsamples = [rand(𝑋, N) |> matrixtotuple for 𝑋 ∈ 𝑋ᵨ];

# ╔═╡ 18dfc0b8-cc89-43b9-bb42-05700c467120
let
	p1 = scatter(𝑋ᵨfixedsamples[1]; label=L"\rho=-1.0", alpha=45.0/N, 
		markerstrokewidth=0)
	p2 = scatter(𝑋ᵨfixedsamples[2]; label=L"\rho=-0.5", alpha=45.0/N, 
		markerstrokewidth=0)
	p3 = scatter(𝑋ᵨfixedsamples[3]; label=L"\rho=0.0", alpha=45.0/N, 
		markerstrokewidth=0)
	p4 = scatter(𝑋ᵨfixedsamples[4]; label=L"\rho=0.5", alpha=45.0/N, 
		markerstrokewidth=0)
	p5 = scatter(𝑋ᵨfixedsamples[5]; label=L"\rho=1.0", alpha=45.0/N, 
		markerstrokewidth=0)
	p6 = plot(; framestyle=nothing, showaxis=false, xticks=false, yticks=false, 
		margin=0mm)
	p7 = plot(; title="Correlated Bivariate Normals, Xᵨ", framestyle=nothing, 
		showaxis=false, xticks=false, yticks=false, margin=0mm)
	plot(p1, p2, p3, p4, p5, p6, p7; layout=@layout([A B; C D; E F; G{-0.05h}]))
end

# ╔═╡ a0d2913a-2d19-4723-bc77-2ccddab7e8cf
md"

Thus, we can observe in the plots for $\mathbf{X}_\rho$ above, and in section 1.1 as well, that the correlation coefficient captures the degree to which the joint variates have an affine relationship. For $\rho=-1$, it is exactly the case that $X_2 \propto -X_1$ about the mean, and the opposite, i.e. $X_2 \propto X_1$, for $\rho=1$. For $\rho=0$, there is no single affine relationship explaining the spread of the bivariate distribution and the random variates are thus uncorrelated.

"

# ╔═╡ 05318f3b-2b49-4e73-b147-d4402cbc24f0
md"

Note, that for any partition of a multivariate normal distribution $\mathbf{Z} \sim \mathcal{N}(\boldsymbol{\mu}_\mathbf{Z}, \boldsymbol{\Sigma}_\mathbf{Z})$ will have the case that

$\mathbf{Z} = (\mathbf{X}, \mathbf{Y}) \sim \mathcal{N}\left(
\begin{bmatrix}
\boldsymbol{\mu}_\mathbf{X}\\
\boldsymbol{\mu}_\mathbf{Y}
\end{bmatrix}, 
\begin{bmatrix}
\boldsymbol{\Sigma}_{\mathbf{X}\mathbf{X}} && \boldsymbol{\Sigma}_{\mathbf{X}\mathbf{Y}}\\
\boldsymbol{\Sigma}_{\mathbf{Y}\mathbf{X}} && \boldsymbol{\Sigma}_{\mathbf{Y}\mathbf{Y}}
\end{bmatrix}
\right)$

where,

$\boldsymbol{\mu}_\mathbf{X} = E[\mathbf{X}]\text{, } \boldsymbol{\Sigma}_{\mathbf{X}\mathbf{X}} = E[(\mathbf{X}−\boldsymbol{\mu}_\mathbf{X})(\mathbf{X}−\boldsymbol{\mu}_\mathbf{X})^\intercal] \text{, and } \boldsymbol{\Sigma}_{\mathbf{X}\mathbf{Y}} = E[(\mathbf{X}−\boldsymbol{\mu}_\mathbf{X})(\mathbf{Y}−\boldsymbol{\mu}_\mathbf{Y})^\intercal]$

Similar definitions follow for $\mathbf{Y}$, where $\boldsymbol{\Sigma}_{\mathbf{Y}\mathbf{X}} = \boldsymbol{\Sigma}_{\mathbf{X}\mathbf{Y}}^\intercal$.

The marginal distributions, as we visually guessed before, are

$\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}_\mathbf{X}, \boldsymbol{\Sigma}_{\mathbf{X}\mathbf{X}}) \text{, and } \mathbf{Y} \sim \mathcal{N}(\boldsymbol{\mu}_\mathbf{Y}, \boldsymbol{\Sigma}_{\mathbf{Y}\mathbf{Y}}).$ 

In higher dimensions, these will also be joint normal distributions. To verify this general result for the simpler bivariate case, we will choose $\mathbf{X}_\rho$ with $\rho=0.5$ above and extract the marginal density for $X_1$ which should approximately be the univariate $\mathcal{N}(\mu_1=1, \sigma_1^2 = 2)$.

"

# ╔═╡ 1c76cbbf-463f-49ac-b0f6-41c2144fc04e
𝑋₁samples = [X[1] for X ∈ 𝑋ᵨfixedsamples[4]];

# ╔═╡ 8a067a43-cecc-4b8c-b924-64b213874f05
mean(𝑋₁samples) # ≈ 1

# ╔═╡ b1520d42-b028-4bcd-bab3-5074a741712e
var(𝑋₁samples; corrected=false) # ≈ 2

# ╔═╡ 14710a6d-db15-4bbd-89ea-f10fcddffc70
𝑋₁hist = fit(Histogram, 𝑋₁samples, nbins=10);

# ╔═╡ c946a8a0-10cd-4780-8c8d-202dc6b32a0e
let
	normalhist = normalize(𝑋₁hist; mode=:pdf)
	plot(normalhist; label="Marginal Histogram")
	xlabel!(L"X_1")
	ylabel!("Normalized Density")
	title!("Normalized Histogram of X₁ in Xᵨ(ρ=0.5)")
	plot!(fit(Normal, 𝑋₁samples); color="orange", linewidth=5, label="Marginal pdf")
end

# ╔═╡ 43f302e7-a023-4acc-9a77-d5fb6da5e2c5
md"

Note that the mean and variance are sufficient statistics to construct the pdf of a collection of samples with an underlying normal distribution, and therefore, we have empiricially verified this result for the bivariate case.

"

# ╔═╡ db4852fe-28f0-470d-b699-a824b77c3961
md"

## 2. Empirical Verification of the Law of Large Numbers and the Central Limit Theorem

In this section, we will begin by verifing the strong law of large numbers, which says that for the sample mean, or the unbiased estimator for $E[X]$, $M_n =
\frac{1}{n}\sum_{i=1}^{n}X_i$ of an independent and identically distributed (iid) sequence of random variables with the pdf of $X$, converges almost surely to the theoretical mean, i.e. $M_n \overset{a.s}{\longrightarrow} E[X], n \rightarrow \infty$.This means that the probability associated with the set of realizations of the random process $M_n$ that converge to $E[X]$ is equal to $1$.

Then we will empirically verify the central limit theorem for the same sequence $X_1, X_2, \ldots$ of iid random variables above with mean $\mu$ and variance $\sigma^2$. The theorem states that a process $Z_n = \frac{1}{\sigma\sqrt{n}}\sum_{i=1}^{n}(X_i - \mu)$, will converge in distribution to the univariate normal with zero mean and unit variance, i.e. $Z_n \overset{a.s}{\longrightarrow} \mathcal{N}(0,1), n \rightarrow \infty$.

We will use a sequaence of iid uniform random variables $X \sim \mathcal{U}(0,1)$ for undertaking both verifications.

"

# ╔═╡ e8245df1-9ddf-4869-9748-d14d0af34cff
md"

### 2.1 Numerical Simulation

Below is a slider for the number of samples we wish to take per random process in this section.

"

# ╔═╡ c7e05572-f25d-451f-a16d-7d23b95b57f3
md"

𝑁₂ = $(@bind N₂ Slider(1:1:100; show_value=true, default=50))

"

# ╔═╡ 8dcd76cd-2f90-4dd7-8f56-9e6eaa716498
md"

We begin by defining $N_2$ number of uniform random variables, $U_i$, over the support $[0,1]$, which can be viewed as a sequence of iid random variables to construct a random process.

"

# ╔═╡ 024931b6-07da-4ed5-ad29-4e78a05023b1
𝑈(n) = [Uniform(0, 1) for n ∈ 1:n]

# ╔═╡ 6278ceaa-95a1-41a6-9ba1-01102ed3bab3
𝑈ₙ = 𝑈(N₂);

# ╔═╡ e0fcff32-4e61-4422-b040-949edded7ec1
md"

We then define the $n-$degree sum, $S_n$, of the sequence $U_n$,

$S_n = \sum_{i=1}^{n}U_i.$

Then, we would like to check if the value of the unbiased mean estimator $M_n = \frac{S_n}{n}$ converges to $E[X]$ as $n$ increases.

"

# ╔═╡ 55a6f583-eb2a-4969-bf00-e51b1abe6fa3
𝑆(𝑈ₙ) = rand.(𝑈ₙ) |> sum

# ╔═╡ b0fa3ca1-5755-49a7-8f2d-f6153bc2c081
𝑀ₙ = 𝑆(𝑈ₙ)/length(𝑈ₙ)

# ╔═╡ 90bbbba1-2293-4455-9892-d54c080f0f79
md"

### 2.2 Summary of Results

Here we will empirically test for the convergence of $M_n$ and $Z_n$. We begin by defining $M_n$ by composing $S_n$ and $U_n$ and dividing by the number of iid variables we consider in the sequence, $n$. Then, we test for convergence by plotting $M_n$ for $1 \leq n \leq 1000$ and observe that it is indeed showing convergence to the true mean of the underlying uniform distribution.


"

# ╔═╡ b3454e47-4c42-47ca-9336-497ecb3ce0e1
𝑀(n) = (𝑆 ∘ 𝑈)(n)/n # composing the S and U definitions over n

# ╔═╡ b43f46c7-8d60-40ba-a0e9-cc719a60252f
begin
	plot(𝑀, 1:1000; label=L"M_n")
	plot!(n->0.5, 1:1000; line=:dash, linewidth=3, label="True Mean")
	title!("Convergence of Mₙ")
	xlabel!("n")
	ylabel!("Statistic")
end

# ╔═╡ 446c67d5-7639-4893-8202-f235f1d0585f
md"

We can now use the definition of the process $M_n$ to further define $Z_n$ as

$Z_n = \frac{S_n - n\mu}{\sigma\sqrt{n}} = \frac{\sqrt{n}}{\sigma}(M_n - \mu).$

"

# ╔═╡ 86e60e23-79b0-4dec-9623-105277c817dc
𝑍(n, μ, σ) = (𝑀(n) - μ) * (√n)/σ

# ╔═╡ 6a315e06-2479-4528-a7d4-bf28ce329dd8
md"

Now, we will test for the convergence of the process $Z_n$ by sampling the $Z_{100}$ random variable $N=1000$ times. We will plot the normalized histogram and also the best-fit normal distribution and check if it is close to the theoretical convergence limit $\mathcal{N}(0,1)$.

"

# ╔═╡ 8e8d5823-c6e4-49e2-8aa9-1b5829d34a69
𝑍₁₀₀samples = [𝑍(100, 0.5, 1/√12) for n ∈ 1:1000];

# ╔═╡ 88197dce-a633-4f06-b011-49205f96dc5e
𝑍₁₀₀hist = fit(Histogram, 𝑍₁₀₀samples, nbins=15);

# ╔═╡ 5d849655-04e3-450f-94df-c53b302d38f0
let
	normalhist = normalize(𝑍₁₀₀hist; mode=:pdf)
	plot(normalhist; label="Empirical Histogram")
	xlabel!(L"Z_{100}")
	ylabel!("Normalized Density")
	title!("Normalized Histogram of Z₁₀₀ over 1000 Samples")
	plot!(fit(Normal, 𝑍₁₀₀samples); color="orange", linewidth=5, label="Estimated Normal pdf")
end

# ╔═╡ dd96ee87-f278-4e37-ace1-1dad265a76ad
mean(𝑍₁₀₀samples) # ≈ 0

# ╔═╡ ea0e123c-6b5d-4ae8-9fb9-54b3ec2b5256
var(𝑍₁₀₀samples; corrected=false) # ≈ 1

# ╔═╡ ad5fea86-aac1-4880-8b6e-54bb3265730d
md"

Note that the normal distribution constructed using the mean and uncorrected variance of the $Z_{100}$ samples corresponds to the normal maximum likelihood estimate (MLE) of the data, which is close to the theoretical limit of converging to the underlying zero mean and unit variance normal variate structure of $Z_n$ for $n\rightarrow \infty$. 

"

# ╔═╡ 64984036-d40c-4849-b268-b705e3c90bdb
md"

## 3. Code

Note that this lab report can be run on the cloud and viewed as is on the github repository page [here](https://pranshumalik14.github.io/ece537-labs/lab2/lab2.jl.html). All code for the notebook can be accessed [here](https://github.com/pranshumalik14/ece537-labs).

"

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Measures = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
Distributions = "~0.25.21"
LaTeXStrings = "~1.2.1"
Measures = "~0.3.1"
PlutoUI = "~0.7.16"
StatsBase = "~0.33.12"
StatsPlots = "~0.14.28"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "485ee0867925449198280d4af84bdb46a2a404d0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.0.1"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra"]
git-tree-sha1 = "2ff92b71ba1747c5fdd541f8fc87736d82f40ec9"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.4.0"

[[Arpack_jll]]
deps = ["Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "e214a9b9bd1b4e1b4f15b22c0994862b66af7ff7"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.0+3"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f2202b55d816427cd385a9a4f3ffb226bee80f99"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+0"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "0541d306de71e267c1a724f84d44bbc981f287b4"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.10.2"

[[Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "75479b7df4167267d75294d14b58244695beb2ac"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.2"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "a851fec56cb73cfdf43762999ec72eff5b86882a"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.15.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "31d0151f5716b655421d9d75b7fa74cc4e744df2"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.39.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[DataValues]]
deps = ["DataValueInterfaces", "Dates"]
git-tree-sha1 = "d88a19299eba280a6d062e135a43f00323ae70bf"
uuid = "e7dc6d0d-1eca-5fa6-8ad6-5aecde8b7ea5"
version = "0.4.13"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "09d9eaef9ef719d2cd5d928a191dc95be2ec8059"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.5"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["ChainRulesCore", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "15dad92b6a36400c988de3fc9490a372599f5b4c"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.21"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "463cb335fa22c4ebacfd1faba5fde14edb80d96c"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.4.5"

[[FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "8756f9935b7ccc9064c6eef0bff0ad643df733a3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.7"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "dba1e8614e98949abfa60480b13653813d8f0157"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+0"

[[GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "d189c6d2004f63fd3c91748c458b09f26de0efaa"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.61.0"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "cafe0823979a5c9bff86224b3b8de29ea5a44b2e"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.61.0+0"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "7bf67e9a481712b3dbe9cb3dac852dc4b1162e02"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+0"

[[Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "14eece7a3308b4d8be910e265c724a6ba51a9798"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.16"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "8a954fed8ac097d5be04921d595f741115c1b2ad"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+0"

[[Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[HypertextLiteral]]
git-tree-sha1 = "5efcf53d798efede8fee5b2c8b09284be359bf24"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.2"

[[IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[Interpolations]]
deps = ["AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "61aa005707ea2cebf47c8d780da8dc9bc4e0c512"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.4"

[[InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "f0c6489b12d28fb4c2103073ec7452f3423bd308"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.1"

[[IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

[[KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "591e8dc09ad18386189610acafb970032c519707"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.3"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "c7f1c695e06c01b95a67f0cd1d34994f3e7db104"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.2.1"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a8f4f279b6fa3c3c4f1adadd78a621b13a506bce"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.9"

[[LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "761a393aeccd6aa92ec3515e428c26bf99575b3b"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+0"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "340e257aada13f95f98ee352d316c3bed37c8ab9"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+0"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "6193c3815f13ba1b78a51ce391db8be016ae9214"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.4"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "5455aef09b40e5020e1520f551fa3135040d4ed0"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2021.1.1+2"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "5a5bc6bf062f0f95e62d0fe0a2d99699fed82dd9"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.8"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "8d958ff1854b166003238fe191ec34b9d592860a"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.8.0"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "16baacfdc8758bc374882566c9187e785e85c2f0"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.9"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[Observables]]
git-tree-sha1 = "fe29afdef3d0c4a8286128d4e45cc50621b1e43d"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.4.0"

[[OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "c0e9e582987d36d5a61e650e6e543b9e44d9914b"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.7"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7937eda4681660b4d6aeeecc2f7e1c81c8ee4e2f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+0"

[[OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "15003dcb7d8db3c6c857fda14891a539a8f2705a"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.10+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "4dd403333bcf0909341cfe57ec115152f937d7d8"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.1"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "f19e978f81eca5fd7620650d7dbea58f825802ee"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.1.0"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "b084324b4af5a438cd63619fd006614b3b20b87b"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.15"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs"]
git-tree-sha1 = "68a8a1f4d5763271d38847f0e22d67a7a61b6565"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.23.0"

[[PlutoUI]]
deps = ["Base64", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "4c8a7d080daca18545c56f1cac28710c362478f3"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.16"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Ratios]]
deps = ["Requires"]
git-tree-sha1 = "01d341f502250e81f6fec0afe662aa861392a3aa"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.2"

[[RecipesBase]]
git-tree-sha1 = "44a75aa7a527910ee3d1751d1f0e4148698add9e"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.2"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "7ad0dfa8d03b7bcf8c597f59f5292801730c55b8"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.4.1"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "f45b34656397a1f6e729901dc9ef679610bd12b5"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.8"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "2d57e14cd614083f132b6224874296287bfa3979"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.8.0"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3c76dde64d03699e074ac02eb2e8ba8254d428da"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.13"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "eb35dcc66558b2dda84079b9a1be17557d32091a"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.12"

[[StatsFuns]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "95072ef1a22b057b1e80f73c2a89ad238ae4cfff"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.12"

[[StatsPlots]]
deps = ["Clustering", "DataStructures", "DataValues", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "eb007bb78d8a46ab98cd14188e3cec139a4476cf"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.14.28"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "2ce41e0d042c60ecd131e9fb7154a3bfadbf50d3"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.3"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "fed34d0e71b91734bf0a7e10eb1bb05296ddbcd0"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.0"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll"]
git-tree-sha1 = "2839f1c1296940218e35df0bbb220f2a79686670"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.18.0+4"

[[Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "80661f59d28714632132c73779f8becc19a113f2"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.4"

[[WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "c45f4e40e7aafe9d086379e5578947ec8b95a8fb"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+0"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─e0b06eb0-0cea-4491-8bad-1d0b19dc2ffc
# ╟─f974e162-359a-11ec-3dd3-9ff38609caa5
# ╠═a32840fb-a432-408c-be05-d37f4f95b0e7
# ╟─cf6f5454-a62b-4d82-81c7-4b2c98d6fa22
# ╟─fb4e913f-eac0-45d5-9828-32cc4f0d9fbe
# ╟─e2b52309-9b20-4ee8-b22c-a10bbc01ce37
# ╠═11861633-309d-4a19-bab0-88cc752fb6a5
# ╠═5f5998fc-8954-4738-ab9e-54e5db821eb2
# ╟─fae5f520-cd44-4595-b5b8-e87fe215d15d
# ╟─1f7c1317-10e9-4e3f-ba24-9fa337a5a4da
# ╟─4caafd5b-f077-4238-b9a5-5b72d0b5f9ac
# ╠═9f1001fc-f7cb-47e6-abc2-3c46d05d4ade
# ╟─a647164f-8585-40bf-9036-729da288a0ca
# ╠═bfc79861-8ca7-4aa0-b9a2-0e9e77c98be5
# ╠═c7a54c8f-f68a-4f6f-ab33-0aff066f4542
# ╠═ae16d377-2f11-4b64-baa1-d1ad1f629b9b
# ╟─06ba1279-2c12-45f9-88d8-cdcfd4000181
# ╟─a4e41a9b-e396-42df-a9ef-04c9f7206cc9
# ╠═7b833bca-d15e-46e2-a0c4-00ab9b28f178
# ╟─3626b71b-e650-4bdc-af24-8523e91a2f4e
# ╠═3132eb86-e219-4990-8f57-1be55667877a
# ╟─27d7df82-221c-4ff1-9e6e-996bd1470122
# ╟─bbc3149b-5c12-4fe3-8fb9-518703e9b733
# ╠═b93ac3f0-dbe3-4608-8149-2191dc4441c5
# ╟─18dfc0b8-cc89-43b9-bb42-05700c467120
# ╟─a0d2913a-2d19-4723-bc77-2ccddab7e8cf
# ╟─05318f3b-2b49-4e73-b147-d4402cbc24f0
# ╠═1c76cbbf-463f-49ac-b0f6-41c2144fc04e
# ╠═8a067a43-cecc-4b8c-b924-64b213874f05
# ╠═b1520d42-b028-4bcd-bab3-5074a741712e
# ╠═14710a6d-db15-4bbd-89ea-f10fcddffc70
# ╟─c946a8a0-10cd-4780-8c8d-202dc6b32a0e
# ╟─43f302e7-a023-4acc-9a77-d5fb6da5e2c5
# ╟─db4852fe-28f0-470d-b699-a824b77c3961
# ╟─e8245df1-9ddf-4869-9748-d14d0af34cff
# ╟─c7e05572-f25d-451f-a16d-7d23b95b57f3
# ╟─8dcd76cd-2f90-4dd7-8f56-9e6eaa716498
# ╠═024931b6-07da-4ed5-ad29-4e78a05023b1
# ╠═6278ceaa-95a1-41a6-9ba1-01102ed3bab3
# ╟─e0fcff32-4e61-4422-b040-949edded7ec1
# ╠═55a6f583-eb2a-4969-bf00-e51b1abe6fa3
# ╠═b0fa3ca1-5755-49a7-8f2d-f6153bc2c081
# ╟─90bbbba1-2293-4455-9892-d54c080f0f79
# ╠═b3454e47-4c42-47ca-9336-497ecb3ce0e1
# ╟─b43f46c7-8d60-40ba-a0e9-cc719a60252f
# ╟─446c67d5-7639-4893-8202-f235f1d0585f
# ╠═86e60e23-79b0-4dec-9623-105277c817dc
# ╟─6a315e06-2479-4528-a7d4-bf28ce329dd8
# ╠═8e8d5823-c6e4-49e2-8aa9-1b5829d34a69
# ╠═88197dce-a633-4f06-b011-49205f96dc5e
# ╟─5d849655-04e3-450f-94df-c53b302d38f0
# ╠═dd96ee87-f278-4e37-ace1-1dad265a76ad
# ╠═ea0e123c-6b5d-4ae8-9fb9-54b3ec2b5256
# ╟─ad5fea86-aac1-4880-8b6e-54bb3265730d
# ╟─64984036-d40c-4849-b268-b705e3c90bdb
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
