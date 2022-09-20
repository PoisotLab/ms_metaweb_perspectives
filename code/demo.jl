# # Demonstration of metaweb embedding using RDPG

using EcologicalNetworks
using CairoMakie
using LinearAlgebra

# The first step is to load a series of bipartite quantitative networks between
# hosts and parasites, coming from the Hadfield et al. paper on host-parasite
# phylogenetic structure.

ids = getfield.(filter(n -> contains("Hadfield")(n.Reference), web_of_life()), :ID)

# We convert these networks to binary bipartite networks, in order to make the
# metaweb aggregation easier.

N = [convert(BipartiteNetwork, web_of_life(n)) for n in ids]

# Transforming the array of networks into a metaweb is done through the union
# operation:

M = reduce(∪, N)

# In order to identify the pairs of species that have never been observed
# together, we use the following loop to create a co-occurence matrix:

C = zeros(Int64, size(M))
for n in N
    i = indexin(species(n; dims=1), species(M; dims=1))
    j = indexin(species(n; dims=2), species(M; dims=2))
    C[i, j] .+= 1
end

# Before moving forward with the results, we will setup the multi-panel figure
# used in main text:

fig = Figure(resolution=(800, 650))
ax1 = Axis(fig[1, 1], xlabel="Rank", ylabel=" ", title="A", titlealign=:left)
ax2 = Axis(fig[1, 2], xlabel="Dimension 1", ylabel="Dimension 2", title="B", titlealign=:left)
ax3 = Axis(fig[2, 1], xlabel="Predicted weight", ylabel="Density", title="C", titlealign=:left)
ax4 = Axis(fig[2, 2], xlabel="Dimension 1 (left-subspace)", ylabel="Number of hosts", title="D", titlealign=:left)
current_figure()

# The first thing we want to do is measure the L2 loss (the sum of squared
# errors) as a function of the rank.  The rank of the metaweb is at most the
# number of species on its least species-rich side:

rank(adjacency(M))

# The following list comprehension will perform the RDPG embedding for each rank
# between 1 and the rank of the metaweb, then multiply the left/right subspaces
# together to get an approximated network, and report the per-interaction
# averaged L2 error:

rnk = collect(1:rank(adjacency(M)))
L2 = [sum((adjacency(M) - prod(rdpg(M, r))) .^ 2) ./ prod(size(M)) for r in rnk]

# We add this to the first panel of the figure:

lines!(ax1, rnk, L2; color=:black, label="L2 Loss")
current_figure()

# In order to identify the point of inflexion at which to perform the embedding,
# we could calculate the second order derivate of the loss function w.r.t. rank,
# using the approximation provided by the finite differences method, for which
# there are closed form solutions for the first (forward difference), last
# (backward difference), and any (central difference point). This can be done on
# the singular values of the SVD yielding the RDPG used for the embedding:

singularvalues = svd(adjacency(M)).S
lines!(ax1, rnk, singularvalues ./ sum(singularvalues); color=:red, label="Variance explained")
axislegend(ax1)
current_figure()

# The finite differences methods to get the inflexion points proper is simply
# given by:

sv = copy(singularvalues)
diff = zeros(size(L2))
for i in axes(diff, 1)
    if i == 1
        diff[i] = sv[i+2] - 2sv[i+1] + sv[i]
    elseif i == length(diff)
        diff[i] = sv[i] - 2sv[i-1] + sv[i-2]
    else
        diff[i] = sv[i+1] - 2sv[i] + sv[i-1]
    end
end

# In practice, the screeplot of variance explained by each rank is noisy,
# meaning that the correct inflexion point is not necessarilly obvious from the
# series of second order derivatives (themselves an approximation due to the
# finite differences method). Here, we are placing the inflexion point at the
# value that is closest to 0 (in absolute value) - this is a slightly
# conservative approach that will keep more dimensions, but provide a better
# approximation of the network.

embedding_rank = last(findmin(abs.(diff)))

# In the next steps, we will perform the RDPG embedding at this rank. We can
# check the L2 loss associated with this representation:

L2[embedding_rank]

# The left/right subspaces are given by the `rdpg` function, which internally
# performs a t-SVD and then multiply each side by the square root of the
# eigenvalues:

L, R = rdpg(M, embedding_rank)

# Checking the sizes of the L and R subspaces is important. For example, the
# size of L is:

size(L)

# The number of rows (parasite richness) and columns (subspaces dimensions) is
# correct, so we can move on to the next step (doing the same exercise for R
# will show why the L*R multiplication will give back a matrix of the correct
# dimension). We will approximate the network at the previously identified rank,
# and call it P:

P = L * R

# With this matrix, we can start looking at the weight given for (i) positive
# interactions, (ii) interactions that are never observed but for which the
# species pair is observed at least once, and (iii) species pairs that are never
# observed. We simply plot the desntiy for each of these three situations and
# add it to the figure:

noc = density!(ax3, P[findall(iszero.(C))])
pos = density!(ax3, P[findall(adjacency(M))])
neg = density!(ax3, P[findall(iszero.(adjacency(M)) .& .~iszero.(C))])
axislegend(ax3, [pos, neg, noc], ["Interactions", "Non-interactions", "No co-occurence"])
current_figure()

# From this panel, it is rather clear that a lot of interactions without
# documented co-occurrence have a lower assigned weight, and are therefore less
# likely to be feasible. Turning this information into a prediction of
# interaction existence can be carried out as a binary classification exercise
# using *e.g.* weight thresholding.

# Another potentially useful visualisation is to look at the position of each
# species on the first/second dimension of the relevant subspace.

para = scatter!(ax2, L[:, 1], L[:, 2]; marker=:rect)
host = scatter!(ax2, R'[:, 1], R'[:, 2])
axislegend(ax2, [para, host], ["Parasites", "Hosts"], position=:lb)
current_figure()

# Finally, we can attempt to link the position on the first dimension to
# ecologically relevant information, like *e.g.* the number of hosts:

scatter!(ax4, L[:, 1], vec(sum(adjacency(M); dims=2)); color=:black)
current_figure()

# This last line is saving the figure, which is used in main text.

save("figures/illustration.png", fig, px_per_unit=2)

# Finally, we compile this document to a notebook, which constitutes the
# supplementary material of the manuscript. Note that the lines to compile the
# script into a notebook are not visible in the notebook distributed as part of
# the Supp. Mat. of the article.

using Literate #src
Literate.notebook(@__FILE__, pwd(); config=Dict("execute" => false, "name" => "SupplementaryMaterial")) #src
