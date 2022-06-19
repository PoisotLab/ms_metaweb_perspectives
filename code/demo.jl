# # Demonstration of  graph embedding using RDPG

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

fig = Figure(resolution=(800, 550))
ax1 = Axis(fig[1, 1], xlabel="Rank", ylabel="L2 loss", title="A", titlealign=:left)
ax2 = Axis(fig[1, 2], xlabel="Dimension 1", ylabel="Dimension 2", title="B", titlealign=:left)
ax3 = Axis(fig[2, 1], xlabel="Predicted weight", ylabel="Density", title="C", titlealign=:left)
ax4 = Axis(fig[2, 2], xlabel="Dimension 1 (left-subspace)", ylabel="Number of hosts", title="D", titlealign=:left)
fig

# The first thing we want to do is measure the L2 loss (the sum of squared
# errors) as a function of the rank.  The rank of the metaweb is at most the
# number of species on its least species-rich side:

rank(adjacency(M))

# The following list comprehension will perform the RDPG embedding for each rank
# between 1 and the rank of the metaweb, then multiply the left/right subspaces
# together to get an approximated network, and report the L2 error:

rnk = collect(1:rank(adjacency(M)))
L2 = [sum((adjacency(M) - prod(rdpg(M, r))) .^ 2) for r in rnk]

# We add this to the first panel of the figure:

lines!(ax1, rnk, L2; color=:black)
fig

# In the next step, we will perform the RDPG embedding at a rank of 25. We can
# check the L2 loss associated with this representation:

L2[25]

# Alternatively, we can distribute the loss across all cells in order to see the
# expected squared error *per species pair*:

L2[25]/prod(size(M))

# The left/right subspaces are given by the `rdpg` function, which internally
# performs a t-SVD and then multiply each side by the square root of the
# eigenvalues:

L, R = rdpg(M, 25)

# Checking the sizes of the L and R subspaces is important. For example, the
# size of L is:

size(L)

# The number of rows (parasite richness) and columns (subspaces dimensions) is
# correct, so we can move on to the next step (doing the same exercise for R
# will show why the L*R multiplication will give back a matrix of the correct
# dimension). We will approximate the network at rank 25, and call it P:

P = L*R

# With this matrix, we can start looking at the weight given for (i) positive
# interactions, (ii) interactions that are never observed but for which the
# species pair is observed at least once, and (iii) species pairs that are never
# observed. We simply plot the desntiy for each of these three situations and
# add it to the figure:

noc = density!(ax3, P[findall(iszero.(C))])
pos = density!(ax3, P[findall(adjacency(M))])
neg = density!(ax3, P[findall(iszero.(adjacency(M)) .& .~iszero.(C))])
axislegend(ax3, [pos, neg, noc], ["Interactions", "Non-interactions", "No co-occurence"])
fig

# Another potentially useful visualisation is to look at the position of each
# species on the first/second dimension of the relevant subspace.

para = scatter!(ax2, L[:, 1], L[:, 2]; marker=:rect)
host = scatter!(ax2, R'[:, 1], R'[:, 2])
axislegend(ax2, [para, host], ["Parasites", "Hosts"], position=:lb)
fig

# Finally, we can attempt to link the position on the first dimension to
# ecologically relevant information, like *e.g.* the number of hosts:

scatter!(ax4, L[:, 1], vec(sum(adjacency(M); dims=2)); color=:black)
fig

# This last line is saving the figure, which is used in main text.

save("figures/illustration.png", fig, px_per_unit=2)

# Finally, we compile this document to a notebook, which constitutes the
# supplementary material of the manuscript. Note that the lines to compile the
# script into a notebook are not visible in the notebook distributed as part of
# the Supp. Mat. of the article.

using Literate #src
Literate.notebook(@__FILE__, pwd(); config=Dict("execute"=>false, "name"=>"SupplementaryMaterial")) #src
