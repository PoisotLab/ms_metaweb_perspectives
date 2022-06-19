# Graph embedding demo
using EcologicalNetworks
using CairoMakie
using LinearAlgebra

# Load a series of bipartite networks
ids = getfield.(filter(n -> contains("Hadfield")(n.Reference), web_of_life()), :ID)
N = [convert(BipartiteNetwork, web_of_life(n)) for n in ids]

# Get the metaweb
M = reduce(∪, N)

# Co-occurrence matrix
C = zeros(Int64, size(M))
for n in N
    i = indexin(species(n; dims=1), species(M; dims=1))
    j = indexin(species(n; dims=2), species(M; dims=2))
    C[i, j] .+= 1
end

# Check the loss due to the embedding
rnk = collect(1:121)
L2 = [sum((adjacency(M) - prod(rdpg(M, r))) .^ 2) for r in rnk]

# Plot layout
fig = Figure(resolution=(800, 550))
ax1 = Axis(fig[1, 1], xlabel="Rank", ylabel="L2 loss", title="A", titlealign=:left)
ax2 = Axis(fig[1, 2], xlabel="Dimension 1", ylabel="Dimension 2", title="B", titlealign=:left)
ax3 = Axis(fig[2, 1], xlabel="Predicted weight", ylabel="Density", title="C", titlealign=:left)
ax4 = Axis(fig[2, 2], xlabel="Dimension 1 (left-subspace)", ylabel="Number of hosts", title="D", titlealign=:left)
fig

# Plot of rank vs. L2
lines!(ax1, rnk, L2)
fig

# Values of co-oc vs. non-co-oc. interactions at rank 25
L, R = rdpg(M, 25)

noc = density!(ax3, (L*R)[findall(iszero.(C))]; normalization=:probability)
pos = density!(ax3, (L*R)[findall(adjacency(M))]; normalization=:probability)
neg = density!(ax3, (L*R)[findall(iszero.(adjacency(M)) .& .~iszero.(C))]; normalization=:probability)
axislegend(ax3, [pos, neg, noc], ["Interactions", "Non-interactions", "No co-occurence"])
fig

# Representation of the network on axis 1

para = scatter!(ax2, L[:, 1], L[:, 2])
host = scatter!(ax2, R'[:, 1], R'[:, 2])
axislegend(ax2, [para, host], ["Parasites", "Hosts"], position=:lb)
fig

scatter!(ax4, L[:, 1], vec(sum(adjacency(M); dims=2)))
fig

save("figures/illustration.png", fig, px_per_unit=2)