Being able to infer *potential* interactions could serve as a significant
breakthrough in our ability to start thinking about species interaction networks
over large spatial scales [@Hortal2015SevSho]. Understanding species
interactions holds enormous potential to not only understand and more rapidly
learn about species interactions and metawebs, but also how changes in
management of a single species may impact non-target species. In a recent
overview of the field of ecological network prediction, @Strydom2021RoaPre
identified two challenges of interest to the prediction of interactions at large
scales. First, there is a relative scarcity of relevant data in most places
globally -- paradoxically, this restricts our ability to infer interactions for
locations where inference is perhaps the least required (and leaves us unable to
make inference in regions without interaction data); second, accurate predictors
are important for accurate predictions, and the lack of methods that can
leverage a small amount of *accurate* data is a serious impediment to our
predictive ability. In most places, our most reliable biodiversity knowledge is
that of a species pool (*i.e.* a set of potentially interacting species in a
given area): through the analysis of databases like GBIF or IUCN, it is possible
to construct a list of species in a region of interest; but inferring the
potential interactions between these species is difficult.

Following the definition of @Dunne2006NetStr, a metaweb is the ecological
network analogue to the species pool; specifically, it inventories all
*potential* interactions between species for a spatially delimited area (and so
captures the $\gamma$ diversity of interactions). The metaweb is not a
prediction of the network at a specific point within the spatial area it covers:
it will have a different structure, notably by having a larger connectance [see
*e.g.* @Wood2015EffSpa] and complexity [see *e.g.* @Galiana2022EcoNet], from any
of these local networks. These local networks (which capture the $\alpha$
diversity of interactions) are a subset of the metaweb's species and realized
interactions, and have been called "metaweb realizations" [@Poisot2015SpeWhy].
Differences between local networks and their metawebs are due to chance, species
abundance and co-occurrence, local environmental conditions, and local
distribution of functional traits, among others. Yet, recent results by
@Saravia2021EcoNet strongly suggest that the local (metaweb) realizations only
respond weakly to local conditions: instead, they reflect constraints inherited
by the structure of their metaweb. This establishes the metaweb structure as the
core goal of predictive network ecology, as it is a required information to
accurately produce downscaled, local predictions.

Because the metaweb represents the joint effect of functional, phylogenetic, and
macroecological processes [@Morales-Castilla2015InfBio], it holds valuable
ecological information. Specifically, it represents the "upper bounds" on what
the composition of the local networks, given a local species pool, can be [see
*e.g.* @McLeod2021SamAsy]; this information can help evaluate the ability of
ecological assemblages to withstand the effects of, for example, climate change
[@Fricke2022EffDef]. These local networks may be reconstructed given an
appropriate knowledge of local species composition and provide information on
the structure of food webs at finer spatial scales. This has been done for
example for tree-galler-parasitoid systems [@Gravel2018BriElt], fish trophic
interactions [@Albouy2019MarFis], tetrapod trophic interactions
[@Braga2019SpaAna; @OConnor2020UnvFoo], and crop-pest networks
[@Grunig2020CroFor]. In this contribution, we highlight the power in viewing
(and constructing) metawebs as *probabilistic* objects in the context of rare
interactions, discuss how a family of machine learning tools (graph embeddings
and transfer learning) can be used to overcome data limitations to metaweb
inference, and highlight how the use of metawebs introduces important questions
for the field of network ecology.

# A metaweb is an inherently probabilistic object

Treating interactions as probabilistic (as opposed to binary) events is a more
nuanced and realistic way to represent them. @Dallas2017PreCry suggested that
most links in ecological networks are cryptic, *i.e.* uncommon or hard to
observe. This argument echoes @Jordano2016SamNet: sampling ecological
interactions is difficult because it requires first the joint observation of two
species, and then the observation of their interaction. In addition, it is
generally expected weak or rare links to be more prevalent in networks than
common or strong links [@Csermely2004StrLin], compared to strong, persistent
links; this is notably the case in food chains, wherein many weaker links are
key to the stability of a system [@Neutel2002StaRea]. In the light of these
observations, we expect to see an over-representation of low-probability (rare)
interactions under a model that accurately predicts interaction probabilities.
Yet the original metaweb definition, and indeed most past uses of metawebs, was
based on the presence/absence of interactions. Moving towards *probabilistic*
metawebs, by representing interactions as Bernoulli events [see *e.g.*
@Poisot2016StrPro], offers the opportunity to weigh these rare interactions
appropriately. The inherent plasticity of interactions is important to capture:
there have been documented instances of food webs undergoing rapid
collapse/recovery cycles over short periods of time [*e.g.*
@Pedersen2017SigCol]. These considerations emphasize why metaweb predictions
should focus on quantitative (preferentially probabilistic) predictions, and
this should constrain the suite of appropriate models used to predict them.

It is important to recall that a metaweb is intended as a catalogue of all
potential interactions, which is then filtered for a given application
[@Morales-Castilla2015InfBio]. In a sense, that most ecological interactions are
elusive can call for a slightly different approach to sampling: once the common
interactions are documented, the effort required in documenting each rare
interaction will increase exponentially. Recent proposals suggest that machine
learning algorithms can also act as data generators [@Hoffmann2019MacLea]: high
quality observational data can be used to infer core rules underpinning network
structure, and be supplemented with synthetic data coming from predictive models
trained on them, thereby increasing the volume of information available for
analysis. Indeed, @Strydom2021RoaPre suggested that knowing the metaweb may
render the prediction of local networks easier, because it fixes an "upper
bound" on which interactions can exist. In this context, a probabilistic metaweb
represents an aggregation of informative priors on the interactions, elusive
information with the potential to boost our predictive ability
[@Bartomeus2016ComFra].

![Overview of the embedding process. A network (**A**), represented here as its
adjacency matrix, is converted into a lower-dimensional object (**B**) where
nodes, subgraphs, or edges have specific values (see @tbl:methods for an
overview of methods and their use for species interactions). For the purposes of
prediction, this low-dimensional object encodes feature vectors for *e.g.* the
nodes. Embedding also allows to visualize the structure in the data differently
(see @fig:illustration), much like with a principal component analysis. From a
low-dimensional feature vector, it is possible to develop predictive approaches.
Nodes in an ecological network are usually species (**C**), for which we can
leverage phylogenetic relatedness [*e.g.* @Strydom2022FooWeb] or functional
traits to fill the values of additional species we would like to project in this
space (here for nodes I, J, K, and L) from the embedding of known species (here,
nodes A, B, C, and D). Because embeddings can be projected back to a graph, this
allows us to reconstruct a network with these new species (**D**). This entire
cycle constitutes an instance of transfer learning, where the transfered
information is the representation of graph **A** through its
embedding.](figures/conceptual_2.png){#fig:embedding}

# Graph embedding offers promises for the inference of potential interactions

Graph (or Network) embedding (@fig:embedding) is a family of machine learning
techniques, whose main task is to learn a mapping function from a discrete graph
to a continuous domain [@Arsov2019NetEmb; @Chami2022MacLea]. Their main goal is
to learn a low dimensional vector representations for the nodes of the graph
(embeddings), such that key properties of the graph (e.g. local or global
structures) are retained in the embedding space [@Yan2005GraEmb]. Ecological
networks are an interesting candidate for the widespread application of
embeddings, as they tend to possess a shared structural backbone [see *e.g.*
@BramonMora2018IdeCom for food webs], which hints at structural invariants in
empirical data; assuming that these structural invariants are indeed widespread,
they would dominate the structure of networks, and therefore be adequately
captured by the first (lower) dimensions of an embedding, without the need to
measure derived aspects of their structure (*e.g.* motifs, paths, modularity,
...).

Indeed, food webs are inherently low-dimensional objects, and can be adequately
represented with less than ten dimensions [@Eklof2013DimEco; @Braga2019SpaAna].
Simulation results by @Botella2022AppGra suggest that there is no best method to
identify architectural similarities between networks, and that multiple
approaches need to be tested and compared to the network descriptor of interest.
This matches previous, more general results on graph embedding, which suggest
that different embedding algorithms yield different network embeddings
[@Goyal2018GraEmb], calling for a careful selection of the problem-specific
approach to use. In @tbl:methods, we present a selection of common graph and
node embedding methods, alongside examples of their use to predict species
interactions; most of these methods rely either on linear algebra, or on
pseudo-random walks on graphs.

One prominent family of approaches we do not discuss in the present manuscript
is Graph Neural Networks [GNN; @Zhou2020GraNeu]. GNN are, in a sense, a method
to embed a graph into a dense subspace, but belong to the family of deep
learning methods, which has its own set of practices [see *e.g.*
@Goodfellow2016DeeLea]. An important issue with methods based on deep learning
is that, because their parameter space is immense, the sample size of the data
fed into them must be similarly large (typically thousands of instances). This
is a requirement for the model to converge correctly during training, but this
assumption is unlikely to be met given the size of datasets currently available
for metawebs (or single time/location species interaction networks). This data
volume requirement is mostly absent from the techniques we list below.
Furthermore, GNN still have some challenges related to their shallow structure,
and concerns related to scalability [see @Gupta2021GraNeu for a review], which
are mostly absent from the methods listed in @tbl:methods. Assuming that the
uptake of next-generation biomonitoring techniques does indeed deliver larger
datasets on species interactions [@Bohan2017NexGlo], there is a potential for
GNN to become an applicable embedding/predictive technique in the coming years.

| Method        | Object          | Technique                        | Reference            | Application                              |
| ------------- | --------------- | -------------------------------- | -------------------- | ---------------------------------------- |
| tSNE          | nodes           | statistical divergence           | @Hinton2002StoNei    | @Gibb2021DatPro; @Cieslak2020TdiSto $^a$ |
| LINE          | nodes           | stochastic gradient descent      | @Tang2015LinLar      |                                          |
| SDNE          | nodes           | gradient descent                 | @Wang2016StrDee      |                                          |
| node2vec      | nodes           | stochastic gradient descent      | @Grover2016NodSca    |                                          |
| HARP          | nodes           | meta-strategy                    | @Chen2017HarHie      |                                          |
| DMSE          | joint nodes     | deep neural network              | @Chen2017DeeMul      | @Chen2017DeeMul $^b$                     |
| graph2vec     | sub-graph       | skipgram network                 | @Narayanan2017GraLea |                                          |
| RDPG          | graph           | SVD                              | @Young2007RanDot     | @Poisot2021ImpMam; @DallaRiva2016ExpEvo  |
| GLEE          | graph           | Laplacian eigenmap               | @Torres2020GleGeo    |                                          |
| DeepWalk      | graph           | stochastic gradient descent      | @Perozzi2014DeeOnl   | @Wardeh2021PreMam                        |
| GraphKKE      | graph           | stochastic differential equation | @Melnyk2020GraGra    | @Melnyk2020GraGra $^a$                   |
| FastEmbed     | graph           | eigen decomposition              | @Ramasamy2015ComSpe  |                                          |
| PCA           | graph           | eigen decomposition              | @S2013GraEmb         | @Strydom2021RoaPre                       |
| Joint methods | multiple graphs | multiple strategies              | @Wang2021JoiEmb      |                                          |

: Overview of some common graph embedding approaches, by type of embedded
objects, alongside examples of their use in the prediction of species
interactions. These methods have not yet been routinely used to predict species
interactions; most examples that we identified were either statistical
associations, or analogues to joint species distribution models. $^a$:
statistical interactions; $^b$: joint-SDM-like approach. Note that the row for
PCA also applies to kernel/probabilistic PCA, which are variations on the more
general method of SVD. Note further that tSNE has been included because it is
frequently used to embed graphs, including of species associations/interactions,
despite not being strictly speaking, a graph embedding technique [see *e.g.*
@Chami2022MacLea] {#tbl:methods}

The popularity of graph embedding techniques in machine learning is more than
the search for structural invariants: graphs are discrete objects, and machine
learning techniques tend to handle continuous data better. Bringing a sparse
graph into a continuous, dense vector space [@Xu2020UndGra] opens up a broader
variety of predictive algorithms, notably of the sort that are able to predict
events as probabilities [@Murphy2022ProMac]. Furthermore, the projection of the
graph itself is a representation that can be learned; @Runghen2021ExpNod, for
example, used a neural network to learn the embedding of a network in which not
all interactions were known, based on the nodes' metadata. This example has many
parallels in ecology (see @fig:embedding **C**), in which node metadata can be
given by phylogeny or functional traits. Rather than directly predicting
biological rules [see *e.g.* @Pichler2020MacLea for an overview], which may be
confounded by the sparse nature of graph data, learning embeddings works in the
low-dimensional space that maximizes information about the network structure.
This approach is further justified by the observation, for example, that the
macro-evolutionary history of a network is adequately represented by some graph
embeddings [Random dot product graphs (RDPG); see @DallaRiva2016ExpEvo]. In a
recent publication, @Strydom2022FooWeb have used an embedding (based on RDPG) to
project a metaweb of trophic interactions between European mammals, and
transferred this information to mammals of Canada, using the phylogenetic
distance between related clades to infer the values in the latent sub-space into
which the European metaweb was projected. By performing the RDPG step on
re-constructed values, this approach yields a probabilistic trophic metaweb for
mammals of Canada based on knowledge of European species, despite a limited
($\approx$ 5%) taxonomic overlap.

Graph embeddings *can* serve as a dimensionality reduction method. For example,
RDPG [@Strydom2022FooWeb] and t-SVD [truncated Singular Value Decomposition;
@Poisot2021ImpMam] typically embed networks using fewer dimensions than the
original network [the original network has as many dimensions as species, and as
many informative dimensions as trophically unique species; @Strydom2021SvdEnt].
But this is not necessarily the case -- indeed, one may perform a PCA (a special
case of SVD) to project the raw data into a subspace that improves the efficacy
of t-SNE [t-distributed stochastic neighbor embedding; @Maaten2009LeaPar]. There
are many dimensionality reductions [@Anowar2021ConEmp] that can be applied to an
embedded network should the need for dimensionality reduction (for example for
data visualisation) arise. In brief, many graph embeddings *can* serve as
dimensionality reduction steps, but not all do, neither do all dimensionality
reduction methods provide adequate graph embedding capacities. In the next
section (and @fig:illustration), we show how the amount of dimensionality
reduction can affect the quality of the embedding.

# An illustration of metaweb embedding

In this section, we illustrate the embedding of a collection of bipartite
networks collected by @Hadfield2014TalTwo, using t-SVD and RDPG [see
@Strydom2022FooWeb for the full details]. Briefly, an RDPG decomposes a network
into two subspaces (left and right), which are matrices that when multiplied
give an approximation of the original network. The code to reproduce this
example is available as supplementary material [note, for the sake of
comparison, that @Strydom2021RoaPre have an example using embedding through PCA
followed by prediction using a deep neural network on the same dataset]. The
resulting (binary) metaweb $\mathcal{M}$ has 2131 interactions between 206
parasites and 121 hosts, and its adjacency matrix has full rank (*i.e.* it
represents a space with 121 dimensions). All analyses were done using Julia
[@Bezanson2017JulFre] version 1.7.2, *Makie.jl* [@Danisch2021MakJl], and
*EcologicalNetworks.jl* [@Poisot2019EcoJl].

![Illustration of an embedding for an host-parasite metaweb, using Random Dot
Product Graphs. **A**, decrease in approximation error as the number of
dimensions in the subspaces increases. **B**, position of hosts and parasites in
the first two dimensions of their respective subspaces. **C**, predicted
interaction weight from the RDPG based on the status of the species pair in the
metaweb. **D**, relationship between the position on the first dimension and
parasite generalism.](figures/illustration.png){#fig:illustration}

The embedding of the metaweb holds several pieces of information
(@fig:illustration). In panel **A**, we show that the $L_2$ loss (*i.e.* the sum
of squared errors) between the empirical and reconstructed metaweb decreases
when the number of dimensions (rank) of the subspace increases, with an
inflection point around 25 dimensions. As discussed by @Runghen2021ExpNod, there
is often a trade-off between the number of dimensions to use (more dimensions
are more computationally demanding) and the quality of the representation. In
this instance, accepting $L_2 = 500$ as an approximation of the network means
that the error for every position in the metaweb is $\approx
\left(500/(206\times 121)\right)^{1/2}$. In @fig:illustration, panel **B**, we
show the positions of hosts and parasites on the first two dimensions of the
left and right subspaces. Note that these values largely skew negative, because
the first dimensions capture the coarse structure of the network: most pairs of
species do not interact, and therefore have negative values. In
@fig:illustration, panel **C**, we show the predicted weight (*i.e.* the result
of the multiplication of the RDGP subspaces at a rank of 25) as a function of
whether the interactions are observed, not-observed, or unknown due to lack of
co-occurrence. This reveals that the observed interactions have higher predicted
weights, although there is some overlap; the usual approach to identify
potential interactions based on this information would be a thresholding
analysis, which is outside the scope of this manuscript (and is done in the
papers cited in this illustration). Note that the values are not bound to the
unit interval, which emphasizes the need for either scaling or clamping
(although thresholding analyses are insensitive to this choice). Finally, in
@fig:illustration, panel **D**, we show that the embedding, as it captures
structural information about the network, holds ecological information; indeed,
the position of the parasite on the first dimension of the left sub-space is a
linear predictor of its number of hosts.

# The metaweb embeds both ecological hypotheses and practices

The goal of metaweb inference is to provide information about the interactions
between species at a large spatial scale. But as @Herbert1965Dun rightfully
pointed out, "[y]ou can't draw neat lines around planet-wide problems"; any
inference of a metaweb at large scales must contend with several novel, and
interwoven, families of problems. In this section, we list some of the most
pressing research priorities (*i.e.* problems that can be adressed with
subsequent data analysis or simulations), as well as issues related to the
application of these methods at the science-policy interface.

The first open research problem is the taxonomic and spatial limit of the
metaweb to embed and transfer. If the initial metaweb is too narrow in scope,
notably from a taxonomic point of view, the chances of finding another area with
enough related species (through phylogenetic relatedness or similarity of
functional traits) to make a reliable inference decreases; this would likely be
indicated by large confidence intervals during estimation of the values in the
low-rank space, meaning that the representation of the original graph is
difficult to transfer to the new problem. Alternatively, if the initial metaweb
is too large (taxonomically), then the resulting embeddings would need to
represent interactions between taxonomic groups that are not present in the new
location. This would lead to a much higher variance in the starting dataset, and
to under-dispersion in the target dataset, resulting in the potential under or
over estimation of the strength of new predicted interactions. The lack of well
documented metawebs is currently preventing the development of more concrete
guidelines. The question of phylogenetic relatedness and distribution is notably
relevant if the metaweb is assembled in an area with mostly endemic species
(*e.g.* a system that has undergone recent radiation or that has remained in
isolation for a long period of time might not have an analogous system with
which to draw knowledge from), and as with every predictive algorithm, there is
room for the application of our best ecological judgement. Because this problem
relates to distribution of species in the geographic or phylogenetic space, it
can certainly be approached through assessing the performance of embedding
transfer in simulated starting/target species pools.

The second series of problems relate to determining which area should be used to
infer the new metaweb in, as this determines the species pool that must be used.
Metawebs can be constructed by assigning interactions in a list of species
within geographic boundaries. The upside of this approach is that information at
the country level is likely to be required for biodiversity assessments, as
countries set conservation goals at the national level [@Buxton2021KeyInf], and
as quantitative instruments are designed to work at these scales
[@Turak2017UsiEss]; specific strategies are often enacted at smaller scales,
nested within a specific country [@Ray2021BioCri]. But there is no guarantee
that these boundaries are meaningful. In fact, we do not have a satisfying
answer to the question of "where does a food web stop?"; the most promising
solutions involve examining the spatial consistency of network area
relationships [see *e.g.* @Galiana2018SpaSca; @Galiana2019GeoVar;
@Galiana2021SpaSca; @Fortin2021NetEco], which is impossible in the absence of
enough information about the network itself. This suggests that inferred
metawebs should be further downscaled to allow for *a posteriori* analyses. The
methodology for metaweb downscaling is currently limited, and it is likely that
the sustained effort to characterize the spatial dependency of food web
structure will lead to more prescriptive guidelines about the need for
prediction downscaling.

The final family of problems relates less to ecological methods than to the
praxis of ecological research. Operating under the context of national
divisions, in large parts of the world, reflects nothing more than the legacy of
settler colonialism, which drives a disparity in available ecological data.
Applying any embedding to biased data does not debias them, but instead embeds
these very same biases, propagating them to the machine learning models using
embeddings to make predictions. Indeed, the use of ecological data is not an
apolitical act [@Nost2021PolEco], as data infrastructures tend to be designed to
answer questions within national boundaries (therefore placing contingencies on
what is available to be embedded), and their use often draws upon and reinforces
territorial statecraft. As per @Machen2021ThiAlg, this is particularly true when
the output of "algorithmic thinking" (*e.g.* relying on machine learning to
generate knowledge) can be re-used for governance (*e.g.* enacting conservation
decisions at the national scale). As information on species interaction networks
structure is increasingly leveraged as a tool to guide conservation actions [see
*e.g.* recent discussions for food-web based conservation; @Eero2021UseFoo;
@Naman2022FooWeb; @Stier2017IntExp], the need to appraise and correct biases
that are unwittingly propagated to algorithms when embedded from the original
data is paramount. Predictive approaches deployed at the continental scale, no
matter their intent, originate in the framework that contributed to the ongoing
biodiversity crisis [@Adam2014EleTre] and reinforced environmental injustice
[@Choudry2013SavBio; @Dominguez2020DecCon]. Particularly on Turtle Island and
other territories that were traditionally stewarded by Indigenous people, these
approaches should be replaced (or at least guided and framed) by Indigenous
principles of land management [@Eichhorn2019SteDec; @Nokmaq2021AwaSle], as part
of an "algorithm-in-the-loop" approach. Human-algorithm interactions are
notoriously difficult and can yield adverse effect [@Green2019DisInt;
@Stevenson2021AlgRis], suggesting the need to systematically study them for the
specific purpose of biodiversity governance, as well as to improve the
algorithmic literacy of decision makers. As we see artificial
intelligence/machine learning being increasingly mobilized to generate knowledge
that is lacking for conservation decisions [*e.g.* @Lamba2019DeeLea;
@MoseboFernandes2020MacLea] and drive policy decisions [@Weiskopf2022IncUpt],
our discussion of these tools need to go beyond the technical and statistical,
and into the governance consequences they can have.

**Acknowledgements:** We acknowledge that this study was conducted on land
within the traditional unceded territory of the Saint Lawrence Iroquoian,
Anishinabewaki, Mohawk, Huron-Wendat, and Omàmiwininiwak nations. TP, TS, DC,
and LP received funding from the Canadian Institute for Ecology & Evolution. FB
is funded by the Institute for Data Valorization (IVADO). TS, SB, and TP are
funded by a donation from the Courtois Foundation. CB was awarded a Mitacs
Elevate Fellowship no. IT12391, in partnership with fRI Research, and also
acknowledges funding from Alberta Innovates and the Forest Resources Improvement
Association of Alberta. M-JF acknowledges funding from NSERC Discovery Grant and
NSERC CRC. RR is funded by New Zealand’s Biological Heritage Ngā Koiora Tuku Iho
National Science Challenge, administered by New Zealand Ministry of Business,
Innovation, and Employment. BM is funded by the NSERC Alexander Graham Bell
Canada Graduate Scholarship and the FRQNT master’s scholarship. LP acknowledges
funding from NSERC Discovery Grant (NSERC RGPIN-2019-05771). TP acknowledges
financial support from the Fondation Courtois, and NSERC through the Discovery
Grants and Discovery Accelerator Supplement programs. MJF is supported by an
NSERC PDF and an RBC Post-Doctoral Fellowship.

**Conflict of interest:** The authors have no conflict interests to disclose

**Authors' contributions:** TS, and TP conceived the ideas discussed in the
manuscript. All authors contributed to writing and editing the manuscript.

**Data availability:** There is no data associated with this manuscript.

# References
