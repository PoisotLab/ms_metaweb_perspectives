Being able to infer *potential* interactions could be the catalyst for
significant breakthroughs in our ability to start thinking about species
interaction networks over large spatial scales [@Hortal2015SevSho].
Understanding species interactions holds enormous potential to not only
understand and more rapidly learn about species interactions and metawebs, but
also how changes in management of a single species may impact non-target
species. In a recent overview of the field of ecological network prediction,
@Strydom2021RoaPre identified two challenges of interest to the prediction of
interactions at large scales. First, there is a relative scarcity of relevant
data in most places globally -- paradoxically, this restricts our ability to
infer interactions for locations where inference is perhaps the least required
(and leaves us unable to make inference in regions without interaction data);
second, accurate predictions often demand accurate predictors, and the lack of
methods that can leverage small amount of *accurate* data is a serious
impediment to our global predictive ability. In most places, our most reliable
biodiversity knowledge is that of a species pool (*i.e.* a set of potentially
interacting species in a given area): through the analysis of databases like
GBIF or IUCN, it is possible to construct a list of species in a region of
interest; but inferring the potential interactions between these species is
difficult.

Following the definition of @Dunne2006NetStr, a metaweb is the ecological
network analogue to the species pool; specifically, it inventories all
*potential* interactions between species for a spatially delimited area (and so
captures the $\gamma$ diversity of interactions). The metaweb is not a
prediction of the network at a specific point within the spatial area it covers:
it will have a different structure, notably by having a larger connectance [see
*e.g.* @Wood2015EffSpa] and complexity [see *e.g.* @Galiana2022EcoNet], from any
of these local networks. These local networks (which capture the $\alpha$
diversity of interactions) are a subset of the metaweb's species and their
interactions, and have been called "metaweb realizations" [@Poisot2015SpeWhy].
Differences between local networks and their metawebs are due to chance, species
abundance and co-occurrence, local environmental conditions, and local
distribution of functional traits, among others. Yet, recent results by
@Saravia2021EcoNet strongly suggest that the local realizations only respond
weakly to local conditions: instead, they reflect constraints inherited by the
structure of their metaweb. This establishes the metaweb structure as the core
goal of predictive network ecology, as it is a required information to
accurately produce downscaled, local predictions.

Because the metaweb represents the joint effect of functional, phylogenetic, and
macroecological processes [@Morales-Castilla2015InfBio], it holds valuable
ecological information. Specifically, it is the "upper bounds" on what the
composition of the local networks, given the local species pool, can be [see
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

# The metaweb is an inherently probabilistic object

Treating interactions probabilistic (as opposed to binary) is a more nuanced and
realistic way to represent interactions. @Dallas2017PreCry suggested that most
links in ecological networks are cryptic, *i.e.* uncommon or hard to observe.
This argument echoes @Jordano2016SamNet: sampling ecological interactions is
difficult because it requires first the joint observation of two species, and
then the observation of their interaction. In addition, it is generally expected
that weak or rare links to be more prevalent in networks than common or rare
links [@Csermely2004StrLin], compared to strong, persistent links; this is
notably the case in food chains, wherein many weaker links are key to the
stability of a system [@Neutel2002StaRea]. In the light of these observations,
we expect to see an over-representation of low-probability interactions under a
model that accurately predicts interaction probabilities. Yet the original
metaweb definition, and indeed most past uses of metawebs, was based on the
presence/absence of interactions. Moving towards *probabilistic* metawebs, by
representing interactions as Bernoulli events [see *e.g.* @Poisot2016StrPro],
offers the opportunity to weigh these rare interactions appropriately. The
inherent plasticity of interactions is important to capture: there have been
documented instances of food webs undergoing rapid collapse/recovery cycles over
short periods of time [*e.g.* @Pedersen2017SigCol]. These considerations
emphasize why metaweb predictions should focus on quantitative (preferentially
probabilistic) predictions; this should constrain the suite of appropriate
models.

Yet it is important to recall that a metaweb is intended as a catalogue of all
potential interactions, which is then filtered [@Morales-Castilla2015InfBio]. In
a sense, that most ecological interactions are elusive can call for a slightly
different approach to sampling: once the common interactions are documented, the
effort required in documenting each rare interaction will increase
exponentially. Recent proposals suggest that machine learning algorithms, in
these situations, can act as data generators [@Hoffmann2019MacLea]: high quality
observational data can generate the core rules underpinning the network
structure, and be supplemented with synthetic data coming from predictive
models, increasing the volume of information available for inference. Indeed,
@Strydom2021RoaPre suggested that knowing the metaweb may render the prediction
of local networks easier, because it fixes an "upper bound" on which
interactions can exist. In this context a probabilistic metaweb represents an
aggregation of informative priors on the interactions, elusive information with
the potential to boost our predictive ability [@Bartomeus2016ComFra].

![Overview of the embedding process. A network (*A*), represented as its
adjacency matrix (*B*), is converted into a lower-dimensional object (*C*) where
nodes, subgraphs, or edges have specific values (see @tbl:methods). For the
purposes of prediction, this low-dimensional object encodes feature vectors for
*e.g.* the nodes. Embedding also allows to visualize the structure in the data
differently (*D*), much like with a principal component
analysis.](figures/conceptual_embedding.png){#fig:embedding}

# Graph embedding offers promises for the inference of potential interactions

Graph embedding (@fig:embedding) is a varied family of machine learning
techniques aiming to transform nodes and edges into a vector space
[@Arsov2019NetEmb], usually of a lower dimension, whilst maximally retaining key
properties of the graph [@Yan2005GraEmb]. Ecological networks are an interesting
candidate for the widespread application of embeddings, as they tend to possess
a shared structural backbone [@BramonMora2018IdeCom], which hints at structural
invariants that can be revealed at lower dimensions. Indeed, food webs are
inherently low-dimensional objects, and can be adequately represented with less
than ten dimensions [@Eklof2013DimEco; @Braga2019SpaAna]. Simulation results by
@Botella2022AppGra suggest that there is no best method to identify
architectural similarities between networks, and that multiple approaches need
to be tested and compared to the network descriptor of interest. This matches
previous, more general results on graph embedding, which suggest that the choice
of embedding algorithm matters for the results [@Goyal2018GraEmb]. In
@tbl:methods, we present a selection of common graph embedding methods,
alongside examples of their use to predict species interactions.

| Method        | Embedding approach                   |            Reference | Application in species interactions     |
| :------------ | :----------------------------------- | -------------------: | --------------------------------------: |
| tSNE          | nodes through statistical divergence |    @Hinton2002StoNei |                 @Cieslak2020TdiSto $^a$ |
| RDPG          | graph through SVD                    |     @Young2007RanDot | @Poisot2021ImpMam; @DallaRiva2016ExpEvo |
| DeepWalk      | graph walk                           |   @Perozzi2014DeeOnl |                       @Wardeh2021PreMam |
| FastEmbed     | graph through PCA/SVD analogue       |  @Ramasamy2015ComSpe |                                         |
| LINE          | nodes through statistical divergence |      @Tang2015LinLar |                                         |
| SDNE          | nodes through auto-encoding          |      @Wang2016StrDee |                                         |
| node2vec      | nodes embedding                      |    @Grover2016NodSca |                                         |
| graph2vec     | sub-graph embedding                  | @Narayanan2017GraLea |                                         |
| DMSE          | joint nodes embedding                |      @Chen2017DeeMul |                    @Chen2017DeeMul $^b$ |
| HARP          | nodes through a meta-strategy        |      @Chen2017HarHie |                                         |
| GraphKKE      | graph embedding                      |    @Melnyk2020GraGra |                  @Melnyk2020GraGra $^a$ |
| Joint methods | multiple graphs                      |      @Wang2021JoiEmb |                                         |

: Overview of some common graph embedding approaches, by time of publication,
alongside examples of their use in the prediction of species interactions. These
methods have not yet been routinely used to predict species interactions; most
examples that we identified were either statistical associations, or analogues
to joint species distribution models. $^a$: statistical interactions; $^b$:
joint-SDM-like approach. {#tbl:methods}

The popularity of graph embedding techniques in machine learning is more than
the search for structural invariants: graphs are discrete objects, and machine
learning techniques tend to handle continuous data better. Bringing a sparse
graph into a continuous, dense vector space [@Xu2020UndGra] opens up a broader
variety of predictive algorithms, notably of the sort that are able to predict
events as probabilities [@Murphy2022ProMac]. Furthermore, the projection of the
graph itself is a representation that can be learned; @Runghen2021ExpNod, for
example, used a neural network to learn the embedding of a network in which not
all interactions were known, based on nodes metadata. This example has many
parallels in ecology (see @fig:prediction), in which node metadata can be given
by phylogeny or functional traits. Rather than directly predicting biological
rules [see *e.g.* @Pichler2020MacLea for an overview], which may be confounded
by the sparse nature of graph data, learning embeddings works in the
low-dimensional space that maximizes information about the network structure.
This approach is further justified by the observation, for example, that the
macro-evolutionary history of a network is adequately represented by some graph
embeddings [RDPG; see @DallaRiva2016ExpEvo].

![From a low-dimensional feature vector (see @fig:embedding), it is possible to
develop predictive approaches. Nodes in an ecological network are species, for
which we can leverage phylogenetic relatedness [*e.g.* @Strydom2021FooWeb] or
functional traits to fill the values of additional species we would like to
project in this space (here, I, J, K, and L) from the embedding of known species
(here, A, B, C, and D). Because embeddings can be projected back to a graph,
this allows us to reconstruct a network with these new species. This approach
constitutes an instance of transfer
learning.](figures/conceptual_prediction.png){#fig:prediction}

# The metaweb embeds ecological hypotheses and practices

The goal of metaweb inference is to provide information about the interactions
between species at a large spatial scale. But as @Herbert1965Dun rightfully
pointed out, "[y]ou can't draw neat lines around planet-wide problems"; any
inference of a metaweb at large scales must contend with several novel, and
interwoven, families of problems.

The first is the taxonomic and spatial limit of the metaweb to embed and
transfer. If the initial metaweb is too narrow in scope, notably from a
taxonomic point of view, the chances of finding another area with enough related
species (through phylogenetic relatedness or similarity of functional traits) to
make a reliable inference decreases; this would likely be indicated by large
confidence intervals during estimation of the values in the low-rank space, or
by non-overlapping trait distributions in the known and unknown species.
Alternatively a metaweb is too large (taxonomically), then the resulting
embeddings would have interactions relative to taxonomic groups that not present
in the new location, resulting in the potential under or over estimation of the
strength of new predicted interactions. The lack of well documented metawebs is
currently preventing the development of more concrete guidelines. The question
of phylogenetic relatedness and dispersal is notably true if the metaweb is
assembled in an area with mostly endemic species (*e.g.* a system that has
undergone recent radiation and might not have an analogous system with which to
draw knowledge from), and as with every predictive algorithm, there is room for
the application of our best ecological judgement.

The second series of problems relate to determining which area should be used to
infer the new metaweb in, as this determines the species pool that must be used.
Metawebs can be constructed by assigning interactions in a list of species
within geographic boundaries. The upside of this approach is that information at
the country level is likely to be required for biodiversity assessments, as
countries set goals at the national level [@Buxton2021KeyInf], and as
quantitative instruments are designed to work at these scales
[@Turak2017UsiEss]; specific strategies are often enacted at smaller scales,
nested within a specific country [@Ray2021BioCri]. But there is no guarantee
that these boundaries are meaningful. In fact, we do not have a satisfying
answer to the question of "where does a food web stop?"; the most promising
solutions involve examining the spatial consistency of network area
relationships [see *e.g.* @Galiana2018SpaSca; @Galiana2019GeoVar;
@Galiana2021SpaSca; @Fortin2021NetEco], which is impossible in the absence of
enough information about the network itself. This suggests that inferred
metawebs should be further downscaled to allow for *a posteriori* analyses.

The final family of problems relates less to ecological concepts and more to the
praxis of ecological research. Operating under the context of national
divisions, in large parts of the world, reflects nothing more than the legacy of
settler colonialism, which not only drive a disparity in available ecological
data, but can entrench said biases with the machine learning models that make
predictions with them. Indeed, the use of ecological data is not an apolitical
act [@Nost2021PolEco], as data infrastructures tend to be designed to answer
questions within national boundaries, and their use often draws upon and
reinforces territorial statecraft. As per @Machen2021ThiAlg, this is
particularly true when the output of "algorithmic thinking" (*e.g.* relying on
machine learning to generate knowledge) can be re-used for governance (*e.g.*
enacting conservation decisions at the national scale). We therefore recognize
that predictive approaches deployed at the continental scale, no matter their
intent, originate in the framework that contributed to the ongoing biodiversity
crisis [@Adam2014EleTre], reinforced environmental injustice
[@Choudry2013SavBio; @Dominguez2020DecCon], and *e.g.* as on Turtle Island,
should be replaced by Indigenous principles of land management
[@Eichhorn2019SteDec; @Nokmaq2021AwaSle]. As we see artificial
intelligence/machine learning being increasingly mobilized to generate knowledge
that is lacking for conservation decisions [*e.g.* @Lamba2019DeeLea;
@MoseboFernandes2020MacLea] and drive policy decisions [@Weiskopf2022IncUpt],
our discussion of these tools need to go beyond the technical, and into the
governance consequences they can have.

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
financial support from NSERC through the Discovery Grants and Discovery
Accelerator Supplement programs. MJF is supported by an NSERC PDF and an RBC
Post-Doctoral Fellowship

# References
