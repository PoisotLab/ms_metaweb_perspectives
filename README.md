Having a general solution for inferring *potential* interactions (despite the
unavailability of interaction data) could be the catalyst for significant
breakthroughs in our ability to start thinking about species interaction
networks over large spatial scales. In a recent overview of the field of
ecological network prediction, @Strydom2021RoaPre identified two challenges of
interest to the prediction of interactions at large scales. First, there is a
relative scarcity of relevant data in most places globally -- paradoxically,
this restricts our ability to infer interactions to locations where inference is
perhaps the least required; second, accurate predictions often demand accurate
predictors, and the lack of methods that can leverage small amount of data is a
serious impediment to our predictive ability globally.

Following the definition of @Dunne2006NetStr, a metaweb is a network analogue to
the regional species pool; specifically, it is an inventory of all *potential*
interactions between species from a spatially delimited area (and so captures
the $\gamma$ diversity of interactions). The metaweb is, therefore, *not* a
prediction of the food web at a specific locale within the spatial area it
covers, and will have a different structure [notably by having a larger
connectance; see *e.g.* @Wood2015EffSpa]. These local food webs (which captures
the $\alpha$ diversity of interactions) are a subset of the metaweb's species
and interactions, and have been called "metaweb realizations"
[@Poisot2015SpeWhy]. Differences between local food web and their metaweb are
due to chance, species abundance and co-occurrence, local environmental
conditions, and local distribution of functional traits, among others.

Because the metaweb represents the joint effect of functional, phylogenetic, and
macroecological processes [@Morales-Castilla2015InfBio], it holds valuable
ecological information. Specifically, it is the "upper bounds" on what the
composition of the local networks can be [see *e.g.* @McLeod2021SamAsy]. These
local networks, in turn, can be reconstructed given appropriate knowledge of
local species composition, providing information on structure of food webs at
finer spatial scales. This has been done for example for tree-galler-parasitoid
systems [@Gravel2018BriElt], fish trophic interactions [@Albouy2019MarFis],
tetrapod trophic interactions [@OConnor2020UnvFoo], and crop-pest networks
[@Grunig2020CroFor]. Whereas the original metaweb definition, and indeed most
past uses of metawebs, was based on the presence/absence of interactions, we
focus on *probabilistic* metawebs where interactions are represented as the
chance of success of a Bernoulli trial [see *e.g.* @Poisot2016StrPro];
therefore, not only does our method recommend interactions that may exist, it
gives each interaction a score, allowing us to properly weigh them.

# The metaweb is an inherently probabilistic object

Yet, owing to the inherent plasticity of interactions, there have been
documented instances of food webs undergoing rapid collapse/recovery cycles over
short periods of time [@Pedersen2017SigCol]. The embedding of a network, in a
sense, embeds its macro-evolutionary history, especially as RDPG captures
ecological signal [@DallaRiva2016ExpEvo]; at this point, it is important to
recall that a metaweb is intended as a catalogue of all potential interactions,
which should then be filtered [@Morales-Castilla2015InfBio]. In practice (and in
this instance) the reconstructed metaweb will predict interactions that are
plausible based on the species' evolutionary history, however some interactions
would/would not be realized due to human impact.

@Dallas2017PreCry suggested that most links in ecological networks may be
cryptic, *i.e.* uncommon or otherwise hard to observe. This argument essentially
echoes @Jordano2016SamNet: the sampling of ecological interactions is difficult
because it requires first the joint observation of two species, and then the
observation of their interaction. In addition, it is generally expected that
weak or rare links would be more common in networks [@Csermely2004StrLin],
compared to strong, persistent links; this is notably the case in food chains,
wherein many weaker links are key to the stability of a system
[@Neutel2002StaRea]. In the light of these observations, the results in
@fig:inflation are not particularly surprising: we expect to see a surge in
these low-probability interactions under a model that has a good predictive
accuracy. Because the predictions we generate are by design probabilistic, then
one can weigh these rare links appropriately. In a sense, that most ecological
interactions are elusive can call for a slightly different approach to sampling:
once the common interactions are documented, the effort required in documenting
each rare interaction may increase exponentially. Recent proposals suggest that
machine learning algorithms, in these situations, can act as data generators
[@Hoffmann2019MacLea]: in this perspective, high quality observational data can
be supplemented with synthetic data coming from predictive models, which
increases the volume of information available for inference. Indeed,
@Strydom2021RoaPre suggested that knowing the metaweb may render the prediction
of local networks easier, because it fixes an "upper bound" on which
interactions can exist; indeed, with a probabilistic metaweb, we can consider
that the metaweb represents an aggregation of informative priors on the
interactions.

# Graph embedding offers promises for the inference of potential interactions

Graph embedding is a varied family of machine learning techniques aiming to
transform nodes and edges into a vector space, usually of a lower dimension,
whilst maximally retaining key properties of the graph. Ecological networks are
an interesting candidate for the widespread application of embeddings, as they
tend to posess a shared sstructural backbone [@Mora2018IdeCom], which hints at
structural invariants that can be revealed a lower dimensions. Indeed, previous
work by @Eklof2013DimEco suggests that food webs are inherently low-dimensional
objects, and can be adequately represented with less than ten dimensions.

The popularity of graph embedding techniques in machine learning is rather
intuitive: while graphs are discrete objects, machine learning techniques tend
to handle continuous data better. Therefore, bringing a discrete graph into a
continuous vector space opens up a broader variety of predictive algorithms.


| Method    | Embedded object             | Reference            |
| --------- | --------------------------- | -------------------- |
| DeepWalk  | graph walk                  | @Perozzi2014DeeOnl   |
| node2vec  | node embedding              | @Grover2016NodSca    |
| graph2vec | sub-graph embedding         | @Narayanan2017GraLea |
| SDNE      | nodes through auto-encoding | @Wang2016StrDee      |

# The metaweb embeds hypotheses about which spatial boundaries are meaningful

As @Herbert1965Dun rightfully pointed out, "[y]ou can't draw neat lines around
planet-wide problems"; in this regard, our approach (and indeed, any inference
of a metaweb at large scales) must contend with several interesting and
interwoven families of problems. The first is the limit of the metaweb to embed
and transfer. If the initial metaweb is too narrow in scope, notably from a
taxonomic point of view, the chances of finding another area with enough related
species to make a reliable inference decreases; this would likely be indicated
by large confidence intervals during ancestral character estimation, but the
lack of well documented metawebs is currently preventing the development of more
concrete guidelines. The question of phylogenetic relatedness and dispersal is
notably true if the metaweb is assembled in an area with mostly endemic species,
and as with every predictive algorithm, there is room for the application of our
best ecological judgement. Conversely, the metaweb should be reliably filled,
which assumes that the $S^2$ interactions in a pool of $S$ species have been
examined, either through literature surveys or expert elicitation. Supp. Mat. 1
provides some guidance as to the type of sampling effort that should be
prioritized. While RDPG was able to maintain very high predictive power when
interactions were missing, the addition of false positive interactions was
immediately detected; this suggests that it may be appropriate to err on the
side of "too many" interactions when constructing the initial metaweb to be
transferred. The second series of problems are related to determining which area
should be used to infer the new metaweb in, as this determines the species pool
that must be used.

In our application, we focused on the mammals of Canada. The
upside of this approach is that information at the country level is likely to be
required by policy makers and stakeholders for their biodiversity assessment, as
each country tends to set goals at the national level [@Buxton2021KeyInf] for
which quantitative instruments are designed [@Turak2017UsiEss], with specific
strategies often enacted at smaller scales [@Ray2021BioCri]. And yet, we do not
really have a satisfying answer to the question of "where does a food web
stop?"; the current most satisfying solutions involve examining the spatial
consistency of network area relationships [see *e.g.* @Galiana2018SpaSca;
@Galiana2019GeoVar; @Galiana2021SpaSca; @Fortin2021NetEco], which is of course
impossible in the absence of enough information about the network itself. This
suggests that an *a posteriori* refinement of the results may be required, based
on a downscaling of the metaweb. The final family of problems relates less to
the availability of data or quantitative tools, and more to the praxis of
spatial ecology. Operating under the context of national divisions, in large
parts of the world, reflects nothing more than the legacy of settler
colonialism. Indeed, the use of ecological data is not an apolitical act
[@Nost2021PolEco], as data infrastructures tend to be designed to answer
questions within national boundaries, and their use both draws upon and
reinforces territorial statecraft; as per @Machen2021ThiAlg, this is
particularly true when the output of "algorithmic thinking" (*e.g.* relying on
machine learning to generate knowledge) can be re-used for governance (*e.g.*
enacting conservation decisions at the national scale). We therefore recognize
that methods such as we propose operate under the framework that contributed to
the ongoing biodiversity crisis [@Adam2014EleTre], reinforced environmental
injustice [@Choudry2013SavBio; @Dominguez2020DecCon], and on Turtle Island
especially, should be replaced by Indigenous principles of land management
[@Eichhorn2019SteDec; @Nokmaq2021AwaSle]. As we see AI/ML being increasingly
mobilized to generate knowledge that is lacking for conservation decisions
[*e.g.* @Lamba2019DeeLea; @MoseboFernandes2020MacLea], our discussion of these
tools need to go beyond the technical, and into the governance consequences they
can have.

**Acknowledgements:** We acknowledge that this study was conducted on land
within the traditional unceded territory of the Saint Lawrence Iroquoian,
Anishinabewaki, Mohawk, Huron-Wendat, and Omàmiwininiwak nations. TP, TS, DC,
and LP received funding from the Canadian Institue for Ecology & Evolution. FB
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
Accelerator Supplement programs.

# References
