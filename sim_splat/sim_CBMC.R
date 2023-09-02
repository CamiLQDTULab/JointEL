## This script does not work

rm(list = ls())
library(Seurat)
library(ggplot2)
library(patchwork)

cbmc.rna <- as.sparse(read.csv(file = "/data/hoan/multiomics/sim_splat/cbmc_data/GSE100866_CBMC_8K_13AB_10X-RNA_umi.csv.gz", sep = ",", 
                               header = TRUE, row.names = 1))

# To make life a bit easier going forward, we're going to discard all but the top 100 most
# highly expressed mouse genes, and remove the 'HUMAN_' from the CITE-seq prefix
cbmc.rna <- CollapseSpeciesExpressionMatrix(cbmc.rna)

# Load in the ADT UMI matrix
cbmc.adt <- as.sparse(read.csv(file = "/data/hoan/multiomics/sim_splat/cbmc_data/GSE100866_CBMC_8K_13AB_10X-ADT_umi.csv.gz", sep = ",", 
                               header = TRUE, row.names = 1))

cbmc.adt <- cbmc.adt[setdiff(rownames(x = cbmc.adt), c("CCR5", "CCR7", "CD10")), ]

cbmc <- CreateSeuratObject(counts = cbmc.rna)

# standard log-normalization
cbmc <- NormalizeData(cbmc)

# choose ~1k variable features
cbmc <- FindVariableFeatures(cbmc)

# standard scaling (no regression)
cbmc <- ScaleData(cbmc)

# Run PCA, select 13 PCs for tSNE visualization and graph-based clustering
cbmc <- RunPCA(cbmc, verbose = FALSE)
cbmc <- FindNeighbors(cbmc, dims = 1:25)
cbmc <- FindClusters(cbmc, resolution = 0.8)
# cbmc <- RunTSNE(cbmc, dims = 1:25, method = "FIt-SNE")

new.cluster.ids <- c("Memory CD4 T", "CD14+ Mono", "Naive CD4 T", "NK", "CD14+ Mono", "Mouse", "B", 
                     "CD8 T", "CD16+ Mono", "T/Mono doublets", "NK", "CD34+", "Multiplets", "Mouse", "Eryth", "Mk", 
                     "Mouse", "DC", "pDCs")
names(new.cluster.ids) <- levels(cbmc)
cbmc <- RenameIdents(cbmc, new.cluster.ids)


### ADT
cbmc[["ADT"]] <- CreateAssayObject(counts = cbmc.adt)

# Now we can repeat the preprocessing (normalization and scaling) steps that we typically run
# with RNA, but modifying the 'assay' argument.  For CITE-seq data, we do not recommend typical
# LogNormalization. Instead, we use a centered log-ratio (CLR) normalization, computed
# independently for each feature.  This is a slightly improved procedure from the original
# publication, and we will release more advanced versions of CITE-seq normalizations soon.
cbmc <- NormalizeData(cbmc, assay = "ADT", normalization.method = "CLR")
cbmc <- ScaleData(cbmc, assay = "ADT")

# You can see that our unknown cells co-express both myeloid and lymphoid markers (true at the
# RNA level as well). They are likely cell clumps (multiplets) that should be discarded. We'll
# remove the mouse cells now as well
cbmc <- subset(cbmc, idents = c("Multiplets", "Mouse"), invert = TRUE)

DefaultAssay(cbmc) <- "ADT"
cbmc <- RunPCA(cbmc, features = rownames(cbmc), reduction.name = "pca_adt", reduction.key = "pca_adt_", 
               verbose = FALSE)

adt.data <- GetAssayData(cbmc, slot = "data")
adt.dist <- dist(t(adt.data))

# Before we recluster the data on ADT levels, we'll stash the RNA cluster IDs for later
cbmc[["rnaClusterID"]] <- Idents(cbmc)

# Now, we rerun tSNE using our distance matrix defined only on ADT (protein) levels.
# cbmc[["tsne_adt"]] <- RunTSNE(adt.dist, assay = "ADT", reduction.key = "adtTSNE_")
cbmc[["adt_snn"]] <- FindNeighbors(adt.dist)$snn
cbmc <- FindClusters(cbmc, resolution = 0.2, graph.name = "adt_snn")

new.cluster.ids <- c("CD4 T", "CD14+ Mono", "NK", "B", "CD8 T", "NK", "CD34+", "T/Mono doublets", 
                     "CD16+ Mono", "pDCs", "B")
names(new.cluster.ids) <- levels(cbmc)
cbmc <- RenameIdents(cbmc, new.cluster.ids)

adt_labels <- cbmc@active.ident

subset_name <- "B"
cbmc_cd8 <- cbmc[,adt_labels==subset_name]
# cbmc_cd8 <- cbmc


cd8_rna <- as.matrix(cbmc_cd8@assays[["RNA"]]@counts)
cd8_adt <- as.matrix(adt.data[,adt_labels==subset_name])

N <- dim(cd8_adt)[1]
M <- dim(cd8_adt)[2]
cd8_adt <- cd8_adt + matrix( rnorm(N*M,mean=0,sd=3), N, M)
cd8_adt <- floor(cd8_adt - min(cd8_adt))
cd8_adt[cd8_adt<mean(cd8_adt)] <- 0


### Simulate data using Splatter.
## ADT
params <- splatEstimate(round(10*cd8_adt))

params <- setParam(params, "batchCells", 1000)

n_groups <- 5
# de.prob - DE probability This parameter controls the probability that a gene will be selected to be differentially expressed.
# The higher the more separation between groups
deprob <- rep(0.2, n_groups)
# de.facLoc Differential expression factors are produced from a log-normal distribution in a similar way to batch effect factors and expression outlier factors. Changing these parameters can result in more or less extreme differences between groups.
# The higher the more separation between groups
de.facLoc <- 1.1

group.prob <- c(20, 20, 20, 20, 20)/100

sim <- splatSimulateGroups(params, group.prob = group.prob, de.prob = deprob, de.facLoc = de.facLoc,
                           verbose = FALSE)


## Analysis
sim <- logNormCounts(sim)
dat <- logcounts(sim)
# PCA data
pca <- prcomp(t(dat), center = TRUE, scale. = FALSE)
N <- 10
Data2 <- pca$x[, 1:N, drop = FALSE]
# dataname <- "simLogcount"
# write.table(Data2,file = paste0("/data/hoan/spectral_clustering_matlab/data/", dataname, "_pca.csv"),
#             sep = ",", row.names = F, col.names = F)
# write.table(labels,file = paste0("/data/hoan/spectral_clustering_matlab/data/", dataname, "_pca_labels.csv"),
#             sep = ",", row.names = F, col.names = F)
# tSNE
tsne_out2 <- Rtsne(Data2, pca=F)
# plot(tsne_out2$Y, asp=1, main = "logCount preprocessing")
plot(tsne_out2$Y,col=sim$Group, asp=1, main = "logCount preprocessing")



