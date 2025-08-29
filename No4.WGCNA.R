#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-
#
# No4.WGCNA_full_pipeline.R
#
# 一体化 WGCNA 全流程：合并表达 & 表型 → 构建网络 → 模块检测 →
# 模块-性状关联 → hub 基因筛选 → 富集 & PPI → 各类可视化

# 0. 设置工作目录 & 输出目录
baseDir <- "D:/BRCA/BRCA/DMFF/modelV4.0"
outDir  <- file.path(baseDir, "No4.WGCNA")
dir.create(outDir, showWarnings=FALSE, recursive=TRUE)
setwd(baseDir)
cat("Working directory:", getwd(), "\n")
cat("Results will be saved under:", outDir, "\n\n")

options(stringsAsFactors = FALSE, warn = -1)

# 1. 载入依赖
if (!requireNamespace("WGCNA", quietly=TRUE)) install.packages("WGCNA")
if (!requireNamespace("BiocManager", quietly=TRUE)) install.packages("BiocManager")
BiocManager::install(
  c("clusterProfiler","org.Hs.eg.db","STRINGdb"),
  ask=FALSE, update=FALSE
)
library(WGCNA); enableWGCNAThreads()
library(clusterProfiler); library(org.Hs.eg.db); library(STRINGdb)
library(pheatmap); library(ggplot2)

# 2. 合并表达矩阵（按簇拆分的 CSV）
rnaFiles <- list.files(
  path       = baseDir,
  pattern    = "^cluster_[0-9]+_RNA\\.csv$",
  full.names = TRUE
)
if (length(rnaFiles)==0) stop("ERROR: 没有找到任何 cluster_*_RNA.csv 文件。")

exprList <- lapply(rnaFiles, function(f){
  raw <- read.csv(f, check.names=FALSE, stringsAsFactors=FALSE)
  colnames(raw)[1] <- "Gene"
  agg <- aggregate(. ~ Gene, data=raw,
                   FUN=function(x) mean(as.numeric(x), na.rm=TRUE))
  rownames(agg) <- agg$Gene; agg$Gene <- NULL
  return(agg)
})
expr <- Reduce(function(x,y){
  m <- merge(x,y,by="row.names")
  rownames(m) <- m$Row.names; m$Row.names <- NULL
  return(m)
}, exprList)
cat(">> merged expr dims:", dim(expr), "\n")
if (ncol(expr)==0) stop("ERROR: 合并后 expr 没有列。")

# 3. WGCNA 输入
datExpr <- t(expr)
cat(">> datExpr dims (samples×genes):", dim(datExpr), "\n\n")

# 4. 网络构建 & 模块检测
powers    <- c(1:10, seq(12,20,2))
sft       <- pickSoftThreshold(datExpr, powerVector=powers, verbose=0)
softPower <- if (!is.na(sft$powerEstimate)) sft$powerEstimate else 6
cat("Chosen softPower =", softPower, "\n\n")

adj        <- adjacency(datExpr, power=softPower)
TOM        <- TOMsimilarity(adj)
dissTOM    <- 1 - TOM
geneTree   <- hclust(as.dist(dissTOM), method="average")
dynMods    <- cutreeDynamic(
  dendro=geneTree, distM=dissTOM,
  deepSplit=2, pamRespectsDendro=FALSE,
  minClusterSize=30
)
moduleColors <- labels2colors(dynMods)
MEList     <- moduleEigengenes(datExpr, colors=moduleColors)
MEs        <- orderMEs(MEList$eigengenes)

write.csv(moduleColors,
          file.path(outDir, "moduleColors.csv"),
          quote=FALSE)
write.csv(MEs,
          file.path(outDir, "moduleEigengenes.csv"),
          quote=FALSE)

# 5. 合并表型 & 生存
clinFiles <- list.files(pattern="^cluster_[0-9]+_clinical\\.csv$")
phenoList <- lapply(clinFiles, function(f){
  cid  <- sub("^cluster_([0-9]+)_clinical\\.csv$","\\1",f)
  clin <- read.csv(f, check.names=FALSE, stringsAsFactors=FALSE)
  surv <- read.csv(sub("clinical","survival",f),
                   check.names=FALSE, stringsAsFactors=FALSE)
  colnames(clin)[1] <- "Sample"; colnames(surv)[1] <- "Sample"
  df <- merge(clin, surv, by="Sample", sort=FALSE)
  df$Cluster <- paste0("C",cid)
  rownames(df) <- df$Sample; df$Sample <- NULL
  return(df)
})
pheno <- do.call(rbind, phenoList)
common <- intersect(rownames(pheno), colnames(expr))
expr   <- expr[, common]
pheno  <- pheno[common, ,drop=FALSE]
write.csv(pheno,
          file.path(outDir, "trait_data.csv"),
          quote=FALSE)
cat(">> merged pheno dims:", dim(pheno), "\n\n")

# 6. 模块–性状关联
traitDF        <- data.frame(
  OS_time = as.numeric(pheno$OS.time),
  Cluster = as.numeric(as.factor(pheno$Cluster))
)
moduleTraitCor <- cor(MEs, traitDF, use="p")
moduleTraitP   <- corPvalueStudent(moduleTraitCor, nrow(datExpr))
write.csv(moduleTraitCor,
          file.path(outDir,"moduleTraitCor.csv"), quote=FALSE)
write.csv(moduleTraitP,
          file.path(outDir,"moduleTraitP.csv"),   quote=FALSE)

# 7. 挑显著模块；若无 p<0.05，则选 |cor(Cluster)| 排名前3
sigMods   <- rownames(moduleTraitP)[apply(moduleTraitP<0.05,1,any)]
top3_mods <- names(sort(abs(moduleTraitCor[,"Cluster"]),
                        decreasing=TRUE))[1:3]
if (length(sigMods)==0) {
  sigMods <- top3_mods
  cat("⚠️ 无模块满足 p<0.05，改用 |cor(Cluster)| 排名前3：",
      paste(sigMods, collapse=", "), "\n")
} else {
  cat("显著模块 (p<0.05)：", paste(sigMods, collapse=", "), "\n")
}

# 8. hub 基因 & 富集 & PPI（跳过空模块）
hubList   <- list()
string_db <- STRINGdb$new(version="11", species=9606, score_threshold=400)
for (mod in sigMods) {
  modID <- sub("^ME","",mod)
  genes <- names(moduleColors)[moduleColors==modID]
  if (length(genes)==0) {
    cat("Warning: 模块",mod,"无基因，跳过 hub 筛选\n")
    next
  }
  MEvec <- MEs[[mod]]
  kME   <- cor(datExpr[,genes], MEvec, use="p")
  GS    <- abs(cor(datExpr[,genes], traitDF$OS_time, use="p"))
  hubs  <- genes[kME>0.8 & GS>0.2]
  hubList[[mod]] <- hubs
  writeLines(hubs,
             file.path(outDir, paste0("hubs_",mod,".txt")))
  # GO 富集
  if (length(hubs)>0) {
    eg  <- bitr(hubs, fromType="SYMBOL", toType="ENTREZID",
                OrgDb="org.Hs.eg.db")
    ego <- enrichGO(eg$ENTREZID, OrgDb="org.Hs.eg.db",
                    ont="BP", pAdjustMethod="BH", pvalueCutoff=0.05)
    write.csv(as.data.frame(ego),
              file.path(outDir,paste0("GO_",mod,".csv")),
              row.names=FALSE)
    # PPI
    df_map <- string_db$map(data.frame(symbol=hubs),
                            "symbol", removeUnmappedRows=TRUE)
    ints    <- string_db$get_interactions(df_map$STRING_id)
    write.csv(ints,
              file.path(outDir,paste0("PPI_",mod,".csv")),
              row.names=FALSE)
  }
  cat("Module",mod,"→ hubs:",length(hubs),"\n")
}

# 9. 各类可视化
## 9.1 基因树 + 模块色条
pdf(file.path(outDir,"1_geneDendro_modules.pdf"), width=10, height=6)
plotDendroAndColors(geneTree, moduleColors, "Module",
                    dendroLabels=FALSE, hang=0.03,
                    addGuide=TRUE, guideHang=0.05,
                    main="Gene dendrogram & module colors")
dev.off()

## 9.2 模块–性状热图（带 cor & p）
textMat <- matrix(paste0(signif(moduleTraitCor,2), "\n(",
                         signif(moduleTraitP,1),")"),
                  nrow=nrow(moduleTraitCor),
                  ncol=ncol(moduleTraitCor),
                  dimnames=dimnames(moduleTraitCor))
pdf(file.path(outDir,"2_module_trait_heatmap.pdf"),
    width=7, height=6)
labeledHeatmap(Matrix=moduleTraitCor,
               xLabels=colnames(traitDF),
               yLabels=rownames(moduleTraitCor),
               textMatrix=textMat,
               colors=blueWhiteRed(50),
               main="Module–trait\n(cor & p-value)")
dev.off()

## 9.3 GS vs MM 散点（空模块跳过）
for (mod in sigMods) {
  modID <- sub("^ME","",mod)
  genes <- names(moduleColors)[moduleColors==modID]
  if (length(genes)<2) {
    cat("Module",mod,"基因 <2，跳过 GS vs MM\n")
    next
  }
  MEcol <- match(mod, colnames(MEs))
  GS    <- abs(cor(datExpr[,genes], traitDF$OS_time, use="p"))
  MM    <- abs(cor(datExpr[,genes], MEs[,MEcol], use="p"))
  pdf(file.path(outDir,sprintf("3_GS_vs_MM_%s.pdf",modID)),
      width=6, height=6)
  plot(GS, MM,
       xlab="Gene significance (OS_time)",
       ylab=paste("Module membership in",modID),
       main=paste(modID,"module: GS vs MM"),
       pch=20, col=modID)
  abline(lm(MM~GS), col="gray")
  dev.off()
}

## 9.4 TOM 热图（空模块跳过）
for (mod in sigMods) {
  modID <- sub("^ME","",mod)
  genes <- names(moduleColors)[moduleColors==modID]
  if (length(genes)<2) {
    cat("Module",mod,"基因 <2，跳过 TOM heatmap\n")
    next
  }
  MEvec <- MEs[[mod]]
  kME   <- cor(datExpr[,genes], MEvec, use="p")
  topG  <- names(sort(kME,decreasing=TRUE))[1:min(200,length(kME))]
  pdf(file.path(outDir,sprintf("4_TOMheatmap_%s.pdf",modID)),
      width=7, height=6)
  TOMplot(dissTOM[topG,topG], geneTree, moduleColors[topG],
          main=paste("TOM heatmap for",modID,"module"))
  dev.off()
}

## 9.5 注解热图
pdf(file.path(outDir,"5_module_trait_pheatmap.pdf"),
    width=8, height=6)
pheatmap(moduleTraitCor,
         color=colorRampPalette(c("blue","white","red"))(50),
         display_numbers=textMat,
         number_color="black",
         fontsize_number=6,
         fontsize_row=8,
         fontsize_col=8,
         angle_col=45,
         border_color=NA,
         main="Module–trait\n(correlations & p-values)")
dev.off()

cat("\n✅ 全流程完成，所有结果已保存到：", outDir, "\n")
