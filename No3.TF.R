#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-
#
# TF_activity_by_cluster_final.R (styled to match example)
# - Times New Roman + Bold + Big fonts
# - TIFF 600dpi LZW via ragg
# - KM: High=red / Low=blue, CI band, legend inside bottom-left,
#       bottom-right annotation "HR=... (per +1 SD) | logrank p=..."

# 0) 路径 -----------------------------------------------------------------------
baseDir <- "/Users/yuezhang/Desktop/DMFF/modelV4.0"
outDir  <- normalizePath(file.path(baseDir, "No3.转录因子活性1"),
                         winslash = "/", mustWork = FALSE)
if (!dir.exists(outDir)) dir.create(outDir, recursive = TRUE)
message("✅ Output directory: ", outDir)

# 1) 依赖 -----------------------------------------------------------------------
if (!requireNamespace("BiocManager", quietly=TRUE)) install.packages("BiocManager")
pkgs <- c(
  "dorothea","viper",                    # Bioc
  "dplyr","purrr","survival","survminer",# CRAN
  "ggplot2","readr","stringr","tibble",
  "ragg","systemfonts","grid","ComplexHeatmap","circlize"
)
for (pkg in pkgs) {
  if (!requireNamespace(pkg, quietly=TRUE)) {
    if (pkg %in% c("dorothea","viper")) BiocManager::install(pkg, ask=FALSE, update=FALSE)
    else install.packages(pkg)
  }
  library(pkg, character.only=TRUE)
}

# 1.1) 字体与保存工具 -----------------------------------------------------------
get_font_family <- function() {
  fam <- "Times New Roman"
  mi  <- tryCatch(systemfonts::match_font(fam), error=function(e) NULL)
  if (is.null(mi) || is.na(mi$path) || !nzchar(mi$path)) fam <- "Times"  # 回退
  fam
}
FONT_FAM <- get_font_family()

# 全局 ggplot 粗体 + 放大
theme_set(
  theme_bw(base_size = 16, base_family = FONT_FAM) +
    theme(
      text = element_text(family = FONT_FAM, face = "bold"),
      plot.title = element_text(face="bold", size=18, hjust=0.5),
      axis.title = element_text(face="bold", size=16),
      axis.text  = element_text(face="bold", size=13),
      legend.title = element_text(face="bold", size=14),
      legend.text  = element_text(face="bold", size=12)
    )
)

# ragg TIFF 保存
save_tiff <- function(plot, filename, width=5, height=4, dpi=600) {
  ggsave(filename = filename, plot = plot,
         device = ragg::agg_tiff, compression = "lzw",
         width = width, height = height, units = "in", dpi = dpi, bg = "white")
}

# 2) fusion_features ------------------------------------------------------------
fusion <- readr::read_csv(file.path(baseDir,"fusion_features.csv"),
                          show_col_types=FALSE)
idcol <- which(sapply(fusion, function(col) all(grepl("^TCGA-", col))))
if (length(idcol)==0) idcol <- 1
rownames(fusion) <- fusion[[idcol]]
fusion[[idcol]] <- NULL
fusion <- as.data.frame(lapply(fusion, as.numeric), row.names = rownames(fusion))
message("✅ Loaded fusion: ", dim(fusion)[1],"×",dim(fusion)[2])

# 3) regulon_list ---------------------------------------------------------------
data("dorothea_hs", package="dorothea")
regs <- dorothea_hs %>% dplyr::filter(confidence %in% c("A","B"))
regulon_list <- regs %>%
  split(.$tf) %>%
  purrr::map(~ list(
    tfmode     = setNames(ifelse(.x$mor=="A", 1, -1), .x$target),
    likelihood = setNames(ifelse(.x$confidence=="A",1,0.8), .x$target)
  ))

# 4) 循环 cluster ---------------------------------------------------------------
files <- list.files(baseDir, "^cluster_\\d+_RNA\\.csv$", full.names=TRUE)
for (fn in files) {
  k <- stringr::str_extract(basename(fn), "(?<=cluster_)\\d+(?=_RNA\\.csv)")
  message("\n=== Cluster ", k, " ===")
  
  # 4.1 表达 + 聚基因
  raw <- readr::read_csv(fn, show_col_types=FALSE)
  gene <- raw[[1]]
  mat  <- raw[,-1] %>% purrr::map_df(as.numeric)
  agg  <- stats::aggregate(mat, by=list(Gene=gene), FUN=function(x) mean(x,na.rm=TRUE))
  rownames(agg) <- agg$Gene; agg$Gene <- NULL
  expr <- as.matrix(agg)
  
  # 4.2 VIPER TF 活性
  tf_act <- viper(expr, regulon_list, minsize=5, verbose=FALSE)
  readr::write_csv(tibble::as_tibble(tf_act, rownames="TF"),
                   file.path(outDir, sprintf("cluster%s_TF_activity.csv",k)))
  message("  ➤ TF activity saved")
  
  # 4.3 临床/生存
  clin <- readr::read_csv(file.path(baseDir, sprintf("cluster_%s_clinical.csv",k)), show_col_types=FALSE)
  surv <- readr::read_csv(file.path(baseDir, sprintf("cluster_%s_survival.csv",k)), show_col_types=FALSE)
  colnames(clin)[1] <- colnames(surv)[1] <- "SampleID"
  pheno <- merge(clin, surv, by="SampleID", sort=FALSE)
  rownames(pheno) <- pheno$SampleID
  samp <- colnames(tf_act)
  pheno <- pheno[samp, ]
  
  # 5) TF vs Fusion 相关 --------------------------------------------------------
  tfs   <- rownames(tf_act)
  feats <- colnames(fusion)
  cor_m <- matrix(NA_real_, length(tfs), length(feats), dimnames=list(tfs,feats))
  p_m   <- cor_m
  for (tf in tfs) for (ft in feats) {
    x <- tf_act[tf,]; y <- fusion[samp, ft]
    ok <- complete.cases(x,y)
    if (sum(ok) < 2) next
    ct <- suppressWarnings(cor.test(x[ok], y[ok], method="spearman", exact=FALSE))
    cor_m[tf,ft] <- unname(ct$estimate); p_m[tf,ft] <- ct$p.value
  }
  padj <- apply(p_m, 2, p.adjust, "fdr")
  readr::write_csv(tibble::as_tibble(cor_m, rownames="TF"),
                   file.path(outDir, sprintf("cluster%s_TF_fusion_corr.csv",k)))
  readr::write_csv(tibble::as_tibble(p_m,   rownames="TF"),
                   file.path(outDir, sprintf("cluster%s_TF_fusion_pval.csv",k)))
  readr::write_csv(tibble::as_tibble(padj,  rownames="TF"),
                   file.path(outDir, sprintf("cluster%s_TF_fusion_padj.csv",k)))
  message("  ➤ TF–fusion matrices saved")
  
  # —— 热图（ComplexHeatmap，统一字体/字号） ——
  sig <- which(padj < 0.05, arr.ind = TRUE)
  if (nrow(sig) > 0) {
    tfsig <- rownames(padj)[unique(sig[,1])]
    ftsig <- colnames(padj)[unique(sig[,2])]
    mat_hm <- cor_m[tfsig, ftsig, drop=FALSE]
    col_fun <- circlize::colorRamp2(c(-1,0,1), c("blue","white","red"))
    fname <- file.path(outDir, sprintf("cluster%s_TF_fusion_heatmap.tiff", k))
    ragg::agg_tiff(fname, width=6, height=6, units="in", res=600, compression="lzw")
    ht <- ComplexHeatmap::Heatmap(
      mat_hm, name="ρ", col=col_fun,
      column_title = paste0("C",k," TF–Fusion (ρ, padj<0.05)"),
      column_title_gp = grid::gpar(fontfamily=FONT_FAM, fontface="bold", fontsize=16),
      row_names_gp    = grid::gpar(fontfamily=FONT_FAM, fontface="bold", fontsize=12),
      column_names_gp = grid::gpar(fontfamily=FONT_FAM, fontface="bold", fontsize=12),
      heatmap_legend_param = list(
        title = "ρ",
        title_gp  = grid::gpar(fontfamily=FONT_FAM, fontface="bold", fontsize=12),
        labels_gp = grid::gpar(fontfamily=FONT_FAM, fontface="bold", fontsize=10)
      )
    )
    ComplexHeatmap::draw(ht)
    dev.off()
    message("  ➤ Heatmap saved (TNR bold)")
  }
  
  # 6) CoxPH（HR per +1 SD） ----------------------------------------------------
  res <- tibble::tibble(TF=tfs, HR=NA_real_, low95=NA_real_, high95=NA_real_, p=NA_real_)
  for (i in seq_along(tfs)) {
    tf0 <- tfs[i]
    df0 <- tibble::tibble(time=pheno$OS.time, event=pheno$OS, act=tf_act[tf0, samp])
    # 用 scale(act) → HR 按 “每 +1 SD”
    fit <- survival::coxph(survival::Surv(time, event) ~ scale(act), df0)
    s   <- summary(fit)
    res[i, 2:5] <- list(
      unname(s$coefficients[,"exp(coef)"]),
      unname(s$conf.int[,"lower .95"]),
      unname(s$conf.int[,"upper .95"]),
      unname(s$coefficients[,"Pr(>|z|)"])
    )
  }
  res <- dplyr::mutate(res, padj = p.adjust(p, "fdr"))
  readr::write_csv(res, file.path(outDir, sprintf("cluster%s_TF_survival_cox.csv", k)))
  message("  ➤ CoxPH saved")
  
  # 7) KM（与示例一致的外观） ---------------------------------------------------
  RED  <- "#d62728"; BLUE <- "#1f77b4"
  top3 <- res %>% dplyr::arrange(padj) %>% dplyr::slice(1:3) %>% dplyr::pull(TF)
  
  for (tf0 in top3) {
    vals <- tf_act[tf0, samp]
    med  <- stats::median(vals, na.rm = TRUE)
    df1  <- tibble::tibble(
      time  = pheno$OS.time,
      event = pheno$OS,
      grp   = dplyr::if_else(vals >= med, "High", "Low")
    )
    nH <- sum(df1$grp=="High"); nL <- sum(df1$grp=="Low")
    
    # HR per +1 SD（与图上角标一致）
    cfit <- survival::coxph(survival::Surv(time,event) ~ scale(vals), data = df1)
    HR   <- unname(exp(coef(cfit)[1]))
    
    # log-rank p
    lr   <- survival::survdiff(survival::Surv(time,event) ~ grp, data=df1)
    p_lr <- stats::pchisq(lr$chisq, df = length(lr$n)-1, lower.tail = FALSE)
    
    # 决定哪个是高风险颜色
    high_risk_grp <- if (HR > 1) "High" else "Low"
    pal <- if (high_risk_grp=="High") c(RED, BLUE) else c(BLUE, RED)
    
    leg_labs <- c(sprintf("High (n=%d) : high risk", nH),
                  sprintf("Low (n=%d)  : low risk",  nL))
    
    fit1 <- survival::survfit(survival::Surv(time,event) ~ grp, data=df1)
    
    g <- survminer::ggsurvplot(
      fit1, data = df1,
      conf.int     = TRUE,
      conf.int.alpha = 0.25,
      pval         = FALSE,         # 我们自己写角标
      risk.table   = FALSE,
      palette      = pal,           # High→红 / Low→蓝（按 HR）
      size         = 1.4,           # 线更粗
      legend.title = "",
      legend.labs  = leg_labs,
      legend       = "right",       # 先设一个位置，稍后移到图内
      ggtheme      = theme_bw(base_size=12, base_family = FONT_FAM) +
        theme(
          text = element_text(family=FONT_FAM, face="bold"),
          plot.title = element_text(face="bold", size=12, hjust=0.5),
          panel.grid.major = element_line(color="grey80"),
          panel.grid.minor = element_blank()
        ),
      title        = paste0("Cluster ", k, " – ", tf0),
      xlab         = "Time", ylab = "Survival Probability"
    )
    
    # 把图例放到图内左下角 & 调大字号
    g$plot <- g$plot +
      theme(
        legend.position = c(0.18, 0.22),
        legend.background = element_rect(fill = "white", color = NA),
        legend.text = element_text(family = FONT_FAM, face="bold", size = 10)
      )
    
    # 右下角角标（Times New Roman 粗体）
    xmax <- max(df1$time, na.rm = TRUE)
    g$plot <- g$plot +
      annotate("text",
               x = xmax*0.98, y = 0.05,
               label = sprintf("HR=%.2f (per +1 SD) | logrank p=%.4g", HR, p_lr),
               hjust = 1, vjust = 0,
               family = FONT_FAM, fontface = "bold", size = 3.8,
               label.size = NA, color = "black",
               fill = "white", alpha = 0.85)
    
    # 保存：TIFF 600dpi LZW（ragg）
    save_tiff(
      plot = g$plot,
      filename = file.path(outDir, sprintf("cluster%s_%s_KM.tiff", k, tf0)),
      width = 5, height = 4, dpi = 600
    )
    message("  ➤ KM saved: cluster", k, "_", tf0, "_KM.tiff")
  }
}

message("\n✅ All done, results are in: ", outDir, "\n")
