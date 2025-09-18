====================================================================
DEATH: Differential Expression Analysis Tool for High-throughput NGS
====================================================================

Basic Information
=================

**Repository:** https://github.com/lnfu/death

Problem to Solve
================

Gene expression analysis is a fundamental problem in genomics and
bioinformatics. While microarray technology dominated the early 2000s, RNA
sequencing (RNA-seq) has become the standard approach for transcriptome
profiling. RNA-seq involves converting sample RNA to cDNA, fragmenting it,
sequencing using Next Generation Sequencing (NGS) technology, and aligning
reads to a reference genome to quantify gene expression through read counts.

A core task in RNA-seq analysis is differential gene expression (DGE) analysis,
which determines whether transcript or exon read counts differ significantly
between experimental conditions. For example, to identify genes with
significantly different expression between normal and cancer cells, researchers
collect 3-6 samples per condition, perform RNA-seq, and apply
statistical methods to detect differential expression.

The challenge lies in the limited sample sizes (typically 2-6 biological
replicates per condition) and inherent technical variability of RNA-seq
experiments. Current methods like edgeR and DESeq2 address these challenges
using more complicated statistical models, primarily based on the negative
binomial distribution to handle overdispersion in count data.

This project implements a streamlined DGE analysis tool following DESeq2's
methodology, providing both C++ performance and Python accessibility.

**Statistical Framework:**

DESeq2 models read counts using a negative binomial distribution:

.. code-block::

   K_ij ~ NB(μ_ij, α_i)

where μ_ij is the expected count for gene i in sample j, and α_i is the gene specific dispersion parameter.

The generalized linear model (GLM) framework employs:

- Linear predictor: log(μ_ij) = X_j β_i
- Iteratively Reweighted Least Squares (IRLS) for parameter estimation
- Wald test for significance testing
- Benjamini-Hochberg correction for multiple testing

Prospective Users
=================

- Bioinformaticians analyzing RNA-seq datasets
- Researchers studying gene regulation and expression
- Clinical researchers investigating disease-related expression changes
- Students learning computational genomics methods

System Architecture
===================

**Workflow:**

1. **Input:** Accept BAM files or count matrices (~100K genes x 3-6 samples)
2. **Normalization:** Calculate size factors to adjust for sequencing depth
3. **Dispersion Estimation:** Estimate gene-specific dispersion parameters
4. **GLM Fitting:** Use IRLS to estimate regression coefficients
5. **Statistical Testing:** Apply Wald test for differential expression
6. **Multiple Testing Correction:** Benjamini-Hochberg FDR correction
7. **Output:** Generate results table with statistics

API Description
===============

**C++ Core API:**

.. code-block:: cpp

   class Analyzer {
   public:
       Analyzer(const CountMatrix& counts, 
                    const ExperimentalDesign& design);
       
       void estimateDispersion();
       void fitGLM();
       DGEResults runAnalysis();
       
   private:
       CountMatrix counts_;
       GLMFitter glm_fitter_;
       DispersionEstimator dispersion_estimator_;
   };

**Python Interface:**

.. code-block:: python

   import death
   
   # Load count data and experimental design
   analyzer = death.Analyzer(
       count_file="counts.tsv",
       design_formula="~ condition"
   )
   
   # Run differential expression analysis
   results = analyzer.run_analysis()
   
   # Export results
   results.to_csv("dge_results.csv")
   
   # Visualization
   death.plot_ma(results)
   death.plot_volcano(results)

**Example Output:**

.. code-block::

   gene_id    baseMean  log2FC   lfcSE    stat    pvalue      padj
   GENE1      2847.1    2.341    0.187   12.53   <2e-16    <2e-16  ***
   GENE2      1923.4    1.876    0.203    9.24   2.4e-20   1.2e-18  ***
   TP53        365.8   -0.678    0.125   -5.42   5.9e-08   2.9e-07  ***
   GAPDH      1807.2    0.387    0.089    4.35   1.4e-05   3.4e-05  ***

Engineering Infrastructure
==========================

1. Build system: CMake
2. Version control: Git + GitHub
3. Testing framework: Google Test (gtest) for C++, pytest for Python
4. Documentation: README.md

Schedule
========

Development timeline (10 weeks):

* Week 01 (09/22): Repository setup, literature review, finalize proposal
* Week 02 (09/29): Create Python bindings structure, implement data input and 
  preprocessing modules
* Week 03 (10/06): Develop size factor calculation, unit test with synthetic
  count matrix
* Week 04 (10/13): Implement dispersion estimation algorithms, test mean
  variance fitting convergence
* Week 05 (10/20): Build GLM fitting module with IRLS implementation, test
  coefficient convergence
* Week 06 (10/27): Add statistical testing (Wald test) and correction, unit
  test p-value calculations
* Week 07 (11/03): Complete API interface, comprehensive end-to-end pipeline
  testing
* Week 08 (11/10): Validation testing against DESeq2 output comparison,
  performance benchmarking, documentation
* Week 09 (11/17): Final debugging
* Week 10 (11/24): Project presentation preparation

References
==========

1. Robinson, M.D., McCarthy, D.J., & Smyth, G.K. (2010). edgeR: a Bioconductor 
   package for differential expression analysis of digital gene expression 
   data. *Bioinformatics*, 26(1), 139-140.
2. Anders, S. & Huber, W. (2010). Differential expression analysis for sequence 
   count data. *Genome Biology*, 11(10), R106.
3. Love, M.I., Huber, W., & Anders, S. (2014). Moderated estimation of fold 
   change and dispersion for RNA-seq data with DESeq2. *Genome Biology*, 
   15(12), 550.
