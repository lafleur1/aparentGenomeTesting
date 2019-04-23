# APAReNt - APA Regression Net
This repository contains the code for training and running APAReNt, a deep neural network that can predict human 3' UTR Alternative Polyadenylation (APA), annotate genetic variants based on the impact of APA regulation, and engineer new polyadenylation signals according to target isoform abundances or cleavage profiles.

APAReNt was trained on >3.5 million randomized 3' UTR polyadenylation signals expressed on mini gene reporters.

Forward-engineering of new polyadenylation signals is done using the included SeqProp (Stochastic Sequence Backpropagation) software, which implements a gradient-based input maximization algorithm and uses APAReNt as the predictor.


# Designed MPRA Analysis Notebooks:
[Notebook 0a: Basic MPRA Library Statistics](analysis/analyze_aparent_designed_mpra_stats_legacy.ipynb)
[Notebook 0b: MPRA LoFi vs. HiFi Replicates](analysis/analyze_aparent_designed_mpra_lofi_vs_hifi_legacy.ipynb)

[Notebook 1a: SeqProp Target Isoforms (Summary)](analysis/analyze_aparent_designed_mpra_seqprop_iso_summary_legacy.ipynb)
[Notebook 1b: SeqProp Target Isoforms (Detailed)](analysis/analyze_aparent_designed_mpra_seqprop_iso_detailed_legacy.ipynb)

[Notebook 2a: SeqProp Target Cut (Summary)](analysis/analyze_aparent_designed_mpra_seqprop_cut_summary_legacy.ipynb)
[Notebook 2b: SeqProp Target Cut (Detailed)](analysis/analyze_aparent_designed_mpra_seqprop_cut_detailed_legacy.ipynb)

[Notebook 3: Human Wildtype APA Prediction](analysis/analyze_aparent_designed_mpra_wildtype_human_apa_legacy.ipynb)

[Notebook 4a: Human Variant Analysis (Summary)](analysis/analyze_aparent_designed_mpra_variant_summary_legacy.ipynb)
[Notebook 4b: Disease-Implicated Variants/UTRs (Detailed)](analysis/analyze_aparent_designed_mpra_pathogenic_utrs_legacy.ipynb)
[Notebook 4c: Cleavage-Altering Varaints (Detailed)](analysis/analyze_aparent_designed_mpra_complex_cut_variants_legacy.ipynb)

[Notebook 5a: Complex Functional Variants (Summary)](analysis/analyze_aparent_designed_mpra_rare_functional_variants_summary_legacy.ipynb)
[Notebook 5b: Complex Functional Variants (Canonical CSE)](analysis/analyze_aparent_designed_mpra_rare_functional_variants_detailed_canonical_cse_legacy.ipynb)
[Notebook 5c: Complex Functional Variants (Cryptic CSE)](analysis/analyze_aparent_designed_mpra_rare_functional_variants_detailed_cryptic_cse_legacy.ipynb)
[Notebook 5d: Complex Functional Variants (CFIm25)](analysis/analyze_aparent_designed_mpra_rare_functional_variants_detailed_tgta_legacy.ipynb)
[Notebook 5e: Complex Functional Variants (CstF)](analysis/analyze_aparent_designed_mpra_rare_functional_variants_detailed_tgtct_legacy.ipynb)
[Notebook 5f: Complex Functional Variants (Folding)](analysis/analyze_aparent_designed_mpra_rare_functional_variants_detailed_folding_legacy.ipynb)

[Notebook Bonus: TGTA Motif Saturation Mutagenesis](analysis/analyze_aparent_designed_mpra_tgta_mutation_maps_legacy.ipynb)
