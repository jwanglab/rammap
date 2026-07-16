# Alignment validation and performance

rammap 1.0.0 vs minimap2 v2.31

## System

| | |
|---|---|
| **CPU** | Intel Xeon Gold 6140 (2.3 GHz, AVX512) |
| **RAM** | 32 GB allocated |
| **Threads** | 8 |

## Test Data

Publicly available ONT, PacBio (HiFi and Onso), Illumina (PCR-free and Hi-C),
BGISEQ, and Element AVITI DNA sequencing data and ONT RNA-sequencing data were
evaluated across various presets and alignment modes. Datasets are drawn from
the Genome in a Bottle (GIAB) reference releases and the Human Pangenome
Reference Consortium (HPRC) production data releases, supplemented by a public
direct-RNA reference run for ONT's RNA004 chemistry.

| Dataset | Source | Access |
|---------|--------|--------|
| ONT R10 (sup) | HG002; GIAB | `s3://ont-open-data/giab_2025.01/basecalling/sup/HG002/` |
| ONT ultra-long | HG002; GIAB | [HG002_ONT-UL_GIAB_20200204.fastq.gz](https://ftp-trace.ncbi.nlm.nih.gov/giab/ftp/data/AshkenazimTrio/HG002_NA24385_son/Ultralong_OxfordNanopore/guppy-V3.4.5/HG002_ONT-UL_GIAB_20200204.fastq.gz) |
| PacBio HiFi (Revio, May 2024) | HG002 | [hg002v1.0.1_hifi_revio_pbmay24.bam](https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/HG002/assemblies/polishing/HG002/v1.0/mapping/hifi_revio_pbmay24/hg002v1.0.1_hifi_revio_pbmay24.bam) |
| PacBio Onso | HG002; HPRC | `s3://human-pangenomics/T2T/scratch/HG002/sequencing/onso/Broad-Onso-HG002/HG002/` |
| Illumina PCR-free 2×250 bp | HG002; GIAB | [HG002.GRCh38.2x250.bam](https://ftp-trace.ncbi.nlm.nih.gov/giab/ftp/data/AshkenazimTrio/HG002_NA24385_son/NIST_Illumina_2x250bps/novoalign_bams/HG002.GRCh38.2x250.bam) |
| Illumina Hi-C | HG002; HPRC | `s3://human-pangenomics/working/HPRC_PLUS/HG002/raw_data/hic/downsampled/` |
| Element AVITI 2×150 bp, ~500–600 bp insert | HG002; HPRC; AVITI chemistry | `s3://human-pangenomics/T2T/scratch/HG002/sequencing/element/trio/HG002/ins500_600/` |
| Element AVITI 2×150 bp, ~1 kbp insert | HG002; HPRC; AVITI chemistry | `s3://human-pangenomics/T2T/scratch/HG002/sequencing/element/trio/HG002/ins1000/` |
| BGISEQ-500 PCR-free 2×150 bp (L1) | HG002; GIAB | [BGISEQ500_PCRfree_NA24385_CL100076190_L01_read_{1,2}.fq.gz](https://ftp-trace.ncbi.nlm.nih.gov/giab/ftp/data/AshkenazimTrio/HG002_NA24385_son/BGISEQ500/) |
| ONT direct RNA (RNA004, sup) | Universal Human Reference (UHR) RNA | [PNXRXX240011_dorado7213sup.fastq.gz](https://gtgseq.s3.amazonaws.com/ont-rna004-rna/UHR/analyses/basecalls/dorado7213sup/PNXRXX240011_dorado7213sup.fastq.gz) |
| GRCh38 (human reference) | Genome Reference Consortium | [GCA_000001405.15_GRCh38_genomic.fna.gz](https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/GCA_000001405.15_GRCh38/GCA_000001405.15_GRCh38_genomic.fna.gz) |
| GRCm38 (mouse reference) | Genome Reference Consortium | [GCF_000001635.20_GRCm38_genomic.fna.gz](https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/635/GCF_000001635.20_GRCm38/GCF_000001635.20_GRCm38_genomic.fna.gz) |
| T2T-CHM13v2.0 (human reference) | Telomere-to-Telomere Consortium | [GCA_009914755.4_T2T-CHM13v2.0_genomic.fna.gz](https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/009/914/755/GCA_009914755.4_T2T-CHM13v2.0/GCA_009914755.4_T2T-CHM13v2.0_genomic.fna.gz) |

## Performance Comparison (8 Threads)

CPU time, wall time, and peak RSS for rammap (rm) vs minimap2 (mm2). **Bold**
marks the better outcome per metric (lower is better). In all cases, rammap
produces identical alignment output to minimap2 (with the exception of SAM
headers including the tool name itself).

### Long-Read Presets

| Dataset | Preset | rm CPU (s) | mm2 CPU (s) | rm Wall | mm2 Wall | rm Mem | mm2 Mem |
|---------|--------|-----------:|------------:|--------:|---------:|-------:|--------:|
| ONT R10 | `-cx lr:hq` | **230538** | 260983 | **8:16:02** | 10:23:27 | **13.0 GB** | 16.4 GB |
| ONT UL | `-cx map-ont` | **664349** | 684362 | **25:30:04** | 26:00:17 | 16.6 GB | **14.1 GB** |
| ONT UL (1M reads) | `-cx ava-ont` | **30446** | 31645 | **1:10:37** | 1:16:12 | 47.2 GB | **38.1 GB** |
| PacBio HiFi | `-cx map-hifi` | **274260** | 318231 | **9:53:33** | 11:08:45 | **12.9 GB** | 16.1 GB |

### Splice / RNA Presets

| Dataset | Preset | rm CPU (s) | mm2 CPU (s) | rm Wall | mm2 Wall | rm Mem | mm2 Mem |
|---------|--------|-----------:|------------:|--------:|---------:|-------:|--------:|
| ONT RNA | `-cx splice -uf -k14` | **121612** | 131063 | **4:23:17** | 4:37:06 | **21.9 GB** | 22.5 GB |

### Short-Read Presets

| Dataset | Preset | rm CPU (s) | mm2 CPU (s) | rm Wall | mm2 Wall | rm Mem | mm2 Mem |
|---------|--------|-----------:|------------:|--------:|---------:|-------:|--------:|
| BGI PCR-free | `-ax sr` | 432217 | **307897** | 15:16:59 | **12:33:02** | 14.9 GB | **12.6 GB** |
| Illumina PCR-free | `-ax sr` | 355364 | **306620** | 12:44:05 | **11:02:44** | 15.0 GB | **12.9 GB** |
| PacBio Onso | `-ax sr` | 161168 | **152442** | 5:44:25 | **5:20:48** | 14.9 GB | **12.5 GB** |
| AVITI (1 kbp insert) | `-ax sr` | 620332 | **430574** | 21:45:35 | **17:12:46** | 14.9 GB | **12.6 GB** |
| AVITI (500–600 bp insert) | `-ax sr` | 760693 | **608294** | 27:21:27 | **27:05:20** | 14.9 GB | **12.5 GB** |
| Illumina Hi-C | `-ax sr --frag=no` | 2410520 | **1727916** | 83:41:55 | **64:32:11** | 15.6 GB | **13.0 GB** |

### Assembly Presets

| Dataset | Preset | rm CPU (s) | mm2 CPU (s) | rm Wall | mm2 Wall | rm Mem | mm2 Mem |
|---------|--------|-----------:|------------:|--------:|---------:|-------:|--------:|
| GRCh38 / GRCm38 | `-x asm20` | **1522** | 2004 | **0:08:13** | 0:10:15 | 23.5 GB | **21.0 GB** |
| GRCh38 / T2T-CHM13 | `-x asm5` | **6894** | 7084 | **0:54:06** | 1:00:54 | 30.3 GB | **28.2 GB** |
