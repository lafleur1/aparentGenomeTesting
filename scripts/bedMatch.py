# bedMatch.py - grab the annotated BED file and
#  compare whether our results match the annotations

# imports
import numpy as np
from Bio import SeqIO

# load BED file

# build a structure of [chromosome, region, polyA or not]

# threshold our model esult files (.npy files) to determine regions with polyA signals


# compare algo:
# for each result from our model, check whether it is True positive or False positive
# for each remaining labeled polyA region from BED file, check for False negative
# can't get True negatives (right?)
