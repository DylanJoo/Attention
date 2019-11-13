#TEST CASE

from TFmodule import *
from TFutils import *

mhattn = MultiheadAttention(100,2)
ff = nn.Linear(100, 2048)
position = PositionalEncoding(100, 0.5)

Encoder(EncoderLayer(100, mhattn, (ff, ff), 0.5), 2)
