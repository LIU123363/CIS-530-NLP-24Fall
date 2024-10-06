### Append stop word ###
STOP_WORD = False
### Capitalization
CAPITALIZATION = True

### small number
EPSILON = 1e-100

### Inference Types ###
GREEDY = 0
BEAM = 1; BEAM_K = 2
VITERBI = 2
INFERENCE = VITERBI 

### Smoothing Types ###
LAPLACE = 0; LAPLACE_FACTOR = .2
INTERPOLATION = 1; LAMBDAS =  None
KNESERNEY = 2
GOODTURING = 3
SMOOTHING = INTERPOLATION

# NGRAMM
NGRAMM = 3

## Handle unknown words TnT style
TNT_UNK = True
UNK_C = 2 #words with count to be considered, don't change, it's the best threshold
UNK_M = 5 #substring length to be considered
