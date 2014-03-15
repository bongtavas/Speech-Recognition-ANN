from features import mfcc
from features import logfbank
import scipy.io.wavfile as wav
import hmm_faster.py

(rate,sig) = wav.read("test2.wav")
mfcc_feat = mfcc(sig,rate)


hmm_model = hmm_faster.HMM()

