from voc import *

sample = run("20240730-0001 propane 200 us", flip=False)

sample.smooth()
sample.plot("200us-test")