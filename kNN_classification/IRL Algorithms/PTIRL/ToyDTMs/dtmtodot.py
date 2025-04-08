#!/usr/bin/env python3

import sys
import pydtm

dtm=pydtm.pydtm(sys.argv[1])
dtm.toDOT(sys.argv[1][:-3]+"dot")


