#!/usr/bin/env python3

from testis import call

# --log-level 0: testis' call() waits for the process before reading its piped
# stdout, so a long stdout ((T) progress lines) would block the run
call("{SISI4S_RUN} -i in.yaml --log-level 0")
