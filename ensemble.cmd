Universe = vanilla
Executable = ensemble.sh
Output = condor_output/ensemble.out
Log = condor_output/ensemble.log
Error = condor_output/ensemble.err
request_memory = 8192
request_GPUs = 1
Requirements = (Machine == "patas-gn2.ling.washington.edu")
Queue