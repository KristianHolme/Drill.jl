using Drill
using Test
using ParallelTestRunner

# Run all other tests in parallel
ParallelTestRunner.runtests(Drill, ARGS)
