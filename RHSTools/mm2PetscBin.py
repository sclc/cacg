import sys, scipy.io, PetscBinaryIO
assert (len(sys.argv) < 3 and len(sys.argv) > 1), "argument number wrong"
print "start converting", sys.argv[1]
A = scipy.io.mmread(sys.argv[1])
outputName = sys.argv[1]+".pbin"
PetscBinaryIO.PetscBinaryIO().writeMatSciPy(open(outputName,'w'), A)
print "converting done"
