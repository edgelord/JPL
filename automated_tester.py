import detect as dt
import pgmio as io
import time
import numpy as np


def test_site1():
    print "Testing for Set1!"
    dir_site = "resources/Set1/solution.pgm"
    our_np_array = dt.detect(dt.surf1)
    their_np_array = io.read_pgm(dir_site)
    
    testing(our_np_array, their_np_array)

    io.write_pgm(our_np_array, "results/test1_our.pgm")
    io.write_pgm(their_np_array, "results/test1_their.pgm")
    
def test_site2():
    print "Testing for Set2!"
    dir_site = "resources/Set2/solution.pgm"
    our_np_array = dt.detect(dt.surf2)
    their_np_array = io.read_pgm(dir_site)
    
    testing(our_np_array, their_np_array)
    io.write_pgm(our_np_array, "results/test2_our.pgm")
    io.write_pgm(their_np_array, "results/test2_their.pgm")

def test_site3():
    print "Testing for Set3!"
    dir_site = "resources/Set3/solution.pgm"
    our_np_array = dt.detect(dt.surf3)
    their_np_array = io.read_pgm(dir_site)
    
    testing(our_np_array, their_np_array)
    io.write_pgm(our_np_array, "results/test3_our.pgm")
    io.write_pgm(their_np_array, "results/test3_their.pgm")

def test_site4():
    print "Testing for Set4!"
    dir_site = "resources/Set4/solution.pgm"
    our_np_array = np.transpose(dt.detect(dt.surf4))
    their_np_array = io.read_pgm(dir_site)
    
    testing(our_np_array, their_np_array)
    io.write_pgm(our_np_array, "results/test4_our.pgm")
    io.write_pgm(their_np_array, "results/test4_their.pgm")

def testing(our_np_array, their_np_array):
    a = time.time()
    iter_result = our_np_array == their_np_array

    count = 0
    total = 0
    for i in iter_result:
        if i:
            count += 1
        total += 1

    print "-" * 30
    print "You have an accuracy of %10.4f%%" % (float(count)/float(total) * 100)

    b = time.time()
    print "Time: %10.10f" % (b-a)

if __name__ == '__main__':
    # Only available for judge presentation
    pass
