import array

class FileIOException(Exception):
    pass


def write_pgm(numpyarray, file_directory=None):
    if file_directory == None:
        FileIOException('You must specify a file directory!')

    f = open(file_directory, "wb")
    pgmheader = "P5\n" + str(1000) + '  ' + str(1000) + '  ' + str(255) + '\n'

    f.write(pgmheader)
    _buffer = array.array("B")
    _buffer.extend(numpyarray.flatten())

    _buffer.tofile(f)
    f.close()

def read_pgm(file_directory=None):
    f = open(file_directory)
    pgm
    
