import numpy as np
import tempfile
import atexit

class ArrayMMapProxy(object):
    __slots__ = ('_array', '_params',)

    def __init__(self, array):
        if isinstance(array, np.memmap):
            order = 'F' if np.isfortran(array) else 'C'
            self._params = (array.filename, array.dtype, array.mode,
                            array.offset, array.shape, order)
        else:
            raise TypeError('Unsupported type %s' % type(array))

        self._array = array


    @classmethod
    def fromarray(cls, array, tmpdir=None, mode='r'):
        f = tempfile.NamedTemporaryFile(dir=tmpdir)
        atexit.register(lambda: f.close())
        array.tofile(f.name) # just f does not work?
        return cls(np.memmap(f.name, array.dtype, mode,
                             0, array.shape, 'C'))


    def __getstate__(self):
        return self._params


    def __setstate__(self, state):
        self._params = state
        self._array = np.memmap(*self._params)


    def get(self):
        return self._array
