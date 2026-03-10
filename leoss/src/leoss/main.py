import math

R2D     = 180/math.pi
D2R     = math.pi/180
ZERO    = 1e-12

class Vector:
    __slots__ = ("x", "y", "z")
    def __init__(self, x=0.0, y=0.0, z=0.0):
        '''initialize a 3D vector'''
        self.x = x
        self.y = y
        self.z = z
    def __repr__(self): return f'Vector({self.x}, {self.y}, {self.z})'
    def __getitem__(self, item):
        if item == 0 : return self.x
        if item == 1 : return self.y
        if item == 2 : return self.z
        raise IndexError("There are only three elements in the vector")
    def __eq__(self, other):
        if not isinstance(other, Vector):
            return NotImplemented
        return self.x == other.x and self.y == other.y and self.z == other.z
    def __len__(self): return 3
    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z
    def __add__(self, other):
        if not isinstance(other, Vector):
            return NotImplemented
        return Vector(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z
        )
    def __sub__(self, other):
        if not isinstance(other, Vector):
            return NotImplemented
        return Vector(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z
        )
    def __mul__(self, scalar):
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return Vector(self.x * scalar, self.y * scalar, self.z * scalar)
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    def __truediv__(self, scalar):
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        if abs(scalar) < ZERO:
            raise ZeroDivisionError("Cannot divide by zero")
        return Vector(self.x / scalar, self.y / scalar, self.z / scalar)
    def __neg__(self): return Vector(-self.x, -self.y, -self.z)

    def copy(self, other):
        if not isinstance(other, Vector):
            raise TypeError("Operand must be a Vector")
        self.x = other.x
        self.y = other.y
        self.z = other.z
        return self
    def set(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        return self

    def add(self, other):
        '''
        adds another vector
        overwrites existing vector with result
        '''
        if not isinstance(other, Vector):
            raise TypeError("Operand must be a Vector")
        self.x += other.x
        self.y += other.y
        self.z += other.z
        return self
    def sub(self, other):
        '''
        subtracts another vector
        overwrites existing vector with result
        '''
        if not isinstance(other, Vector):
            raise TypeError("Operand must be a Vector")
        self.x -= other.x
        self.y -= other.y
        self.z -= other.z
        return self
    def scale(self, s):
        '''
        multiply elements with scalar
        overwrites existing vector with result
        '''
        if not isinstance(s, (int, float)):
            raise TypeError("Operand must be a scalar")
        self.x *= s
        self.y *= s
        self.z *= s
        return self
    def normalize(self):
        '''
        normalize current vector
        overwrites existing vector with result
        '''
        mag = self.mag()
        if mag < ZERO:
            raise ZeroDivisionError("Cannot normalize a zero vector")
        self.x /= mag
        self.y /= mag
        self.z /= mag
        return self
    def negate(self):
        '''
        negates current vector
        overwrites existing vector with result
        '''
        self.x *= -1
        self.y *= -1
        self.z *= -1
        return self

    def dot(self, other):
        '''
        dot product with another vector
        '''
        if not isinstance(other, Vector):
            raise TypeError("Operand must be a Vector")
        return self.x*other.x + self.y*other.y + self.z*other.z
    
    def cross(self, other):
        '''
        cross product with another vector
        overwrites existing vector with result
        '''
        if not isinstance(other, Vector):
            raise TypeError("Operand must be a Vector")
        x = self.y*other.z - self.z*other.y
        y = self.z*other.x - self.x*other.z
        z = self.x*other.y - self.y*other.x

        self.x = x
        self.y = y
        self.z = z
        return self

    def mag2(self): return self.x*self.x + self.y*self.y + self.z*self.z
    def mag(self): return math.sqrt(self.mag2())
    def sum(self): 
        '''add all elements'''
        return self.x + self.y + self.z

    magnitude = mag
    __str__ = __repr__

class Matrix:
    __slots__ = ("x","y","z")

    def __init__(self, x=None, y=None, z=None):
        if x is None: self.x = Vector(1.0, 0.0, 0.0)
        else:
            if not isinstance(x, Vector):
                raise TypeError("x must be a Vector")
            self.x = Vector(x.x, x.y, x.z)

        if y is None: self.y = Vector(0.0, 1.0, 0.0)
        else:
            if not isinstance(y, Vector):
                raise TypeError("y must be a Vector")
            self.y = Vector(y.x, y.y, y.z)

        if z is None: self.z = Vector(0.0, 0.0, 1.0)
        else:
            if not isinstance(z, Vector):
                raise TypeError("z must be a Vector")
            self.z = Vector(z.x, z.y, z.z)

    @property
    def xx(self): return self.x.x
    @property
    def xy(self): return self.x.y
    @property
    def xz(self): return self.x.z
    @property
    def yx(self): return self.y.x
    @property
    def yy(self): return self.y.y
    @property
    def yz(self): return self.y.z
    @property
    def zx(self): return self.z.x
    @property
    def zy(self): return self.z.y
    @property
    def zz(self): return self.z.z

    def __repr__(self):
        return f'Matrix:\n\t{self.xx}, {self.yx}, {self.zx}\n\t{self.xy}, {self.yy}, {self.zy}\n\t{self.xz}, {self.yz}, {self.zz}'
    
    def __eq__(self, other):
        if not isinstance(other, Matrix):
            return NotImplemented
        return self.x == other.x and self.y == other.y and self.z == other.z
    def __mul__(self, other):
        if isinstance(other, Vector):
            return Vector(
                self.x.x * other.x + self.y.x * other.y + self.z.x * other.z,
                self.x.y * other.x + self.y.y * other.y + self.z.y * other.z,
                self.x.z * other.x + self.y.z * other.y + self.z.z * other.z,
            )

        if isinstance(other, Matrix):
            return Matrix(
                self * other.x,
                self * other.y,
                self * other.z,
            )

        if isinstance(other, (int, float)):
            return Matrix(
                self.x * other,
                self.y * other,
                self.z * other,
            )

        return NotImplemented
    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return self * other
        return NotImplemented
    def __truediv__(self, scalar):
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return Matrix(
            self.x / scalar,
            self.y / scalar,
            self.z / scalar,
        )

    def copy(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("Operand must be a Matrix")
        self.x.copy(other.x)
        self.y.copy(other.y)
        self.z.copy(other.z)
        return self

    def row(self, i):
        if i == 0:
            return Vector(self.x.x, self.y.x, self.z.x)
        if i == 1:
            return Vector(self.x.y, self.y.y, self.z.y)
        if i == 2:
            return Vector(self.x.z, self.y.z, self.z.z)
        raise IndexError("Matrix has only 3 rows")
    def col(self, i):
        if i == 0:
            return Vector(self.x.x, self.x.y, self.x.z)
        if i == 1:
            return Vector(self.y.x, self.y.y, self.y.z)
        if i == 2:
            return Vector(self.z.x, self.z.y, self.z.z)
        raise IndexError("Matrix has only 3 columns")

    def transpose(self):
        return Matrix(
            Vector(self.x.x, self.y.x, self.z.x),
            Vector(self.x.y, self.y.y, self.z.y),
            Vector(self.x.z, self.y.z, self.z.z),
        )
    def trace(self): return self.x.x + self.y.y + self.z.z

    def inverse(self):
        '''
        Fastest implementation for inverse matrix 'vs. np.linalg.inv()'
        ---------------------------------------------------------------
        Reference:
        [1] https://stackoverflow.com/questions/42489310/matrix-inversion-3-3-python-hard-coded-vs-numpy-linalg-inv
        ---------------------------------------------------------------
        '''
        m1 = self.xx; m2 = self.yx; m3 = self.zx
        m4 = self.xy; m5 = self.yy; m6 = self.zy
        m7 = self.xz; m8 = self.yz; m9 = self.zz
        
        x = Vector( m5*m9-m6*m8, m6*m7-m4*m9, m4*m8-m5*m7 )
        y = Vector( m3*m8-m2*m9, m1*m9-m3*m7, m2*m7-m1*m8 )
        z = Vector( m2*m6-m3*m5, m3*m4-m1*m6, m1*m5-m2*m4 )
        inv = Matrix(x, y, z)

        w = Vector(inv.xx, inv.yx, inv.zx)
        det = w.dot(self.x)
        if abs(det) < ZERO:
            raise ZeroDivisionError("Matrix is singular")
        return inv / det
    
    __str__ = __repr__