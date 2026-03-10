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
