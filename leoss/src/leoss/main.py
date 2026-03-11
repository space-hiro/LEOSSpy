import math

R2D     = 180/math.pi
D2R     = math.pi/180
ZERO    = 1e-12

class Vector:
    __slots__ = ("x", "y", "z")
    def __init__(self, x=0.0, y=0.0, z=0.0):
        '''initialize a 3D vector'''
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

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
        '''
        if not isinstance(other, Vector):
            raise TypeError("Operand must be a Vector")
        return Vector(
            self.y*other.z - self.z*other.y,
            self.z*other.x - self.x*other.z,
            self.x*other.y - self.y*other.x
        )
    
    def crossVectors(self, vec1, vec2):
        '''
        cross product two vectors
        overwrites existing vector with result
        '''
        if not isinstance(vec1, Vector) or not isinstance(vec2, Vector) :
            raise TypeError("Operands must be type Vector")
        self.x = vec1.y*vec2.z - vec1.z*vec2.y
        self.y = vec1.z*vec2.x - vec1.x*vec2.z
        self.z = vec1.x*vec2.y - vec1.y*vec2.x
        return self
    def applyQuaternion(self, q):
        '''
        applies the given quaternion rotation to this vector
        overwrites existing vector with result
        '''
        if not isinstance(q, Quaternion) or q.mag2() - 1 > ZERO:
            raise TypeError("Operand must a unit Quaternion")
        
        vx = self.x
        vy = self.y
        vz = self.z

        # t = 2 * (qv x v)
        tx = 2.0 * (q.y * vz - q.z * vy)
        ty = 2.0 * (q.z * vx - q.x * vz)
        tz = 2.0 * (q.x * vy - q.y * vx)

        # v' = v + qw * t + (qv x t)
        self.x = vx + q.w * tx + (q.y * tz - q.z * ty)
        self.y = vy + q.w * ty + (q.z * tx - q.x * tz)
        self.z = vz + q.w * tz + (q.x * ty - q.y * tx)
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

class Quaternion:
    __slots__ = ("w","x","y","z")
    def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0):
        '''initialize a 4D quaternion'''
        self.w = float(w)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __repr__(self): return f'Quaternion({self.w}, {self.x}, {self.y}, {self.z})'
    def __getitem__(self, item):
        if item == 0 : return self.w
        if item == 1 : return self.x
        if item == 2 : return self.y
        if item == 3 : return self.z
        raise IndexError("There are only four elements in the quaternion")
    def __eq__(self, other):
        if not isinstance(other, Quaternion):
            return NotImplemented
        return self.w == other.w and self.x == other.x and self.y == other.y and self.z == other.z
    def __len__(self): return 4
    def __iter__(self):
        yield self.w
        yield self.x
        yield self.y
        yield self.z
    
    def __add__(self, other):
        if not isinstance(other, Quaternion):
            return NotImplemented
        return Quaternion(
            self.w + other.w,
            self.x + other.x,
            self.y + other.y,
            self.z + other.z
        )
    def __sub__(self, other):
        if not isinstance(other, Quaternion):
            return NotImplemented
        return Quaternion(
            self.w - other.w,
            self.x - other.x,
            self.y - other.y,
            self.z - other.z
        )
    def __mul__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(
                self.w*other.w - self.x*other.x - self.y*other.y - self.z*other.z,
                self.w*other.x + self.x*other.w + self.y*other.z - self.z*other.y,
                self.w*other.y - self.x*other.z + self.y*other.w + self.z*other.x,
                self.w*other.z + self.x*other.y - self.y*other.x + self.z*other.w
            )
        if isinstance(other, (int, float)):
            return Quaternion(
                self.w * other,
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
        if abs(scalar) < ZERO:
            raise ZeroDivisionError("Cannot divide by zero")
        return Quaternion(self.w / scalar, self.x / scalar, self.y / scalar, self.z / scalar)
    def __neg__(self): return Quaternion(-self.w, -self.x, -self.y, -self.z)

    def copy(self, other):
        if not isinstance(other, Quaternion):
            raise TypeError("Operand must be a Quaternion")
        self.w = other.w
        self.x = other.x
        self.y = other.y
        self.z = other.z
        return self
    def set(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z
        return self

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)
    def add(self, other):
        '''
        adds another quaternion
        overwrites existing quaternion with result
        '''
        if not isinstance(other, Quaternion):
            raise TypeError("Operand must be a Quaternion")
        self.w += other.w
        self.x += other.x
        self.y += other.y
        self.z += other.z
        return self
    def sub(self, other):
        '''
        subtracts another quaternion
        overwrites existing quaternion with result
        '''
        if not isinstance(other, Quaternion):
            raise TypeError("Operand must be a Quaternion")
        self.w -= other.w
        self.x -= other.x
        self.y -= other.y
        self.z -= other.z
        return self
    def scale(self, s):
        '''
        multiply elements with scalar
        overwrites existing quaternion with result
        '''
        if not isinstance(s, (int, float)):
            raise TypeError("Operand must be a scalar")
        self.w *= s
        self.x *= s
        self.y *= s
        self.z *= s
        return self
    def mul(self, other):
        '''
        multiply with another quaternion
        qOut = qSelf ⊗ qInput
        overwrites existing quaternion with result
        '''
        if not isinstance(other, Quaternion):
            raise TypeError("Operand must be a Quaternion")
        w = self.w*other.w - self.x*other.x - self.y*other.y - self.z*other.z
        x = self.w*other.x + self.x*other.w + self.y*other.z - self.z*other.y
        y = self.w*other.y - self.x*other.z + self.y*other.w + self.z*other.x
        z = self.w*other.z + self.x*other.y - self.y*other.x + self.z*other.w
        self.w = w
        self.x = x
        self.y = y
        self.z = z
        return self
    def diff(self, other):
        '''
        quataernion difference or error 
        solves for qOut in qSelf = qInput ​⊗ qOut
        '''
        if not isinstance(other, Quaternion):
            raise TypeError("Operand must be a Quaternion")
        return (other.conjugate() * self).normalize()
    def diff2(self, other):
        '''
        quataernion difference or error 
        solves for qOut in qSelf = qOut ​⊗ qInput
        '''
        if not isinstance(other, Quaternion):
            raise TypeError("Operand must be a Quaternion")
        return (self * other.conjugate()).normalize()
    def normalize(self):
        '''
        normalize current quaternion
        overwrites existing quaternion with result
        '''
        mag = self.mag()
        if mag < ZERO:
            raise ZeroDivisionError("Cannot normalize a zero quaternion")
        self.w /= mag
        self.x /= mag
        self.y /= mag
        self.z /= mag
        return self
    def negate(self):
        '''
        negates current quaternion
        overwrites existing vector with result
        '''
        self.w *= -1
        self.x *= -1
        self.y *= -1
        self.z *= -1
        return self

    def mag2(self): return self.w*self.w + self.x*self.x + self.y*self.y + self.z*self.z
    def mag(self): return math.sqrt(self.mag2())

    def rotate(self, v):
        '''
        returns the rotated vector by this quaternion
            v' = q ⊗ (0, v) ⊗ q*
        uses a slightly faster implementation
            v' = v + 2qv × ( qv ​× v + qw v)
        '''
        if not isinstance(v, Vector):
            raise TypeError("Operand must be a Vector")

        # qv = Quaternion(0.0, v.x, v.y, v.z)
        # r = self * qv * self.conjugate()

        qw = self.w
        qx = self.x
        qy = self.y
        qz = self.z

        vx = v.x
        vy = v.y
        vz = v.z

        # t = 2 * (qv x v)
        tx = 2.0 * (qy * vz - qz * vy)
        ty = 2.0 * (qz * vx - qx * vz)
        tz = 2.0 * (qx * vy - qy * vx)

        # v' = v + qw * t + (qv x t)
        return Vector(
            vx + qw * tx + (qy * tz - qz * ty),
            vy + qw * ty + (qz * tx - qx * tz),
            vz + qw * tz + (qx * ty - qy * tx),
        )
    def setFromAxisAngle(self, axis, angle):
        '''
        sets this quaternion for given axis and angle
        '''
        if not isinstance(axis, Vector):
            raise TypeError("Axis must be a Vector")
        if axis.mag2() - 1 > ZERO:
            raise ValueError("Axis must be a unit Vector")
        half = angle*0.5
        s = math.sin(half)
        self.w = math.cos(half)
        self.x = axis.x * s
        self.y = axis.y * s
        self.z = axis.z * s
        return self

    magnitude = mag
    __str__ = __repr__

class State:
    __slots__ = ("m","p","v","q","w")
    def __init__(self, mass=0.0, position=None, velocity=None, quaternion=None, bodyrate=None):
        self.m = float(mass)
        if position is None: self.p = Vector()
        else:
            if not isinstance(position, Vector):
                raise TypeError("position must be a Vector")
            self.p = Vector(position.x, position.y, position.z)
        if velocity is None: self.v = Vector()
        else:
            if not isinstance(velocity, Vector):
                raise TypeError("velocity must be a Vector")
            self.v = Vector(velocity.x, velocity.y, velocity.z)
        if quaternion is None: self.q = Quaternion(1.0, 0.0, 0.0, 0.0)
        else:
            if not isinstance(quaternion, Quaternion):
                raise TypeError("quaternion must be a Quaternion")
            self.q = Quaternion(quaternion.w, quaternion.x, quaternion.y, quaternion.z)
        if bodyrate is None: self.w = Vector()
        else:
            if not isinstance(bodyrate, Vector):
                raise TypeError("bodyrate must be a Vector")
            self.w = Vector(bodyrate.x, bodyrate.y, bodyrate.z)

    def __repr__(self):
        out = str(self.m)
        for i in range(1,len(self),1):
            out = out + ", " + str(self[i])
        return f'State({out})'
    def __getitem__(self, item):
        if item == 0 : return self.m
        if item == 1 : return self.p
        if item == 2 : return self.v
        if item == 3 : return self.q
        if item == 4 : return self.w
        raise IndexError("There are only five elements in the State")
    def __len__(self): return 5
    def __iter__(self):
        yield self.m
        yield self.p
        yield self.v
        yield self.q
        yield self.w
    def __eq__(self, other):
        if not isinstance(other, State):
            return NotImplemented
        return (
            self.m == other.m and
            self.p == other.p and
            self.v == other.v and
            self.q == other.q and
            self.w == other.w
        )

    def copy(self, other):
        if not isinstance(other, State):
            raise TypeError("Operand must be a State")
        self.m = other.m
        self.p.copy(other.p)
        self.v.copy(other.v)
        self.q.copy(other.q)
        self.w.copy(other.w)
        return self
    def zero(self):
        '''
        set all elements of state to zero
        '''
        self.m = 0.0
        self.p.set(0.0, 0.0, 0.0)
        self.v.set(0.0, 0.0, 0.0)
        self.q.set(0.0, 0.0, 0.0, 0.0)
        self.w.set(0.0, 0.0, 0.0)
        return self
    def add(self, other):
        '''
        adds two states
        overwrites existing state with result
        '''
        if not isinstance(other, State):
            raise TypeError("Operand must be a State")
        self.m += other.m
        self.p.add(other.p)
        self.v.add(other.v)
        self.q.add(other.q)
        self.w.add(other.w)
        return self
    def sub(self, other):
        '''
        subtracts two states
        overwrites existing state with result
        '''
        if not isinstance(other, State):
            raise TypeError("Operand must be a State")
        self.m -= other.m
        self.p.sub(other.p)
        self.v.sub(other.v)
        self.q.sub(other.q)
        self.w.sub(other.w)
        return self
    def add_scaled(self, other, s):
        if not isinstance(other, State):
            raise TypeError("Operand must be a State")
        if not isinstance(s, (int, float)):
            raise TypeError("Operand must be a scalar")

        self.m   += other.m * s
        self.p.x += other.p.x * s
        self.p.y += other.p.y * s
        self.p.z += other.p.z * s

        self.v.x += other.v.x * s
        self.v.y += other.v.y * s
        self.v.z += other.v.z * s

        self.q.w += other.q.w * s
        self.q.x += other.q.x * s
        self.q.y += other.q.y * s
        self.q.z += other.q.z * s

        self.w.x += other.w.x * s
        self.w.y += other.w.y * s
        self.w.z += other.w.z * s
        return self
    def scale(self, other):
        '''
        multiply elements with scalar
        overwrites existing state with result
        '''
        if not isinstance(other, (int, float)):
            raise TypeError("Operand must be a scalar")
        self.m *= other
        self.p.scale(other)
        self.v.scale(other)
        self.q.scale(other)
        self.w.scale(other)
        return self

    __str__ = __repr__
