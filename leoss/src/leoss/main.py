import math
import inspect
import time as clock
from datetime import datetime, timezone

from tqdm import tqdm
import pyIGRF14 as IGRF
import pandas as pd

#### globals

R2D     = 180/math.pi
D2R     = math.pi/180
ZERO    = 1e-12

MU_EARTH_M = 398_600.441_8e9
ER_EARTH_M = 6_378.137e3

#### classes

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
    def zero(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
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
        returns a SCALAR
        '''
        if not isinstance(other, Vector):
            raise TypeError("Operand must be a Vector")
        return self.x*other.x + self.y*other.y + self.z*other.z
    def cross(self, other):
        '''
        cross product with another vector
        returns a NEW vector
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
        Applies a quaternion as a PASSIVE rotation (frame transformation).
        The results overwrite the existing vector.

        Definition:
            Let q_BA represent the rotation from frame A → frame B.

        Then this operation assumes:
            - the current vector is expressed in frame A
            - the result will be expressed in frame B

        Mathematically:
            v_B = q_BA^{-1} ⊗ v_A ⊗ q_BA

        Equivalent to:
            v_B = T_BA @ v_A
            where T_BA is the transformation matrix corresponding to q_BA

        Notes:
            - This is a passive transformation (change of coordinates),
            not an active rotation of a physical vector.
            - q must be a unit quaternion.
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

        # active form:  v' = v + qw * t + (qv x t)
        # passive form: v' = v - qw * t + (qv x t)
        self.x = vx - q.w * tx + (q.y * tz - q.z * ty)
        self.y = vy - q.w * ty + (q.z * tx - q.x * tz)
        self.z = vz - q.w * tz + (q.x * ty - q.y * tx)
        return self
    def applyEuler(self, e):
        '''
        applies the given euler sequence rotation to this vector, ACTIVE rotation
        overwrites existing vector with result
        '''
        q = Quaternion().setFromEuler(e)
        self.applyQuaternion(q)
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
            # column-vector multiplication: v_out = self @ other
            return Vector(
                self.x.x * other.x + self.y.x * other.y + self.z.x * other.z,
                self.x.y * other.x + self.y.y * other.y + self.z.y * other.z,
                self.x.z * other.x + self.y.z * other.y + self.z.z * other.z,
            )

        if isinstance(other, Matrix):
            # column-wise matrix product:
            # (self @ other) columns = self @ each column of other
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
    
    def setFromQuaternion(self, q):
        '''
        Constructs the passive frame transformation matrix from a unit quaternion.

        Storage convention:
            self.x, self.y, self.z are COLUMN vectors

        Definition:
            Let q_BA represent the rotation from frame A -> frame B.

        Then:
            v_B = R_BA @ v_A

        Equivalent quaternion form:
            v_B = q_BA^{-1} ⊗ v_A ⊗ q_BA
        '''
        if not isinstance(q, Quaternion) or abs(q.mag2() - 1.0) > ZERO:
            raise TypeError("Operand must be a unit Quaternion")

        x, y, z, w = q.x, q.y, q.z, q.w

        xx = x * x
        yy = y * y
        zz = z * z
        wx = w * x
        wy = w * y
        wz = w * z
        xy = x * y
        xz = x * z
        yz = y * z

        # First COLUMN
        self.x.x = 1.0 - 2.0 * (yy + zz)
        self.x.y = 2.0 * (xy - wz)
        self.x.z = 2.0 * (xz + wy)

        # Second COLUMN
        self.y.x = 2.0 * (xy + wz)
        self.y.y = 1.0 - 2.0 * (xx + zz)
        self.y.z = 2.0 * (yz - wx)

        # Third COLUMN
        self.z.x = 2.0 * (xz - wy)
        self.z.y = 2.0 * (yz + wx)
        self.z.z = 1.0 - 2.0 * (xx + yy)

        return self

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
    def dot(self, other):
        '''
        quaternion dot product
        returns a scalar
        '''
        if not isinstance(other, Quaternion):
            raise TypeError("Operand must be a Quaternion")
        return self.x*other.x + self.y*other.y + self.z*other.z + self.w*other.w
    def diff(self, other):
        '''
        quaternion difference or error 
        solves for qOut in qSelf = qInput ​⊗ qOut
        '''
        if not isinstance(other, Quaternion):
            raise TypeError("Operand must be a Quaternion")
        return (other.conjugate() * self).normalize()
    def diff2(self, other):
        '''
        quaternion difference or error 
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
    def angleTo(self, other):
        '''
        qngle between this quaternion and the other one
        '''
        if not isinstance(other, Quaternion):
            raise TypeError("Operand must be a Quaternion")
        return 2 * math.acos( abs( clamp(self.dot(other), -1, 1) ))

    def mag2(self): return self.w*self.w + self.x*self.x + self.y*self.y + self.z*self.z
    def mag(self): return math.sqrt(self.mag2())

    def rotate(self, v):
        '''
        returns the rotated vector (new) by this quaternion
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
    def setFromAxisAngle(self, axis: Vector, angle):
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
    def setFromEuler(self, euler):
        '''
        sets this quaternion for given euler sequence rotation
        '''
        cos = math.cos
        sin = math.sin
        c1  = cos(euler.x*0.5)
        c2  = cos(euler.y*0.5)
        c3  = cos(euler.z*0.5)
        s1  = sin(euler.x*0.5)
        s2  = sin(euler.y*0.5)
        s3  = sin(euler.z*0.5)
        
        match(euler.getOrder()):
            case 'XYZ':
                self.x = s1 * c2 * c3 + c1 * s2 * s3
                self.y = c1 * s2 * c3 - s1 * c2 * s3
                self.z = c1 * c2 * s3 + s1 * s2 * c3
                self.w = c1 * c2 * c3 - s1 * s2 * s3
            case 'YXZ':
                self.x = s1 * c2 * c3 + c1 * s2 * s3
                self.y = c1 * s2 * c3 - s1 * c2 * s3
                self.z = c1 * c2 * s3 - s1 * s2 * c3
                self.w = c1 * c2 * c3 + s1 * s2 * s3
            case 'ZXY':
                self.x = s1 * c2 * c3 - c1 * s2 * s3
                self.y = c1 * s2 * c3 + s1 * c2 * s3
                self.z = c1 * c2 * s3 + s1 * s2 * c3
                self.w = c1 * c2 * c3 - s1 * s2 * s3
            case 'ZYX':
                self.x = s1 * c2 * c3 - c1 * s2 * s3
                self.y = c1 * s2 * c3 + s1 * c2 * s3
                self.z = c1 * c2 * s3 - s1 * s2 * c3
                self.w = c1 * c2 * c3 + s1 * s2 * s3
            case 'YZX':
                self.x = s1 * c2 * c3 + c1 * s2 * s3
                self.y = c1 * s2 * c3 + s1 * c2 * s3
                self.z = c1 * c2 * s3 - s1 * s2 * c3
                self.w = c1 * c2 * c3 - s1 * s2 * s3
            case 'XZY':
                self.x = s1 * c2 * c3 - c1 * s2 * s3
                self.y = c1 * s2 * c3 - s1 * c2 * s3
                self.z = c1 * c2 * s3 + s1 * s2 * c3
                self.w = c1 * c2 * c3 + s1 * s2 * s3
            case _:
                raise ValueError("Euler order sequence is invalid")
        return self
    def setFromMatrix(self, matrix: Matrix):
        C = matrix
        B = []
        B.append(0.25*(1+C.trace()))
        B.append(0.25*(1+(2*C.xx)-C.trace()))
        B.append(0.25*(1+(2*C.yy)-C.trace()))
        B.append(0.25*(1+(2*C.zz)-C.trace()))

        BwBx = 0.25*( C.zy-C.yz )
        BwBy = 0.25*( C.xz-C.zx )
        BwBz = 0.25*( C.yx-C.xy )
        ByBz = 0.25*( C.zy+C.yz )
        BzBx = 0.25*( C.xz+C.zx )
        BxBy = 0.25*( C.yx+C.xy )

        b = [math.sqrt(item) for item in B]
        Q = Quaternion()

        if B[0] == max(B):
            Q.set(b[0], BwBx/b[0], BwBy/b[0], BwBz/b[0])
        if B[1] == max(B):
            Q.set(BwBx/b[1], b[1], BxBy/b[1], BzBx/b[1])
        if B[2] == max(B):
            Q.set(BwBy/b[2], BxBy/b[2], b[2], ByBz/b[2])
        if B[3] == max(B):
            Q.set(BwBz/b[3], BzBx/b[3], ByBz/b[3], b[3])

        return Q.conjugate()

    magnitude = mag
    __str__ = __repr__

class Euler:
    __slots__ = ("x", "y", "z", "__order")
    def __init__(self, x=0.0, y=0.0, z=0.0, order = 'XYZ'):
        '''initialize an euler angles sequence rotation'''
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.__order = order

    def __repr__(self): return f'Euler({self.x}, {self.y}, {self.z}, {self.__order})'
    
    def __getitem__(self, item):
        if item == 0 : return self.x
        if item == 1 : return self.y
        if item == 2 : return self.z
        raise IndexError("There are only three elements in the euler")
    def __eq__(self, other):
        if not isinstance(other, Euler):
            return NotImplemented
        return self.x == other.x and self.y == other.y and self.z == other.z
    def __len__(self): return 3
    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def scale(self, s):
        '''
        multiply elements with scalar
        overwrites existing euler with result
        '''
        if not isinstance(s, (int, float)):
            raise TypeError("Operand must be a scalar")
        self.x *= s
        self.y *= s
        self.z *= s
        return self
    def getOrder(self): return self.__order
    def setFromQuaternion(self, q, order=None):
        """
        Set this Euler from a unit quaternion.

        Assumes the same intrinsic Tait-Bryan convention as setFromEuler().
        Overwrites this Euler in-place.
        """
        if not isinstance(q, Quaternion):
            raise TypeError("Operand must be a Quaternion")
        if abs(q.mag2() - 1.0) > ZERO:
            raise ValueError("Quaternion must be a unit Quaternion")

        if order is not None:
            self.__order = order

        x = q.x
        y = q.y
        z = q.z
        w = q.w

        # rotation matrix elements from quaternion
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z

        m11 = 1.0 - 2.0 * (yy + zz)
        m12 = 2.0 * (xy - wz)
        m13 = 2.0 * (xz + wy)

        m21 = 2.0 * (xy + wz)
        m22 = 1.0 - 2.0 * (xx + zz)
        m23 = 2.0 * (yz - wx)

        m31 = 2.0 * (xz - wy)
        m32 = 2.0 * (yz + wx)
        m33 = 1.0 - 2.0 * (xx + yy)

        eps = 1.0 - 1e-12
        asin = math.asin
        atan2 = math.atan2

        match self.__order:
            case 'XYZ':
                self.y = asin(clamp(m13, -1.0, 1.0))
                if abs(m13) < eps:
                    self.x = atan2(-m23, m33)
                    self.z = atan2(-m12, m11)
                else:
                    self.x = atan2(m32, m22)
                    self.z = 0.0
            case 'YXZ':
                self.x = asin(-clamp(m23, -1.0, 1.0))
                if abs(m23) < eps:
                    self.y = atan2(m13, m33)
                    self.z = atan2(m21, m22)
                else:
                    self.y = atan2(-m31, m11)
                    self.z = 0.0
            case 'ZXY':
                self.x = asin(clamp(m32, -1.0, 1.0))
                if abs(m32) < eps:
                    self.y = atan2(-m31, m33)
                    self.z = atan2(-m12, m22)
                else:
                    self.y = 0.0
                    self.z = atan2(m21, m11)
            case 'ZYX':
                self.y = asin(-clamp(m31, -1.0, 1.0))
                if abs(m31) < eps:
                    self.x = atan2(m32, m33)
                    self.z = atan2(m21, m11)
                else:
                    self.x = 0.0
                    self.z = atan2(-m12, m22)
            case 'YZX':
                self.z = asin(clamp(m21, -1.0, 1.0))
                if abs(m21) < eps:
                    self.x = atan2(-m23, m22)
                    self.y = atan2(-m31, m11)
                else:
                    self.x = 0.0
                    self.y = atan2(m13, m33)
            case 'XZY':
                self.z = asin(-clamp(m12, -1.0, 1.0))
                if abs(m12) < eps:
                    self.x = atan2(m32, m22)
                    self.y = atan2(m13, m11)
                else:
                    self.x = atan2(-m23, m33)
                    self.y = 0.0
            case _:
                raise ValueError("Euler order sequence is invalid")

        return self

    __str__ = __repr__

class State:
    __slots__ = (
        "mass",
        "pos",
        "vel",
        "quat",
        "omega"
    )
    def __init__(self, mass=0.0, position=None, velocity=None, quaternion=None, bodyrate=None):
        self.mass = float(mass)
        if position is None: self.pos = Vector()
        else:
            if not isinstance(position, Vector):
                raise TypeError("position must be a Vector")
            self.pos = Vector(position.x, position.y, position.z)
        if velocity is None: self.vel = Vector()
        else:
            if not isinstance(velocity, Vector):
                raise TypeError("velocity must be a Vector")
            self.vel = Vector(velocity.x, velocity.y, velocity.z)
        if quaternion is None: self.quat = Quaternion(1.0, 0.0, 0.0, 0.0)
        else:
            if not isinstance(quaternion, Quaternion):
                raise TypeError("quaternion must be a Quaternion")
            self.quat = Quaternion(quaternion.w, quaternion.x, quaternion.y, quaternion.z)
        if bodyrate is None: self.omega = Vector()
        else:
            if not isinstance(bodyrate, Vector):
                raise TypeError("bodyrate must be a Vector")
            self.omega = Vector(bodyrate.x, bodyrate.y, bodyrate.z)

    def __repr__(self):
        out = ""
        for slot_name in self.__slots__:
            value = getattr(self, slot_name)
            out += f"{slot_name}: {value}\n"
        return out
    def __getitem__(self, item):
        if item == 0 : return self.mass
        if item == 1 : return self.pos
        if item == 2 : return self.vel
        if item == 3 : return self.quat
        if item == 4 : return self.omega
        raise IndexError("There are only five elements in the State")
    def __len__(self): return 5
    def __iter__(self):
        yield self.mass
        yield self.pos
        yield self.vel
        yield self.quat
        yield self.omega
    def __eq__(self, other):
        if not isinstance(other, State):
            return NotImplemented
        return (
            self.mass   == other.mass and
            self.pos    == other.pos and
            self.vel    == other.vel and
            self.quat   == other.quat and
            self.omega  == other.omega
        )

    def copy(self, other):
        if not isinstance(other, State):
            raise TypeError("Operand must be a State")
        self.mass = other.mass
        self.pos.copy(other.pos)
        self.vel.copy(other.vel)
        self.quat.copy(other.quat)
        self.omega.copy(other.omega)
        return self
    def zero(self):
        '''
        set all elements of state to zero
        '''
        self.mass = 0.0
        self.pos.set(0.0, 0.0, 0.0)
        self.vel.set(0.0, 0.0, 0.0)
        self.quat.set(0.0, 0.0, 0.0, 0.0)
        self.omega.set(0.0, 0.0, 0.0)
        return self
    def add(self, other):
        '''
        adds two states
        overwrites existing state with result
        '''
        if not isinstance(other, State):
            raise TypeError("Operand must be a State")
        self.mass += other.mass
        self.pos.add(other.pos)
        self.vel.add(other.vel)
        self.quat.add(other.quat)
        self.omega.add(other.omega)
        return self
    def sub(self, other):
        '''
        subtracts two states
        overwrites existing state with result
        '''
        if not isinstance(other, State):
            raise TypeError("Operand must be a State")
        self.mass -= other.mass
        self.pos.sub(other.pos)
        self.vel.sub(other.vel)
        self.quat.sub(other.quat)
        self.omega.sub(other.omega)
        return self
    def add_scaled(self, other, s):
        if not isinstance(other, State):
            raise TypeError("Operand must be a State")
        if not isinstance(s, (int, float)):
            raise TypeError("Operand must be a scalar")

        self.mass   += other.mass * s
        self.pos.x += other.pos.x * s
        self.pos.y += other.pos.y * s
        self.pos.z += other.pos.z * s

        self.vel.x += other.vel.x * s
        self.vel.y += other.vel.y * s
        self.vel.z += other.vel.z * s

        self.quat.w += other.quat.w * s
        self.quat.x += other.quat.x * s
        self.quat.y += other.quat.y * s
        self.quat.z += other.quat.z * s

        self.omega.x += other.omega.x * s
        self.omega.y += other.omega.y * s
        self.omega.z += other.omega.z * s
        return self
    def scale(self, other):
        '''
        multiply elements with scalar
        overwrites existing state with result
        '''
        if not isinstance(other, (int, float)):
            raise TypeError("Operand must be a scalar")
        self.mass *= other
        self.pos.scale(other)
        self.vel.scale(other)
        self.quat.scale(other)
        self.omega.scale(other)
        return self

    __str__ = __repr__

class Spacecraft:
    __slots__ = (
        "name",
        "size",
        "state",
        "inertia",
        "netFORCE",
        "netTORQUE",
        "netMOMENTUM",
        "customFORCE",
        "customTORQUE",
        "customMOMENTUM",
        "FUNC",
        "planet",
        "__forceFUNC",
        "__torqueFUNC",
        "__momentumFUNC",
        "__customFUNC",
        "__customARGS",
        "__stateRECORD"
    )

    def __init__(self, name):
        self.name    = name
        self.size    = Vector()
        self.state   = State()
        self.inertia = Matrix()

        self.netFORCE    = Vector()
        self.netTORQUE   = Vector()
        self.netMOMENTUM = Vector()
        self.FUNC = {}

        self.customFORCE    = Vector()
        self.customTORQUE   = Vector()
        self.customMOMENTUM = Vector()

        self.planet = None

        self.__forceFUNC    = {}
        self.__torqueFUNC   = {}
        self.__momentumFUNC = {}
        self.__customFUNC   = {}
        self.__customARGS   = {}

        self.__stateRECORD = {}

    def __repr__(self):
        out = ""
        for slot_name in self.__slots__:
            if slot_name.startswith("__"): continue
            value = getattr(self, slot_name)
            out += f"{slot_name}: {value}\n"
        return out

    def setmass(self, mass):
        if not isinstance(mass, (int, float)):
            raise TypeError("Operance should be int or float")
        self.state.mass = float(mass)
    def setsize(self, vector):
        if isinstance(vector, Vector):
            self.size.copy(vector)
        else:
            raise TypeError("Operand should be a Vector")
    def setposition(self, vector):
        if isinstance(vector, Vector):
            self.state.pos.copy(vector)
        else:
            raise TypeError("Operand should be a Vector")
    def setvelocity(self, vector):
        if isinstance(vector, Vector):
            self.state.vel.copy(vector)
        else:
            raise TypeError("Operand should be a Vector")
    def setbodyrate(self, vector):
        if isinstance(vector, Vector):
            self.state.omega.copy(vector).scale(D2R)
        else:
            raise TypeError("Operand should be a vector in 'deg/s'")
    def setquaternion(self, quat):
        if isinstance(quat, Quaternion):
            self.state.quat.copy(quat)
        else:
            raise TypeError("Operand should be a quaternion")

    def derivative(self, state:State, time, dstate:State):

        # self.computeCUSTOM(state, time)
        self.computeEXTERNAL(state, time)

        dstate.mass  = 0

        dstate.pos.copy(state.vel)

        dstate.vel.copy(self.netFORCE).scale(1/state.mass)
        
        omega_q = Quaternion(0.0, state.omega.x, state.omega.y, state.omega.z)
        dstate.quat.copy(state.quat).mul(omega_q).scale(0.5)

        dstate.omega.copy(self.inertia.inverse()*(self.netTORQUE-state.omega.cross(self.netMOMENTUM)))
    
    def computeEXTERNAL(self, state:State, time):
        self.netFORCE.copy(self.customFORCE)
        self.netTORQUE.copy(self.customTORQUE)
        self.netMOMENTUM.copy(self.customMOMENTUM)

        self.netMOMENTUM.add(self.inertia*state.omega)

        for force in self.__forceFUNC.values():
            self.netFORCE.add(force(state, time))

        for torq in self.__torqueFUNC.values():
            self.netTORQUE.add(torq(state, time))

        for moment in self.__momentumFUNC.values():
            self.netMOMENTUM.add(moment(state, time))

    def computeCUSTOM(self, state:State, time):
        self.FUNC.clear()
        self.customFORCE.zero()
        self.customTORQUE.zero()
        self.customMOMENTUM.zero()
        for name, func in self.__customFUNC.items():
            self.FUNC[name] = func(self, state, time, self.__customARGS[name])

    def addFORCE(self, func, desc='NoName'):
        if not callable(func):
            print(f"WARNING: input is not callable, addFORCE failed.")
            return False
        
        sig = inspect.signature(func)

        if len(sig.parameters) != 2:
            print(f"WARNING: input must accept 2 arguments as `func(state, time)`, addFORCE failed.")
            return False

        if not checkFUNC(func, self.state, 0):
            print(f"WARNING: input is not valid, addFORCE failed.")
            return False
        
        if type(func(self.state, 0)) != Vector:
            print(f"WARNING: input does not return Vector, addFORCE failed.")
            return False

        if desc in self.__forceFUNC:
            print(f"WARNING: input force name already exists, func overwritten")
        self.__forceFUNC[desc] = func
        return True
    
    def addTORQUE(self, func, desc='NoName'):
        if not callable(func):
            print(f"WARNING: input is not callable, addTORQUE failed.")
            return False
        
        sig = inspect.signature(func)

        if len(sig.parameters) != 2:
            print(f"WARNING: input must accept 2 arguments as `func(state, time)`, addTORQUE failed.")
            return False
        
        if not checkFUNC(func, self.state, 0):
            print(f"WARNING: input is not valid, addTORQUE failed.")
            return False
        
        if type(func(self.state, 0)) != Vector:
            print(f"WARNING: input does not return Vector, addTORQUE failed.")
            return False
        
        if desc in self.__torqueFUNC:
            print(f"WARNING: input torque name already exists, func overwritten")
        self.__torqueFUNC[desc] = func
        return True
    
    def addMOMENTUM(self, func, desc='NoName'):
        if not callable(func):
            print(f"WARNING: input is not callable, addMOMENTUM failed.")
            return False
        
        sig = inspect.signature(func)

        if len(sig.parameters) != 2:
            print(f"WARNING: input must accept 2 arguments as `func(state, time)`, addMOMENTUM failed.")
            return False

        if not checkFUNC(func, self.state, 0):
            print(f"WARNING: input is not valid, addMOMENTUM failed.")
            return False
        
        if type(func(self.state, 0)) != Vector:
            print(f"WARNING: input does not return Vector, addMOMENTUM failed.")
            return False
        
        if desc in self.__momentumFUNC:
            print(f"WARNING: input momentum name already exists, func overwritten")
        self.__momentumFUNC[desc] = func
        return True

    def addCUSTOM(self, func, desc='', args=[0]):
        if not callable(func):
            print(f"WARNING: input is not callable, addCUSTOM failed.")
            return False
        
        sig = inspect.signature(func)

        if len(sig.parameters) < 3:
            print(sig.parameters)
            print(f"WARNING: input must atleast accept 3 arguments as `func(spacecraft, state, time)`, addCUSTOM failed.")
            return False

        if not checkFUNC(func, self, self.state, 1, args):
            print(f"WARNING: input is not valid, addCUSTOM failed.")
            return False
        
        if not isinstance(func(self, self.state, 0, args), (int, float, Vector, Quaternion)):
            print(f"WARNING: input does not return any valid type, addCUSTOM failed.")
            return False
        
        if desc in self.__customFUNC:
            print(f"WARNING: input func name already exists, func overwritten")
        if desc=='':
            desc = func.__name__
        self.__customFUNC[desc] = func
        self.__customARGS[desc] = args
        return True
    
    def initRECORD(self):
        self.__stateRECORD['Time'] = [self.planet.time]
        #### state
        self.__stateRECORD['Position']    = [Vector().copy(self.state.pos)]
        self.__stateRECORD['Velocity']    = [Vector().copy(self.state.vel)]
        self.__stateRECORD['Quaternion']  = [Quaternion().copy(self.state.quat)]
        self.__stateRECORD['Bodyrate']    = [Vector().copy(self.state.omega)]
        #### externals
        self.__stateRECORD['NetForce']      = [Vector().copy(self.netFORCE)]          
        self.__stateRECORD['NetTorque']     = [Vector().copy(self.netTORQUE)]
        self.__stateRECORD['NetMomentum']   = [Vector().copy(self.netMOMENTUM)]
        #### energy
        self.__stateRECORD['SpecificOrbitalEnergy'] \
            = [ (self.state.vel.mag2()/2 - (self.planet.mu/self.state.pos.mag())) ]
        self.__stateRECORD['SpecificAngularMomentum'] \
            = [ self.state.pos.cross(self.state.vel).mag() ]
        self.__stateRECORD['BodyAngularMomentum'] \
            = [ (self.inertia*self.state.omega).mag() ]
        self.__stateRECORD['RotationalKineticEnergy'] \
            = [ 0.5 * self.state.omega.dot(self.inertia*self.state.omega) ]
        ### custom
        for name in self.__customFUNC.keys():
            self.__stateRECORD[name] = [ self.FUNC[name] ]

    def getRECORD(self):
        return self.__stateRECORD
    
    def updateRECORD(self, deltaTime):
        self.__stateRECORD['Time'].append(self.planet.time + deltaTime)
        #### state
        self.__stateRECORD['Position'].append(Vector().copy(self.state.pos))
        self.__stateRECORD['Velocity'].append(Vector().copy(self.state.vel))
        self.__stateRECORD['Quaternion'].append(Quaternion().copy(self.state.quat))
        self.__stateRECORD['Bodyrate'].append(Vector().copy(self.state.omega))
        #### externals
        self.__stateRECORD['NetForce'].append(Vector().copy(self.netFORCE))          
        self.__stateRECORD['NetTorque'].append(Vector().copy(self.netTORQUE))
        self.__stateRECORD['NetMomentum'].append(Vector().copy(self.netMOMENTUM))
        #### energy
        self.__stateRECORD['SpecificOrbitalEnergy'] \
            .append(self.state.vel.mag2()/2 - (self.planet.mu/self.state.pos.mag()))
        self.__stateRECORD['SpecificAngularMomentum'] \
            .append(self.state.pos.cross(self.state.vel).mag())
        self.__stateRECORD['BodyAngularMomentum'] \
            .append( (self.inertia*self.state.omega).mag() )
        self.__stateRECORD['RotationalKineticEnergy'] \
            .append( 0.5 * self.state.omega.dot(self.inertia*self.state.omega) )
        ### custom
        for name in self.__customFUNC.keys():
            self.__stateRECORD[name].append( self.FUNC[name] )

    def checkConservation(self, tol=ZERO):
        OE = self.__stateRECORD['SpecificOrbitalEnergy']
        OM = self.__stateRECORD['SpecificAngularMomentum']
        RM = self.__stateRECORD['BodyAngularMomentum']
        RE = self.__stateRECORD['RotationalKineticEnergy']

        relOE = abs(OE[-1] - OE[0]) / max(abs(OE[0]), ZERO)
        relOM = abs(OM[-1] - OM[0]) / max(abs(OM[0]), ZERO)
        relRM = abs(RM[-1] - RM[0]) / max(abs(RM[0]), ZERO) if abs(RM[0]) > ZERO else abs(RM[-1] - RM[0])
        relRE = abs(RE[-1] - RE[0]) / max(abs(RE[0]), ZERO) if abs(RE[0]) > ZERO else abs(RE[-1] - RE[0])

        print("\nInvariants Relative Drift:")
        print("\tSpecificOrbitalEnergy    :  ", '%+.6E'%relOE, "\t|",'%+.6E'%OE[0],"\t|",'%+.6E'%OE[-1])
        print("\tSpecificAngularMomentum  :  ", '%+.6E'%relOM, "\t|",'%+.6E'%OM[0],"\t|",'%+.6E'%OM[-1])
        print("\tBodyAngularMomentum      :  ", '%+.6E'%relRM, "\t|",'%+.6E'%RM[0],"\t|",'%+.6E'%RM[-1])
        print("\tRotationalKineticEnergy  :  ", '%+.6E'%relRE, "\t|",'%+.6E'%RE[0],"\t|",'%+.6E'%RE[-1])
        print("\n")

        return relOE < tol and relOM < tol and relRM < tol and relRE < tol
        
    __str__ = __repr__

class Planet:
    __slots__ = (
        "spacecraftObjects",    #### dictionary of spacecraft objects
        "time",                 #### elapsed time from __unix in seconds
        "radi",                 #### radius in meters
        "mu",                   #### gravitational parameter in m^3/s^2
        "__unix"                #### time since unix epoch in seconds
    )

    def __init__(self):
        self.spacecraftObjects = {}
        self.time = 0.0
        self.mu   = 0.0
        self.radi = 0.0

        self.__unix = clock.time()
    
    def setAs(self, name):
        if not isinstance(name, str):
            raise TypeError("Input argument should be a string")
        match(name.lower()):
            case 'earth':
                self.mu   = MU_EARTH_M
                self.radi = ER_EARTH_M
            case _:
                print("WARNING: unknown planet name, set failed.")
        return self
    
    def setEpoch(self, year=0, month=0, day=0, hour=0, minute=0, second=0, microsecond=0):     
        datetime_utc = datetime(year, month, day, hour, minute, second, microsecond, tzinfo=timezone.utc)
        self.__unix = datetime_utc.timestamp()
    def setEpochFromDatetime(self, dt: datetime):
        self.__unix = dt.timestamp()
    def getEpoch(self):
        return self.__unix
    def getEpochDatetime(self):
        return datetime.fromtimestamp(self.__unix)
    def getCurrentUnix(self):
        return self.__unix + self.time
    def getCurrentDatetime(self):
        return datetime.fromtimestamp(self.__unix + self.time)

    def addSpacecraft(self, name):
        spacecraft = Spacecraft(name)
        self.spacecraftObjects[name] = spacecraft
        spacecraft.planet = self
    def getSpacecrafts(self):
        return self.spacecraftObjects
    
    def step(self, deltaTime):
        for spacecraft in self.spacecraftObjects.values():
            runggeKutta4(spacecraft.derivative, spacecraft.state, self.time, deltaTime)

            spacecraft.computeCUSTOM(spacecraft.state, self.time + deltaTime)
            spacecraft.computeEXTERNAL(spacecraft.state, self.time + deltaTime)
            spacecraft.updateRECORD(deltaTime)

        self.time += deltaTime
    def INIT(self):
        for spacecraft in self.spacecraftObjects.values():
            spacecraft.addFORCE(self.gravity, "GRAVITY_2BODY")

            spacecraft.computeCUSTOM(spacecraft.state, self.time)
            spacecraft.computeEXTERNAL(spacecraft.state, self.time)
            spacecraft.initRECORD()

    def gravity(self, state, time):
        rho = state.pos.mag()
        out = Vector().copy(state.pos)
        return out.scale(-(self.mu*state.mass/(rho*rho*rho)))

#### functions

def checkFUNC(func, *args, **kwargs):
    try:
        func(*args, **kwargs)
        return True
    except Exception as error:
        print(f"WARNING: check>{func.__name__}, {error}")
        return False

__RK4__k1   = State().zero()
__RK4__k2   = State().zero()
__RK4__k3   = State().zero()
__RK4__k4   = State().zero()
__RK4__tmp  = State().zero()
def runggeKutta4(derivative, state: State, time, deltaTime):
    __RK4__k1.zero()
    __RK4__k2.zero()
    __RK4__k3.zero()
    __RK4__k4.zero()
    __RK4__tmp.zero()

    derivative(state, time, __RK4__k1)

    __RK4__tmp.copy(state)
    __RK4__tmp.add_scaled(__RK4__k1, 0.5 * deltaTime)
    __RK4__tmp.quat.normalize()
    derivative(__RK4__tmp, time + 0.5 * deltaTime, __RK4__k2)

    __RK4__tmp.copy(state)
    __RK4__tmp.add_scaled(__RK4__k2, 0.5 * deltaTime)
    __RK4__tmp.quat.normalize()
    derivative(__RK4__tmp, time + 0.5 * deltaTime, __RK4__k3)

    __RK4__tmp.copy(state)
    __RK4__tmp.add_scaled(__RK4__k3, deltaTime)
    __RK4__tmp.quat.normalize()
    derivative(__RK4__tmp, time + deltaTime, __RK4__k4)

    state.add_scaled(__RK4__k1, deltaTime / 6.0)
    state.add_scaled(__RK4__k2, deltaTime / 3.0)
    state.add_scaled(__RK4__k3, deltaTime / 3.0)
    state.add_scaled(__RK4__k4, deltaTime / 6.0)
    state.quat.normalize()

def simulateProgress(system: Planet, timeEnd, timeStep=1/4):
    print("\nRun Simulation (from "+str(system.time)+" to "+str(timeEnd)+", step="+str(timeStep)+")")
    t0 = clock.time()

    pbar = tqdm(total=timeEnd-system.time, position=0, desc='Simulating', bar_format='{l_bar}{bar:25}{r_bar}{bar:-25b}')
    
    ## system init
    system.INIT()

    while(system.time < timeEnd):
        prev_time = system.time

        system.step(timeStep)

        pbar.update(system.time - prev_time)
    pbar.close()

    t1 = clock.time()
    print("\nElapsed Time:\t"+str(t1-t0)+" sec.")

def simulate(system: Planet, timeEnd, timeStep=1/4, init = True):
    ## system init
    if (init):
        system.INIT()

    while(system.time < timeEnd):
        system.step(timeStep)

def clamp(v, lo, hi): return lo if v < lo else hi if v > hi else v

def julianDay(year, month, day): return (367*year - int((7*(year + int((month+9)/12)))/4) + int(275*month/9) + day + 1_721_013.5)

def greenwhichMST(year, month, day, hour, minute, second, microsecond):
    julian_day  = julianDay(year, month, day)
    T0          = (julian_day - 2_451_545) / 36_525
    GMST0       = ( 100.460_618_4 + 36_000.770_04 * (T0*T0) - (2.583e-8)*(T0*T0*T0) ) % 360
    hours       = hour + minute/60 + second/3_600 + microsecond/3_600_000_000
    GMST        = GMST0 + 360.985_647_24 * hours/24
    return (GMST % 360)

def ned_to_ecef_Matrix(lat, lon):
    sLat = math.sin(lat)
    cLat = math.cos(lat)
    sLon = math.sin(lon)
    cLon = math.cos(lon)

    # NED basis vectors expressed in ECEF
    north = Vector(-sLat * cLon, -sLat * sLon,  cLat)
    east  = Vector(-sLon,         cLon,          0.0)
    down  = Vector(-cLat * cLon, -cLat * sLon, -sLat)

    return Matrix(north, east, down)

def ecef_to_eci_Matrix(gmst):
    c = math.cos(gmst)
    s = math.sin(gmst)

    x_ecef_in_eci = Vector( c, s, 0.0)
    y_ecef_in_eci = Vector(-s, c, 0.0)
    z_ecef_in_eci = Vector(0.0, 0.0, 1.0)

    return Matrix(x_ecef_in_eci, y_ecef_in_eci, z_ecef_in_eci)

def eci_to_orf_Matrix(pos: Vector, vel: Vector):
    ## z = -pos normalized
    Z = Vector().copy(pos).normalize().scale(-1)
    ## y = z x vel normalized
    Y = Vector().copy(Z.cross(vel)).normalize()
    ## x = y x z normalize
    X = Vector().copy(Y.cross(Z)).normalize()
    return Matrix(X,Y,Z) 

#### built-in custom functions

def __geolocation(sc: Spacecraft, st: State, time: float, args=[0]):
    '''computes for coordinates, latitude (deg), longitude (deg) and altitude (km)'''
    sc_dt   = sc.planet.getCurrentDatetime()
    sc_gmst = greenwhichMST(sc_dt.year, sc_dt.month, sc_dt.day, sc_dt.hour, sc_dt.minute, sc_dt.second, sc_dt.microsecond)

    position    = st.pos
    mag         = position.mag()
    radi        = sc.planet.radi

    theta = math.acos(position.z/mag)
    psi   = math.atan2(position.y,position.x)

    latitude  = 90 - (theta*R2D)
    longitude = psi*R2D
    altitude  = (mag-radi)/1_000

    xy = math.sqrt(position.x**2+position.y**2)

    gd_theta = latitude*D2R
    C   = 0
    gd  = 0
    e2  = 0.006_694_385_000

    while True:
        C   = radi/math.sqrt(1-e2*math.sin(gd_theta)*math.sin(gd_theta))
        gd  = math.atan2(position.z+C*e2*math.sin(gd_theta),xy)
        if abs(gd-gd_theta) < 1e-6:
            gd_theta = gd
            break
        gd_theta = gd
    
    h_ellp = ( xy/math.cos(gd_theta) ) - C  
    
    altitude = h_ellp/1e3
    latitude = gd_theta*R2D
    longitude = longitude - sc_gmst

    if longitude < 0:
        longitude = (((longitude/360) - int(longitude/360)) * 360) + 360    
    if longitude > 180:
        longitude = -360 + longitude
    
    location = Vector(latitude, longitude, altitude)
    return location

def __geomagfield(sc: Spacecraft, st: State, time: float, args=[0]):
    '''computes for body reference magnetic field strength'''
    location = sc.FUNC['GlobalPosition'] if len(sc.FUNC) > 0 else Vector() 
    lat_deg = location.x
    lon_deg = location.y
    alt_km  = location.z
    B_NED   = IGRF.igrf_value(lat_deg, lon_deg, alt_km, 2026)[3:6]
    B_NED   = Vector().set(B_NED[0], B_NED[1], B_NED[2]).scale(1e-9)
    
    lat_rad = lat_deg * D2R
    lon_rad = lon_deg * D2R

    position = st.pos
    psi      = math.atan2(position.y,position.x)
    gmst_deg = psi*R2D - lon_deg

    C_ecef_ned  = ned_to_ecef_Matrix(lat_rad, lon_rad)
    C_eci_ecef  = ecef_to_eci_Matrix(gmst_deg * D2R)
    C_eci_ned   = C_eci_ecef * C_ecef_ned

    B_ECI   = C_eci_ned * B_NED
    B_BODY  = st.quat.conjugate().rotate(B_ECI)
    return B_BODY

def __bdotcontrol(sc: Spacecraft, st: State, time: float, args=[0]):
    control_torque = Vector()
    control_moment = Vector()

    record = sc.getRECORD()
    if not record or len(record['Time']) < 1:
        return control_torque.zero()

    B_now = Vector().copy(sc.FUNC['GeomagField'])
    B_prev = Vector().copy(record['GeomagField'][-1])

    dt = time - record['Time'][-1]
    if dt <= 0.0:
        return control_torque.zero()

    # finite-difference B-dot
    B_dot = Vector().copy(B_now).sub(B_prev).scale(1.0 / dt)

    # start with a much smaller gain first
    K = args[0]
    control_moment.copy(B_dot).scale(-K)

    # Optional saturation to avoid absurd dipole commands
    # m_max = 0.2  # A·m², example only
    # m_mag = control_moment.mag()
    # if m_mag > m_max and m_mag > ZERO:
    #     control_moment.scale(m_max / m_mag)

    control_torque.copy(control_moment.cross(B_now))
    sc.customTORQUE.add(control_torque)
    return control_torque

class _FuncNameSpace:
    __slots__ = (
        "geolocation",
        "geomagfield",
        "bdotcontrol",
    )

FUNC = _FuncNameSpace
FUNC.geolocation = __geolocation
FUNC.geolocation.__name__ = "GlobalPosition"
FUNC.geomagfield = __geomagfield
FUNC.geomagfield.__name__ = "GeomagField"
FUNC.bdotcontrol = __bdotcontrol
FUNC.bdotcontrol.__name__ = "BdotControl"
