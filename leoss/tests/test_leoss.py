from leoss import *
import pytest

#### VECTOR
def test_VECTOR_IMPL():
    ### REPR
    v = Vector(1,2,3)
    assert str(v) == "Vector(1.0, 2.0, 3.0)"

    ### GETITEM
    assert v[0] == 1
    assert v[1] == 2
    assert v[2] == 3
    with pytest.raises(IndexError):
        v[3]

    ### EQUALITY
    a = Vector(1,2,3)
    b = Vector(1,2,3)
    c = Vector(4,5,6)
    assert a == b
    assert a != c
    assert a.__eq__(1) is NotImplemented

    ### MAG
    assert v.mag() == v.magnitude() == math.sqrt(14)
    assert v.mag2() == v.mag() * v.mag() == 14

    ### LEN
    assert len(v) == 3

    ### ITER
    a, b, c = v
    assert a == v.x
    assert b == v.y
    assert c == v.z
    assert list(v) == [v.x, v.y, v.z]

    ### COPY-SET
    a = Vector(1,2,3)
    b = Vector(4,5,6)
    tmp = Vector()
    tmp.copy(a)
    assert tmp == a
    tmp.set(b.x, b.y, b.z)
    assert tmp == b

def test_VECTOR_BASEMATH():
    ### SUM
    v = Vector(1,2,3)
    assert v.sum() == 6
    assert v.sum() == v.x + v.y + v.z

    ### NORMALIZE-ZERO
    v = Vector()
    with pytest.raises(ZeroDivisionError, match="zero vector"):
        v.normalize()

    ### __add__ / __sub__
    a = Vector(1,2,3)
    b = Vector(1,2,3)
    assert a + b == Vector(2,4,6)
    assert a - b == Vector(0,0,0)
    assert a == b

    ### SCALAR __mul__
    a = Vector(1,2,3)
    assert a * 2 == Vector(2,4,6)
    assert 2 * a == Vector(2,4,6)

    ### SCALAR __truediv__
    a = Vector(1,2,3)
    assert a / 2 == Vector(0.5,1,1.5)
    with pytest.raises(ZeroDivisionError):
        a / 0

    ### __neg__
    a = Vector(1,2,3)
    assert -a == Vector(-1,-2,-3)  

    ### ADD INPLACE
    a = Vector(1,2,3)
    b = Vector(1,2,3)
    a.add(b)
    assert a == Vector(2,4,6)
    assert a != b

    ### SUB INPLACE
    a = Vector(2,4,6)
    b = Vector(1,2,3)
    a.sub(b)
    assert a == Vector(1,2,3)

    ### SCALE INPLACE
    a = Vector(1,2,3)
    a.scale(2)
    assert a == Vector(2,4,6)
    a.scale(0.5)
    assert a == Vector(1,2,3)

    ### NEGATE
    a = Vector(1,2,3)
    a.negate()
    assert a == Vector(-1,-2,-3)
    a.negate()
    assert a == Vector(1,2,3)

    ### NORMALIZE
    a = Vector(1,2,3)
    a.normalize()
    expected = Vector(
        1/math.sqrt(14),
        2/math.sqrt(14),
        3/math.sqrt(14)
    )
    assert a == expected
    assert a.mag() == 1

def test_VECTOR_TYPE_ERRORS():
    a = Vector(1,2,3)
    tmp = Vector()

    with pytest.raises(TypeError):
        a.add(1)

    with pytest.raises(TypeError):
        a.sub("x")

    with pytest.raises(TypeError):
        a.scale(Vector(1,2,3))

    with pytest.raises(TypeError):
        tmp.copy(5)

def test_VECTOR_MATH():
    ### DOT-BASIS
    X = Vector(2,0,0)
    Y = Vector(0,2,0)
    Z = Vector(0,0,2)
    assert X.dot(X) == 4
    assert X.dot(Y) == 0
    assert X.dot(Z) == 0
    assert Y.dot(X) == 0
    assert Y.dot(Y) == 4
    assert Y.dot(Z) == 0
    assert Z.dot(X) == 0
    assert Z.dot(Y) == 0
    assert Z.dot(Z) == 4

    ### DOT PRODUCT
    a = Vector(1,2,3)
    b = Vector(4,5,6)
    assert a.dot(b) == 32
    assert b.dot(a) == 32

    ### CROSS PRODUCT
    X = Vector(1,0,0)
    Y = Vector(0,1,0)
    Z = Vector(0,0,1)
    tmp = Vector()
    assert tmp.copy(X).cross(Y) == Z
    assert tmp.copy(Y).cross(Z) == X
    assert tmp.copy(Z).cross(X) == Y
    tmp.crossVectors(X, Y)
    assert tmp == Z
    tmp.crossVectors(Y, Z)
    assert tmp == X
    tmp.crossVectors(Z, X)
    assert tmp == Y

    ### CROSS PRODUCT REVERSE
    assert tmp.copy(Y).cross(X) == -Z
    assert tmp.copy(Z).cross(Y) == -X
    assert tmp.copy(X).cross(Z) == -Y
    tmp.crossVectors(Y, X)
    assert tmp == -Z
    tmp.crossVectors(Z, Y)
    assert tmp == -X
    tmp.crossVectors(X, Z)
    assert tmp == -Y

    ### CROSS SELF 
    assert tmp.copy(X).cross(X) == Vector(0,0,0)
    tmp.crossVectors(X,X)
    assert tmp == Vector(0,0,0)

def test_VECTOR_PRODUCTS_TYPE_ERRORS():
    X = Vector(2,0,0)
    with pytest.raises(TypeError):
        X.dot(1)
    with pytest.raises(TypeError):
        X.cross(1)

def test_VECTOR_APPLY_QUATERNION():
    Q_BA = Quaternion(math.sqrt(2)/2, 0.0, 0.0, math.sqrt(2)/2)
    xA = Vector(1.0, 0.0, 0.0)
    yA = Vector(0.0, 1.0, 0.0)
    zA = Vector(0.0, 0.0, 1.0)

    assert (xA.applyQuaternion(Q_BA) - Vector(0.0, -1.0, 0.0)).mag() < ZERO
    assert (yA.applyQuaternion(Q_BA) - Vector(1.0, 0.0, 0.0)).mag() < ZERO
    assert (zA.applyQuaternion(Q_BA) - Vector(0.0, 0.0, 1.0)).mag() < ZERO

    assert ( (Q_BA * Quaternion(0, 1.0, 0.0, 0.0) * Q_BA.conjugate()).vec() - Vector(0.0,-1.0,0.0)).mag() < ZERO
    assert ( (Q_BA * Quaternion(0, 0.0, 1.0, 0.0) * Q_BA.conjugate()).vec() - Vector(1.0,0.0,0.0)).mag() < ZERO
    assert ( (Q_BA * Quaternion(0, 0.0, 0.0, 1.0) * Q_BA.conjugate()).vec() - Vector(0.0,0.0,1.0)).mag() < ZERO

    ### ERROR
    v = Vector(1, 0, 0)
    with pytest.raises(TypeError):
        v.applyQuaternion(1)

#### MATRIX
def test_MATRIX_IMPL():
    #### DEFAULT
    I = Matrix()
    assert I.xx == 1
    assert I.yy == 1
    assert I.zz == 1
    assert I.xy == 0
    assert I.xz == 0
    assert I.yx == 0
    assert I.yz == 0
    assert I.zx == 0
    assert I.zy == 0

    #### TYPE-ERROR
    with pytest.raises(TypeError):
        Matrix(x=1)
    with pytest.raises(TypeError):
        Matrix(y=1)
    with pytest.raises(TypeError):
        Matrix(z=1)

    #### TRANSPOSE
    M = Matrix(
        Vector(1,2,3),
        Vector(4,5,6),
        Vector(7,8,9)
    )
    T = M.transpose()
    assert T.row(0) == Vector(1,2,3)
    assert T.row(1) == Vector(4,5,6)
    assert T.row(2) == Vector(7,8,9)

    #### ROW-COL
    assert M.row(0) == Vector(1,4,7)
    assert M.row(1) == Vector(2,5,8)
    assert M.row(2) == Vector(3,6,9)
    assert M.col(0) == Vector(1,2,3)
    assert M.col(1) == Vector(4,5,6)
    assert M.col(2) == Vector(7,8,9)
    with pytest.raises(IndexError):
        M.row(3)
    with pytest.raises(IndexError):
        M.col(3)

    #### TRACE
    assert M.trace() == 1 + 5 + 9

    #### TRACE
    B = Matrix()
    B.copy(M)
    assert B == M
    with pytest.raises(TypeError):
        B.copy(1)

    ### INVERSE
    I = Matrix()
    inv = I.inverse()
    assert inv == I
    M = Matrix(
        Vector(1,2,3),
        Vector(0,1,4),
        Vector(5,6,0)
    )
    Minv = M.inverse()
    I = M * Minv
    assert I.xx == pytest.approx(1)
    assert I.yy == pytest.approx(1)
    assert I.zz == pytest.approx(1)
    assert I.xy == pytest.approx(0)
    assert I.xz == pytest.approx(0)
    assert I.yx == pytest.approx(0)
    assert I.yz == pytest.approx(0)
    assert I.zx == pytest.approx(0)
    assert I.zy == pytest.approx(0)

    #### SINGULAR
    M = Matrix(
        Vector(1,2,3),
        Vector(2,4,6),
        Vector(3,6,9)
    )
    with pytest.raises(ZeroDivisionError):
        M.inverse()

def test_MATRIX_MATH():
    #### VECTOR PRODUCT
    M = Matrix(
        Vector(1,0,0),
        Vector(0,2,0),
        Vector(0,0,3)
    )
    v = Vector(1,2,3)
    result = M * v
    assert result == Vector(1,4,9)

    #### MATRIX PRODUCT
    A = Matrix(
        Vector(1,0,0),
        Vector(0,2,0),
        Vector(0,0,3)
    )
    B = Matrix(
        Vector(2,0,0),
        Vector(0,3,0),
        Vector(0,0,4)
    )
    C = A * B
    assert C == Matrix(
        Vector(2,0,0),
        Vector(0,6,0),
        Vector(0,0,12)
    )

    #### SCALAR PRODUCT
    M = Matrix(
        Vector(1,2,3),
        Vector(4,5,6),
        Vector(7,8,9)
    )
    R = 2 * M
    assert R == Matrix(
        Vector(2,4,6),
        Vector(8,10,12),
        Vector(14,16,18)
    )
    M = Matrix(
        Vector(2,4,6),
        Vector(8,10,12),
        Vector(14,16,18)
    )
    R = M / 2
    assert R == Matrix(
        Vector(1,2,3),
        Vector(4,5,6),
        Vector(7,8,9)
    )

def test_MATRIX_SET_FROM_QUATERNION():
    vec  = Vector(1,1,1)
    vec_ = Vector().copy(vec)
    Q    = Quaternion(math.sqrt(2)/2, math.sqrt(2)/2, 0.0, 0.0)

    vec.applyQuaternion(Q)
    mat = Matrix().setFromQuaternion(Q)

    test = vec - (mat * vec_)

    assert test.mag() == pytest.approx(0.0)

#### QUATERNION
def test_QUATERNION_IMPL():
    #### REPR
    q = Quaternion(1, 2, 3, 4)
    assert str(q) == "Quaternion(1.0, 2.0, 3.0, 4.0)"

    #### GETITEM
    q = Quaternion(1, 2, 3, 4)
    assert q[0] == 1
    assert q[1] == 2
    assert q[2] == 3
    assert q[3] == 4
    with pytest.raises(IndexError):
        q[4]

    #### EQUALITY
    a = Quaternion(1, 2, 3, 4)
    b = Quaternion(1, 2, 3, 4)
    c = Quaternion(4, 3, 2, 1)
    assert a == b
    assert a != c
    assert a.__eq__(1) is NotImplemented

    #### ITER
    q = Quaternion(1, 2, 3, 4)
    assert len(q) == 4
    assert list(q) == [1, 2, 3, 4]

    #### MAG
    q = Quaternion(1, 2, 3, 4)
    assert q.mag2() == 30
    assert q.mag() == q.magnitude() == math.sqrt(30)

    #### NORMALIZE-ZERO
    q = Quaternion(0, 0, 0, 0)
    with pytest.raises(ZeroDivisionError, match="zero quaternion"):
        q.normalize()

    #### NORMALIZE
    q = Quaternion(1, 2, 3, 4)
    q.normalize()
    assert q.mag() == pytest.approx(1.0)

    #### COPY-SET
    a = Quaternion(1, 2, 3, 4)
    b = Quaternion(4, 5, 6, 7)
    tmp = Quaternion()
    tmp.copy(a)
    assert tmp == a
    tmp.set(b.w, b.x, b.y, b.z)
    assert tmp == b
    q = Quaternion()
    with pytest.raises(TypeError):
        q.copy(1)
    
    #### CONJUGATE
    q = Quaternion(1, 2, 3, 4)
    qc = q.conjugate()

    assert qc == Quaternion(1, -2, -3, -4)
    assert q == Quaternion(1, 2, 3, 4)

    #### __neg__
    q = Quaternion(1, 2, 3, 4)
    assert -q == Quaternion(-1, -2, -3, -4)

def test_QUATERNION_MATH():
    #### __add__ and __sub__
    a = Quaternion(1, 2, 3, 4)
    b = Quaternion(4, 3, 2, 1)
    assert a + b == Quaternion(5, 5, 5, 5)
    assert a - b == Quaternion(-3, -1, 1, 3)
    assert a == Quaternion(1, 2, 3, 4)
    assert b == Quaternion(4, 3, 2, 1)

    #### SCALAR __mul__
    q = Quaternion(1, 2, 3, 4)
    assert q * 2 == Quaternion(2, 4, 6, 8)
    assert 2 * q == Quaternion(2, 4, 6, 8)

    #### SCALAR __truediv__
    q = Quaternion(2, 4, 6, 8)
    assert q / 2 == Quaternion(1, 2, 3, 4)
    with pytest.raises(ZeroDivisionError):
        q / 0
    
    #### ADD INPLACE
    a = Quaternion(1, 2, 3, 4)
    b = Quaternion(4, 3, 2, 1)
    a.add(b)
    assert a == Quaternion(5, 5, 5, 5)

    #### SUB INPLACE
    a = Quaternion(5, 5, 5, 5)
    b = Quaternion(4, 3, 2, 1)
    a.sub(b)
    assert a == Quaternion(1, 2, 3, 4)

    #### SCALE INPLACE
    q = Quaternion(1, 2, 3, 4)
    q.scale(2)
    assert q == Quaternion(2, 4, 6, 8)
    q.scale(0.5)
    assert q == Quaternion(1, 2, 3, 4)

    #### ERRORs
    q = Quaternion(1, 2, 3, 4)
    with pytest.raises(TypeError):
        q.add(1)
    with pytest.raises(TypeError):
        q.sub("x")
    with pytest.raises(TypeError):
        q.scale(Vector(1, 2, 3))

    #### __mul__
    q = Quaternion(1, 2, 3, 4)
    I = Quaternion(1, 0, 0, 0)
    assert I * q == q
    assert q * I == q

    q = Quaternion(1, 2, 3, 4)
    qc = q.conjugate()
    p = q * qc
    assert p.w == pytest.approx(q.mag2())
    assert p.x == pytest.approx(0.0)
    assert p.y == pytest.approx(0.0)
    assert p.z == pytest.approx(0.0)

    #### MUL INPLACE
    q1 = Quaternion(0, 1, 0, 0)
    q2 = Quaternion(0, 0, 1, 0)
    tmp = Quaternion().copy(q1).mul(q2)
    ref = q1 * q2
    assert tmp == ref

def test_QUATERNION_ANGLETO():
    q = Quaternion().setFromAxisAngle(Vector(1,0,0),math.pi/2)
    assert Quaternion().angleTo(q) == pytest.approx(math.pi/2)

    q = Quaternion().setFromAxisAngle(Vector(0,1,0),math.pi/3)
    assert Quaternion().angleTo(q) == pytest.approx(math.pi/3)

    q = Quaternion().setFromAxisAngle(Vector(0,0,1),math.pi/4)
    assert Quaternion().angleTo(q) == pytest.approx(math.pi/4)

def test_QUATERNION_ROTATE_VECTOR():
    Q_BA = Quaternion(math.sqrt(2)/2, 0.0, 0.0, math.sqrt(2)/2)
    xA = Vector(1.0, 0.0, 0.0)
    yA = Vector(0.0, 1.0, 0.0)
    zA = Vector(0.0, 0.0, 1.0)

    assert (Q_BA.rotate(xA) - Vector(0.0, -1.0, 0.0)).mag() < ZERO
    assert (Q_BA.rotate(yA) - Vector(1.0, 0.0, 0.0)).mag() < ZERO
    assert (Q_BA.rotate(zA) - Vector(0.0, 0.0, 1.0)).mag() < ZERO

    assert ( (Q_BA * Quaternion(0, 1.0, 0.0, 0.0) * Q_BA.conjugate()).vec() - Vector(0.0,-1.0,0.0)).mag() < ZERO
    assert ( (Q_BA * Quaternion(0, 0.0, 1.0, 0.0) * Q_BA.conjugate()).vec() - Vector(1.0,0.0,0.0)).mag() < ZERO
    assert ( (Q_BA * Quaternion(0, 0.0, 0.0, 1.0) * Q_BA.conjugate()).vec() - Vector(0.0,0.0,1.0)).mag() < ZERO

def test_QUATERNION_SET_FROM_AXIS_ANGLE():
    Qx = Quaternion().setFromAxisAngle(Vector(1,0,0), math.pi/2)
    Qy = Quaternion().setFromAxisAngle(Vector(0,1,0), math.pi/2)
    Qz = Quaternion().setFromAxisAngle(Vector(0,0,1), math.pi/2)

    assert (Qx - Quaternion(math.sqrt(2)/2, math.sqrt(2)/2, 0, 0)).mag() < ZERO
    assert (Qy - Quaternion(math.sqrt(2)/2, 0, math.sqrt(2)/2, 0)).mag() < ZERO
    assert (Qz - Quaternion(math.sqrt(2)/2, 0, 0, math.sqrt(2)/2)).mag() < ZERO

def test_QUATERNION_SET_FROM_MATRIX():
    axis = Vector(1,0,0)
    vec  = Vector(1,1,1)
    vec_ = Vector().copy(vec)
    Q    = Quaternion().setFromAxisAngle(axis, math.pi/2)

    mat = Matrix().setFromQuaternion(Q)
    vec.applyQuaternion(Quaternion().setFromMatrix(mat))

    test = vec - (mat * vec_)

    assert test.mag() == pytest.approx(0.0)

def test_QUATERNION_PASSIVE_ROTATION_ORDER():
    #### quaternion passive transform order is from right to left
    #### similar with how matrices are multiplied 
    #### q_DA = q_DC * q_CB * q_BA   D<--A
    #### M_DA = M_DC * M*CB * M_BA   D<--A
    vec  = Vector(1,1,1)
    vec_ = Vector().copy(vec)
    
    q1 = Quaternion().setFromAxisAngle(Vector(1,0,0), math.pi/3)
    q2 = Quaternion().setFromAxisAngle(Vector(0,1,0), math.pi/3)
    q3 = Quaternion().setFromAxisAngle(Vector(0,0,1), math.pi/3)
    
    q = Quaternion().copy(q1).mul(q2).mul(q3)
    qr = q1 * q2 * q3

    assert (q - qr).mag() < ZERO

    vec.applyQuaternion(q)
    vec_.applyQuaternion(q3)
    vec_.applyQuaternion(q2)
    vec_.applyQuaternion(q1)
    
    M  = Matrix().setFromQuaternion(q)
    M1 = Matrix().setFromQuaternion(q1)
    M2 = Matrix().setFromQuaternion(q2)
    M3 = Matrix().setFromQuaternion(q3)
    Mr = M1 * M2 * M3

    assert (vec - vec_).mag() == pytest.approx(0.0)
    assert (vec - M * Vector(1,1,1)).mag() == pytest.approx(0.0)
    assert (vec - Mr * Vector(1,1,1)).mag() == pytest.approx(0.0)

def test_QUATERNION_DIFF_SOLVES_RIGHT_FACTOR():
    q_first = Quaternion().setFromAxisAngle(Vector(1, 0, 0), math.pi / 3)
    q_second = Quaternion().setFromAxisAngle(Vector(0, 0, 1), math.pi / 4)
    q_total = q_second * q_first

    q_out = q_total.diff(q_second)

    assert q_out.w == pytest.approx(q_first.w)
    assert q_out.x == pytest.approx(q_first.x)
    assert q_out.y == pytest.approx(q_first.y)
    assert q_out.z == pytest.approx(q_first.z)

def test_QUATERNION_DIFF2_SOLVES_LEFT_FACTOR():
    q_first = Quaternion().setFromAxisAngle(Vector(1, 0, 0), math.pi / 3)
    q_second = Quaternion().setFromAxisAngle(Vector(0, 0, 1), math.pi / 4)
    q_total = q_second * q_first

    q_out = q_total.diff2(q_first)

    assert q_out.w == pytest.approx(q_second.w)
    assert q_out.x == pytest.approx(q_second.x)
    assert q_out.y == pytest.approx(q_second.y)
    assert q_out.z == pytest.approx(q_second.z)

def test_QUATERNION_DIFF_TYPE_ERRORS():
    q = Quaternion()
    with pytest.raises(TypeError):
        q.diff(1)
    with pytest.raises(TypeError):
        q.diff2(1)

#### EULER

def test_EULER_SET_FROM_QUATERNION_IDENTITY():
    q = Quaternion(1, 0, 0, 0)
    e = Euler(order='XYZ').setFromQuaternion(q)

    assert e.x == pytest.approx(0.0)
    assert e.y == pytest.approx(0.0)
    assert e.z == pytest.approx(0.0)

def test_EULER_SET_FROM_QUATERNION_ERROR():
    #### INVALID Q
    e = Euler()
    with pytest.raises(TypeError):
        e.setFromQuaternion(1)

    #### NON UNIT Q
    e = Euler()
    q = Quaternion(2, 0, 0, 0)
    with pytest.raises(ValueError):
        e.setFromQuaternion(q)

def test_EULER_QUATERNION_ROUNDTRIP():
    e1 = Euler(0.2, -0.3, 0.4, 'ZYX')
    q = Quaternion().setFromEuler(e1)
    e2 = Euler(order='ZYX').setFromQuaternion(q)
    q2 = Quaternion().setFromEuler(e2)

    assert q2.w == pytest.approx(q.w, abs=1e-12)
    assert q2.x == pytest.approx(q.x, abs=1e-12)
    assert q2.y == pytest.approx(q.y, abs=1e-12)
    assert q2.z == pytest.approx(q.z, abs=1e-12)
    assert e1.x == pytest.approx(e2.x, abs=ZERO)
    assert e1.y == pytest.approx(e2.y, abs=ZERO)
    assert e1.z == pytest.approx(e2.z, abs=ZERO)

def test_EULER_APPLY_ROTATIONS_SEQUENCE():
    #### XYZ
    euler   = Euler(math.pi/2, math.pi/2, math.pi/2, order='XYZ')
    xaxis   = Vector(1, 0, 0)
    yaxis   = Vector(0, 1, 0)
    zaxis   = Vector(0, 0, 1)
    x1      = Vector(1, 1, 1)
    x2      = Vector().copy(x1)
    x3      = Vector().copy(x1)

    x1.applyQuaternion(Quaternion().setFromEuler(euler))

    #### sequenced rotations
    q  = Quaternion().setFromAxisAngle(zaxis, math.pi/2)
    qA = Quaternion().setFromAxisAngle(yaxis, math.pi/2)
    qB = Quaternion().setFromAxisAngle(xaxis, math.pi/2)
    q.mul(qA).mul(qB)
    x2.applyQuaternion(q)

    x3.applyEuler(euler)

    assert x1 == x2 == x3
    assert q.diff(Quaternion().setFromEuler(euler)) == Quaternion()
    assert q.angleTo(Quaternion().setFromEuler(euler)) == 0.0

    #### XZY
    euler   = Euler(math.pi/2, math.pi/2, math.pi/2, order='XZY')
    xaxis   = Vector(1, 0, 0)
    yaxis   = Vector(0, 1, 0)
    zaxis   = Vector(0, 0, 1)
    x1      = Vector(1, 1, 1)
    x2      = Vector().copy(x1)
    x3      = Vector().copy(x1)

    x1.applyQuaternion(Quaternion().setFromEuler(euler))

    #### sequenced rotations
    q  = Quaternion().setFromAxisAngle(yaxis, math.pi/2)
    qA = Quaternion().setFromAxisAngle(zaxis, math.pi/2)
    qB = Quaternion().setFromAxisAngle(xaxis, math.pi/2)
    q.mul(qA).mul(qB)
    x2.applyQuaternion(q)

    x3.applyEuler(euler)

    assert x1 == x2 == x3
    assert q.diff(Quaternion().setFromEuler(euler)) == Quaternion()
    assert q.angleTo(Quaternion().setFromEuler(euler)) == 0.0

    #### YXZ
    euler   = Euler(math.pi/2, math.pi/2, math.pi/2, order='YXZ')
    xaxis   = Vector(1, 0, 0)
    yaxis   = Vector(0, 1, 0)
    zaxis   = Vector(0, 0, 1)
    x1      = Vector(1, 1, 1)
    x2      = Vector().copy(x1)
    x3      = Vector().copy(x1)

    x1.applyQuaternion(Quaternion().setFromEuler(euler))

    #### sequenced rotations
    q  = Quaternion().setFromAxisAngle(zaxis, math.pi/2)
    qA = Quaternion().setFromAxisAngle(xaxis, math.pi/2)
    qB = Quaternion().setFromAxisAngle(yaxis, math.pi/2)
    q.mul(qA).mul(qB)
    x2.applyQuaternion(q)

    x3.applyEuler(euler)

    assert x1 == x2 == x3
    assert q.diff(Quaternion().setFromEuler(euler)) == Quaternion()
    assert q.angleTo(Quaternion().setFromEuler(euler)) == 0.0

    #### YZX
    euler   = Euler(math.pi/2, math.pi/2, math.pi/2, order='YZX')
    xaxis   = Vector(1, 0, 0)
    yaxis   = Vector(0, 1, 0)
    zaxis   = Vector(0, 0, 1)
    x1      = Vector(1, 1, 1)
    x2      = Vector().copy(x1)
    x3      = Vector().copy(x1)

    x1.applyQuaternion(Quaternion().setFromEuler(euler))

    #### sequenced rotations
    q  = Quaternion().setFromAxisAngle(xaxis, math.pi/2)
    qA = Quaternion().setFromAxisAngle(zaxis, math.pi/2)
    qB = Quaternion().setFromAxisAngle(yaxis, math.pi/2)
    q.mul(qA).mul(qB)
    x2.applyQuaternion(q)

    x3.applyEuler(euler)

    assert x1 == x2 == x3
    assert q.diff(Quaternion().setFromEuler(euler)) == Quaternion()
    assert q.angleTo(Quaternion().setFromEuler(euler)) == 0.0

    #### ZYX
    euler   = Euler(math.pi/2, math.pi/2, math.pi/2, order='ZYX')
    xaxis   = Vector(1, 0, 0)
    yaxis   = Vector(0, 1, 0)
    zaxis   = Vector(0, 0, 1)
    x1      = Vector(1, 1, 1)
    x2      = Vector().copy(x1)
    x3      = Vector().copy(x1)

    x1.applyQuaternion(Quaternion().setFromEuler(euler))

    #### sequenced rotations
    q  = Quaternion().setFromAxisAngle(xaxis, math.pi/2)
    qA = Quaternion().setFromAxisAngle(yaxis, math.pi/2)
    qB = Quaternion().setFromAxisAngle(zaxis, math.pi/2)
    q.mul(qA).mul(qB)
    x2.applyQuaternion(q)

    x3.applyEuler(euler)

    assert x1 == x2 == x3
    assert q.diff(Quaternion().setFromEuler(euler)) == Quaternion()
    assert q.angleTo(Quaternion().setFromEuler(euler)) == 0.0

    #### ZXY
    euler   = Euler(math.pi/2, math.pi/2, math.pi/2, order='ZXY')
    xaxis   = Vector(1, 0, 0)
    yaxis   = Vector(0, 1, 0)
    zaxis   = Vector(0, 0, 1)
    x1      = Vector(1, 1, 1)
    x2      = Vector().copy(x1)
    x3      = Vector().copy(x1)

    x1.applyQuaternion(Quaternion().setFromEuler(euler))

    #### sequenced rotations
    q  = Quaternion().setFromAxisAngle(yaxis, math.pi/2)
    qA = Quaternion().setFromAxisAngle(xaxis, math.pi/2)
    qB = Quaternion().setFromAxisAngle(zaxis, math.pi/2)
    q.mul(qA).mul(qB)
    x2.applyQuaternion(q)

    x3.applyEuler(euler)

    assert x1 == x2 == x3
    assert q.diff(Quaternion().setFromEuler(euler)) == Quaternion()
    assert q.angleTo(Quaternion().setFromEuler(euler)) == 0.0

#### STATE

def test_STATE_IMPL():
    #### CONSTRUCTOR
    s = State()
    assert s.mass   == 0.0
    assert s.pos    == Vector(0,0,0)
    assert s.vel    == Vector(0,0,0)
    assert s.quat   == Quaternion(1,0,0,0)
    assert s.omega  == Vector(0,0,0)

    #### input makes a copy not a reference
    p = Vector(1,2,3)
    s = State(position=p)
    p.set(0,0,0)
    assert s.pos == Vector(1,2,3)
    assert p == Vector(0,0,0)

    #### __getitem___
    s = State(1, Vector(1,2,3), Vector(4,5,6), Quaternion(1,0,0,0), Vector(7,8,9))
    assert s[0] == 1
    assert s[1] == Vector(1,2,3)
    assert s[2] == Vector(4,5,6)
    assert s[3] == Quaternion(1,0,0,0)
    assert s[4] == Vector(7,8,9)
    assert len(s) == 5
    with pytest.raises(IndexError):
        s[5]

    #### COPY METHOD, copy deeps and does not save reference
    a = State(1, Vector(1,2,3), Vector(4,5,6), Quaternion(1,0,0,0), Vector(7,8,9))
    b = State()
    b.copy(a)
    assert b == a
    a.scale(2)
    assert b != a

def test_STATE_MATH():
    #### ADD method, inplace 
    a = State(1, Vector(1,2,3), Vector(4,5,6), Quaternion(1,1,1,1), Vector(7,8,9))
    b = State(1, Vector(1,1,1), Vector(1,1,1), Quaternion(1,1,1,1), Vector(1,1,1))
    a.add(b)
    assert a.mass == 2
    assert a.pos == Vector(2,3,4)

    #### SUB method, inplace
    a = State(2, Vector(2,3,4), Vector(5,6,7), Quaternion(2,2,2,2), Vector(8,9,10))
    b = State(1, Vector(1,1,1), Vector(1,1,1), Quaternion(1,1,1,1), Vector(1,1,1))
    a.sub(b)
    assert a.mass == 1
    assert a.pos == Vector(1,2,3)

    #### SCALE method, inplace
    s = State(1, Vector(1,2,3), Vector(4,5,6), Quaternion(1,2,3,4), Vector(7,8,9))
    s.scale(2)
    assert s.mass == 2
    assert s.pos == Vector(2,4,6)

    #### ADD then SCALE method, inplace
    a = State(
        1.0,
        Vector(1,2,3),
        Vector(4,5,6),
        Quaternion(1,2,3,4),
        Vector(7,8,9)
    )
    b = State(
        1.0,
        Vector(1,1,1),
        Vector(1,1,1),
        Quaternion(1,1,1,1),
        Vector(1,1,1)
    )
    a.add_scaled(b, 2)
    assert a.mass == 3.0
    assert a.pos == Vector(3,4,5)

    #### ERRORs
    s = State()
    with pytest.raises(TypeError):
        s.add(1)
    with pytest.raises(TypeError):
        s.sub(1)
    with pytest.raises(TypeError):
        s.scale("x")
    with pytest.raises(TypeError):
        s.add_scaled(s, s)
    with pytest.raises(TypeError):
        s.add_scaled(1, 1)

    #### ZERO method
    a = State().zero()
    assert a.mass == 0.0
    assert a.pos.mag2() == 0.0
    assert a.vel.mag2() == 0.0
    assert a.quat.mag2() == 0.0
    assert a.omega.mag2() == 0.0

#### PLANET

def test_PLANET_IMPL():
    #### SETAS
    p = Planet().setAs("earth")

    assert p.mu == MU_EARTH_M
    assert p.radi == ER_EARTH_M
    assert p.time == 0.0
    assert p.getSpacecrafts() == {}

    #### ADD SPACECRAFT
    p.addSpacecraft("SC-1")

    sc = p.getSpacecrafts()["SC-1"]

    assert isinstance(sc, Spacecraft)
    assert sc.name == "SC-1"
    assert sc.planet is p

    #### GRAVITY
    s = State(
        mass=2.0,
        position=Vector(ER_EARTH_M, 0.0, 0.0),
        velocity=Vector(),
        quaternion=Quaternion(),
        bodyrate=Vector(),
    )

    g = p.gravity(s, 0.0)

    expected_mag = p.mu * s.mass / (ER_EARTH_M ** 2)

    assert g.x == pytest.approx(-expected_mag)
    assert g.y == pytest.approx(0.0)
    assert g.z == pytest.approx(0.0)

    #### INIT and RECORD
    p = Planet().setAs("earth")
    p.addSpacecraft("SC")
    sc = p.getSpacecrafts()["SC"]

    sc.setmass(1.0)
    sc.setposition(Vector(ER_EARTH_M + 400e3, 0, 0))
    sc.setvelocity(Vector(0, 1, 0))

    p.INIT()

    rec = sc.getRECORD()

    assert "Time" in rec
    assert "Position" in rec
    assert "Velocity" in rec
    assert "Quaternion" in rec
    assert "Bodyrate" in rec
    assert len(rec["Time"]) == 1
    assert rec["Time"][0] == 0.0

def test_PLANET_STEP_UPDATES_TIME_AND_RECORD():
    p = Planet()
    p.addSpacecraft("SC")
    sc = p.getSpacecrafts()["SC"]

    sc.setmass(1.0)
    sc.setposition(Vector(1, 0, 0))
    sc.setvelocity(Vector(1, 0, 0))
    sc.inertia = Matrix()

    def no_force(state, time):
        return Vector(0, 0, 0)

    sc.addFORCE(no_force, "ZERO")
    sc.initRECORD()

    p.step(0.5)

    assert p.time == pytest.approx(0.5)
    rec = sc.getRECORD()
    assert len(rec["Time"]) == 2
    assert rec["Time"][-1] == pytest.approx(0.5)

#### SPACECRAFT

def test_SPACECRAFT_IMPL():
    #### SETTERS
    sc = Spacecraft("SC")

    sc.setmass(10.0)
    sc.setposition(Vector(1, 2, 3))
    sc.setvelocity(Vector(4, 5, 6))
    sc.setbodyrate(Vector(180, 0, 0))   # deg/s
    sc.setquaternion(Quaternion(1, 0, 0, 0))

    assert sc.state.mass == 10.0
    assert sc.state.pos == Vector(1, 2, 3)
    assert sc.state.vel == Vector(4, 5, 6)
    assert sc.state.omega.x == pytest.approx(math.pi)
    assert sc.state.omega.y == pytest.approx(0.0)
    assert sc.state.omega.z == pytest.approx(0.0)
    assert sc.state.quat == Quaternion(1, 0, 0, 0)

    #### ERRORS

    with pytest.raises(TypeError):
        sc.setmass("x")

    with pytest.raises(TypeError):
        sc.setposition(1)

    with pytest.raises(TypeError):
        sc.setvelocity(1)

    with pytest.raises(TypeError):
        sc.setbodyrate(1)

    with pytest.raises(TypeError):
        sc.setquaternion(1)

def test_SPACECRAFT_FUNC():
    #### ADD TYPES
    sc = Spacecraft("SC")

    def constant_force(state, time):
        return Vector(1, 0, 0)
    
    def torque(state, time):
        return Vector(0, 1, 0)

    def momentum(state, time):
        return Vector(0, 0, 2)

    assert sc.addTORQUE(torque, "T") is True
    assert sc.addMOMENTUM(momentum, "H") is True
    assert sc.addFORCE(constant_force, "F") is True

    #### ADD FORCE INVALID ARGUMENT
    sc = Spacecraft("SC")

    def bad_force(state):
        return Vector(1, 0, 0)

    assert sc.addFORCE(bad_force, "BAD") is False

    #### ADD FORCE INVALID RETURN
    sc = Spacecraft("SC")

    def bad_force(state, time):
        return 123

    assert sc.addFORCE(bad_force, "BAD") is False

    #### OVERWRITE FUNC (prints)
    sc = Spacecraft("SC")

    def torque(state, time):
        return Vector(0, 1, 0)

    def momentum(state, time):
        return Vector(0, 0, 2)
    
    def force(state, time):
        return Vector(3, 0, 2)

    assert sc.addFORCE(force, "F") is True
    assert sc.addTORQUE(torque, "T") is True
    assert sc.addMOMENTUM(momentum, "H") is True
    assert sc.addFORCE(force, "F") is True
    assert sc.addTORQUE(torque, "T") is True
    assert sc.addMOMENTUM(momentum, "H") is True

    #### COMPUTE
    sc = Spacecraft("SC")
    sc.state.mass = 2.0
    sc.state.omega = Vector(1, 2, 3)
    sc.inertia = Matrix(
        Vector(2, 0, 0),
        Vector(0, 3, 0),
        Vector(0, 0, 4),
    )

    def f1(state, time):
        return Vector(1, 0, 0)

    def f2(state, time):
        return Vector(0, 2, 0)

    def t1(state, time):
        return Vector(0, 0, 3)

    def h1(state, time):
        return Vector(4, 5, 6)

    sc.addFORCE(f1, "F1")
    sc.addFORCE(f2, "F2")
    sc.addTORQUE(t1, "T1")
    sc.addMOMENTUM(h1, "H1")

    sc.computeEXTERNAL(sc.state, 0.0)

    assert sc.netFORCE == Vector(1, 2, 0)
    assert sc.netTORQUE == Vector(0, 0, 3)

    # inertia * omega = (2, 6, 12), plus external momentum (4,5,6)
    assert sc.netMOMENTUM == Vector(6, 11, 18)

def test_SPACECRAFT_CUSTOM():
    #### VALID
    sc = Spacecraft("SC")

    def constant_force(sc, state, time, args):
        return Vector(1, 0, 0)

    assert sc.addCUSTOM(constant_force, "F") is True

    #### INVALID ARGUMENT
    sc = Spacecraft("SC")

    def bad_force(state):
        return Vector(1, 0, 0)

    assert sc.addCUSTOM(bad_force, "BAD") is False

    #### INVALID RETURN
    sc = Spacecraft("SC")

    def bad_force(sc, state, time):
        return ""

    assert sc.addCUSTOM(bad_force, "BAD") is False

    #### COMPUTE
    sc = Spacecraft("SC")
    sc.state.mass = 2.0
    sc.state.omega = Vector(1, 2, 3)
    sc.inertia = Matrix(
        Vector(2, 0, 0),
        Vector(0, 3, 0),
        Vector(0, 0, 4),
    )

    def f1(sc, state, time, args):
        return Vector(1, 2, 3)

    def f2(sc, state, time, args):
        return Vector(state.mass, time, 0)

    sc.addCUSTOM(f1, "F1")
    sc.addCUSTOM(f2, "F2")

    sc.computeCUSTOM(sc.state, 0.0)

    assert sc.FUNC['F1'] == Vector(1,2,3)
    assert sc.FUNC['F2'] == Vector(sc.state.mass,0,0)

def test_SPACECRAFT_DERIVATIVE_TRANSLATION_ONLY():
    sc = Spacecraft("SC")
    sc.state.mass = 2.0
    sc.inertia = Matrix()  # identity

    state = State(
        mass=2.0,
        position=Vector(10, 20, 30),
        velocity=Vector(1, 2, 3),
        quaternion=Quaternion(1, 0, 0, 0),
        bodyrate=Vector(0, 0, 0),
    )
    dstate = State().zero()

    def force(state, time):
        return Vector(4, 0, 0)

    sc.addFORCE(force, "F")

    sc.derivative(state, 0.0, dstate)

    assert dstate.mass == 0.0
    assert dstate.pos == Vector(1, 2, 3)
    assert dstate.vel == Vector(2, 0, 0)
    assert dstate.quat == Quaternion(0, 0, 0, 0)
    assert dstate.omega == Vector(0, 0, 0)

def test_SPACECRAFT_RECORD_LENGTHS_STAY_ALIGNED():
    p = Planet().setAs("earth")
    p.addSpacecraft("SC")
    sc = p.getSpacecrafts()["SC"]

    r = ER_EARTH_M + 400e3
    v_circ = math.sqrt(p.mu / r)

    sc.setmass(1.0)
    sc.setposition(Vector(r, 0, 0))
    sc.setvelocity(Vector(0, v_circ, 0))
    sc.inertia = Matrix()

    p.INIT()
    p.step(1.0)
    p.step(1.0)

    rec = sc.getRECORD()
    n = len(rec["Time"])

    for key, value in rec.items():
        assert len(value) == n

#### SIMULATION

def test_SIMULATION_LEO_CONSERVATION():
    system = Planet().setAs('EARTH')
    system.addSpacecraft("MULA")

    spacecrafts = system.getSpacecrafts()

    MULA = spacecrafts["MULA"]

    MULA.setmass(155)
    MULA.setposition(Vector(-5.18435402e+06,4.67140733e+06,-3.33373976e+02))
    MULA.setvelocity(Vector(0.69496868e3,0.7719583e3,7.4857076e3))
    MULA.setbodyrate(Vector(10,10,10))
    MULA.inertia = Matrix(Vector(18.5,0.97,0.97),Vector(0.97,20.0,1.12),Vector(0.97,1.12,17.2))

    simulate(system, 1000, 1/8)

    assert MULA.checkConservation()

def test_SIMULATION_CIRCULAR_CONSERVATION():
    p = Planet().setAs("earth")
    p.addSpacecraft("SC")
    sc = p.getSpacecrafts()["SC"]

    r = ER_EARTH_M + 400e3
    v_circ = math.sqrt(p.mu / r)

    sc.setmass(1.0)
    sc.setposition(Vector(r, 0, 0))
    sc.setvelocity(Vector(0, v_circ, 0))
    sc.inertia = Matrix()

    simulate(p, 1000, 1/4)

    assert sc.checkConservation()

def test_EARTH_ORBIT_SMALL_STEP_SANITY():
    p = Planet().setAs("earth")
    p.addSpacecraft("SC")
    sc = p.getSpacecrafts()["SC"]

    r = ER_EARTH_M + 400e3
    v_circ = math.sqrt(p.mu / r)

    sc.setmass(1.0)
    sc.setposition(Vector(r, 0, 0))
    sc.setvelocity(Vector(0, v_circ, 0))
    sc.inertia = Matrix()

    p.INIT()
    p.step(1.0)

    # After a 1-second step, still near original radius and unit quaternion norm
    assert sc.state.pos.mag() == pytest.approx(r, rel=1e-4)
    assert sc.state.quat.mag() == pytest.approx(1.0, rel=1e-12)

def test_PLANET_DATETIME_BEFORE_AFTER():
    p = Planet().setAs("earth")
    p.addSpacecraft("SC")
    sc = p.getSpacecrafts()["SC"]

    r = ER_EARTH_M + 400e3
    v_circ = math.sqrt(p.mu / r)

    sc.setmass(1.0)
    sc.setposition(Vector(r, 0, 0))
    sc.setvelocity(Vector(0, v_circ, 0))
    sc.inertia = Matrix()

    p.setEpoch(2026, 1, 1, 12, 59, 59, 123456)
    assert p.getEpoch()                 == 1767272399.123456
    assert str(p.getEpochDatetime())    == "2026-01-01 12:59:59.123456"
    assert p.getCurrentUnix()           == 1767272399.123456
    assert str(p.getCurrentDatetime())  == "2026-01-01 12:59:59.123456"

    simulateProgress(p, 1_000)

    assert p.getEpoch()                 == 1767272399.123456
    assert str(p.getEpochDatetime())    == "2026-01-01 12:59:59.123456"
    assert p.getCurrentUnix()           == 1767272399.123456 + 1000
    assert str(p.getCurrentDatetime())  == "2026-01-01 13:16:39.123456"

def test_SPACECRAFT_COMPUTE_METHODS_ARE_SYNC():

    system = Planet().setAs('EARTH')
    system.addSpacecraft("MULA")

    spacecrafts = system.getSpacecrafts()

    MULA = spacecrafts["MULA"]

    MULA.setmass(155)
    MULA.setposition(Vector(-5.18435402e+06,4.67140733e+06,-3.33373976e+02))
    MULA.setvelocity(Vector(0.69496868e3,0.7719583e3,7.4857076e3))
    MULA.setbodyrate(Vector(10,10,10))
    MULA.inertia = Matrix(Vector(18.5,0.97,0.97),Vector(0.97,20.0,1.12),Vector(0.97,1.12,17.2))

    def Gyro(sc: Spacecraft, st: State, t, args):
        out = Vector().zero()
        out.copy(st.omega)
        return out

    MULA.addCUSTOM(Gyro, 'Gyroscope')

    simulate(system, 100, 1/8)
    df = pd.DataFrame.from_dict(MULA.getRECORD())
    GyroData        = [ item for item in df['Gyroscope'] ]
    Bodyrates       = [ item for item in df['Bodyrate'].values.tolist()[:] ]
    AngularMomentum = [ item for item in df['BodyAngularMomentum'] ]
    NetMoment       = [ item.mag() for item in df['NetMomentum'].values.tolist()[:] ]

    assert GyroData[0]  == Bodyrates[0]
    assert GyroData[1]  == Bodyrates[1]
    assert GyroData[-1] == Bodyrates[-1]
    assert NetMoment[0] == AngularMomentum[0]
    assert NetMoment[1] == AngularMomentum[1]
    assert NetMoment[-1] == AngularMomentum[-1]
