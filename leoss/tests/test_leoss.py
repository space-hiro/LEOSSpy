from leoss import *
import pytest

def test_VECTOR_REPR_STR():
    v = Vector(1,2,3)
    assert str(v) == "Vector(1, 2, 3)"

def test_VECTOR_GETITEM():
    v = Vector(1,2,3)

    assert v[0] == 1
    assert v[1] == 2
    assert v[2] == 3

    with pytest.raises(IndexError):
        v[3]

def test_VECTOR_EQUALITY():
    a = Vector(1,2,3)
    b = Vector(1,2,3)
    c = Vector(4,5,6)

    assert a == b
    assert a != c
    assert a.__eq__(1) is NotImplemented

def test_VECTOR_MAGNITUDE():
    v = Vector(1,2,3)

    assert v.mag() == v.magnitude() == math.sqrt(14)
    assert v.mag2() == v.mag() * v.mag() == 14

def test_VECTOR_SUM():
    v = Vector(1,2,3)

    assert v.sum() == 6
    assert v.sum() == v.x + v.y + v.z

def test_VECTOR_LEN():
    v = Vector(1,2,3)

    assert len(v) == 3

def test_VECTOR_ITER():
    v = Vector(1,2,3)

    a, b, c = v

    assert a == v.x
    assert b == v.y
    assert c == v.z

    assert list(v) == [v.x, v.y, v.z]

def test_VECTOR_COPY_SET():
    a = Vector(1,2,3)
    b = Vector(4,5,6)

    tmp = Vector()

    tmp.copy(a)
    assert tmp == a

    tmp.set(b.x, b.y, b.z)
    assert tmp == b

def test_VECTOR_NORMALIZE_ZERO():
    v = Vector()

    with pytest.raises(ZeroDivisionError, match="zero vector"):
        v.normalize()

def test_VECTOR_ADD_SUB():
    a = Vector(1,2,3)
    b = Vector(1,2,3)

    assert a + b == Vector(2,4,6)
    assert a - b == Vector(0,0,0)

    assert a == b

def test_VECTOR_SCALAR_MULTIPLY():
    a = Vector(1,2,3)

    assert a * 2 == Vector(2,4,6)
    assert 2 * a == Vector(2,4,6)

def test_VECTOR_DIVIDE():
    a = Vector(1,2,3)

    assert a / 2 == Vector(0.5,1,1.5)

    with pytest.raises(ZeroDivisionError):
        a / 0

def test_VECTOR_NEGATE_OPERATOR():
    a = Vector(1,2,3)

    assert -a == Vector(-1,-2,-3)   

def test_VECTOR_ADD_INPLACE():
    a = Vector(1,2,3)
    b = Vector(1,2,3)

    a.add(b)

    assert a == Vector(2,4,6)
    assert a != b

def test_VECTOR_SUB_INPLACE():
    a = Vector(2,4,6)
    b = Vector(1,2,3)

    a.sub(b)

    assert a == Vector(1,2,3)

def test_VECTOR_SCALE():
    a = Vector(1,2,3)

    a.scale(2)
    assert a == Vector(2,4,6)

    a.scale(0.5)
    assert a == Vector(1,2,3)

def test_VECTOR_NEGATE_METHOD():
    a = Vector(1,2,3)

    a.negate()
    assert a == Vector(-1,-2,-3)

    a.negate()
    assert a == Vector(1,2,3)

def test_VECTOR_NORMALIZE():
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

def test_VECTOR_DOT_BASIS():
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

def test_VECTOR_DOT_GENERAL():
    a = Vector(1,2,3)
    b = Vector(4,5,6)

    assert a.dot(b) == 32
    assert b.dot(a) == 32

def test_VECTOR_DOT_NO_MUTATION():
    a = Vector(1,2,3)
    b = Vector(4,5,6)

    _ = a.dot(b)

    assert a == Vector(1,2,3)
    assert b == Vector(4,5,6)

def test_VECTOR_PRODUCTS_TYPE_ERRORS():
    X = Vector(2,0,0)

    with pytest.raises(TypeError):
        X.dot(1)

    with pytest.raises(TypeError):
        X.cross(1)

def test_VECTOR_BASIS_NORMALIZATION():
    X = Vector(2,0,0)
    Y = Vector(0,2,0)
    Z = Vector(0,0,2)

    X.normalize()
    Y.normalize()
    Z.normalize()

    assert X.mag() == pytest.approx(1.0)
    assert Y.mag() == pytest.approx(1.0)
    assert Z.mag() == pytest.approx(1.0)

    assert X != Y
    assert Y != Z
    assert X != Z

def test_VECTOR_CROSS_BASIS_RIGHT_HAND_RULE():
    X = Vector(1,0,0)
    Y = Vector(0,1,0)
    Z = Vector(0,0,1)
    tmp = Vector()

    assert tmp.copy(X).cross(Y) == Z
    assert tmp.copy(Y).cross(Z) == X
    assert tmp.copy(Z).cross(X) == Y

def test_VECTOR_CROSS_BASIS_ANTI_COMMUTATIVE():
    X = Vector(1,0,0)
    Y = Vector(0,1,0)
    Z = Vector(0,0,1)
    tmp = Vector()

    assert tmp.copy(Y).cross(X) == -Z
    assert tmp.copy(Z).cross(Y) == -X
    assert tmp.copy(X).cross(Z) == -Y

def test_VECTOR_CROSS_SELF_ZERO():
    X = Vector(1,0,0)
    tmp = Vector()

    assert tmp.copy(X).cross(X) == Vector(0,0,0)

def test_VECTOR_CROSS_GENERAL():
    a = Vector(1,2,3)
    b = Vector(4,5,6)
    tmp = Vector()

    c = tmp.copy(a).cross(b)

    assert c == Vector(-3,6,-3)

def test_VECTOR_CROSS_ORTHOGONALITY():
    a = Vector(1,2,3)
    b = Vector(4,5,6)
    tmp = Vector()

    c = tmp.copy(a).cross(b)

    assert c.dot(a) == pytest.approx(0.0)
    assert c.dot(b) == pytest.approx(0.0)

def test_VECTOR_CROSS_DOES_NOT_MUTATE_OTHER():
    a = Vector(1,2,3)
    b = Vector(4,5,6)
    tmp = Vector()

    _ = tmp.copy(a).cross(b)

    assert b == Vector(4,5,6)

def test_MATRIX_CONSTRUCTOR():
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

def test_MATRIX_TYPECHECK():
    with pytest.raises(TypeError):
        Matrix(x=1)

    with pytest.raises(TypeError):
        Matrix(y=1)

    with pytest.raises(TypeError):
        Matrix(z=1)

def test_MATRIX_VECTOR_MULTIPLY():
    M = Matrix(
        Vector(1,0,0),
        Vector(0,2,0),
        Vector(0,0,3)
    )

    v = Vector(1,2,3)

    result = M * v

    assert result == Vector(1,4,9)

def test_MATRIX_MATRIX_MULTIPLY():
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

def test_MATRIX_SCALAR_MULTIPLY():
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

def test_MATRIX_DIVIDE():
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

def test_MATRIX_ROW_COL():
    M = Matrix(
        Vector(1,2,3),
        Vector(4,5,6),
        Vector(7,8,9)
    )

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

def test_MATRIX_TRANSPOSE():
    M = Matrix(
        Vector(1,2,3),
        Vector(4,5,6),
        Vector(7,8,9)
    )

    T = M.transpose()

    assert T.row(0) == Vector(1,2,3)
    assert T.row(1) == Vector(4,5,6)
    assert T.row(2) == Vector(7,8,9)

def test_MATRIX_TRACE():
    M = Matrix(
        Vector(1,2,3),
        Vector(4,5,6),
        Vector(7,8,9)
    )

    assert M.trace() == 1 + 5 + 9

def test_MATRIX_COPY():
    A = Matrix(
        Vector(1,2,3),
        Vector(4,5,6),
        Vector(7,8,9)
    )

    B = Matrix()
    B.copy(A)

    assert B == A

    with pytest.raises(TypeError):
        B.copy(1)

def test_MATRIX_INVERSE_IDENTITY():
    I = Matrix()

    inv = I.inverse()

    assert inv == I

def test_MATRIX_INVERSE():
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

def test_MATRIX_SINGULAR():
    M = Matrix(
        Vector(1,2,3),
        Vector(2,4,6),
        Vector(3,6,9)
    )

    with pytest.raises(ZeroDivisionError):
        M.inverse()