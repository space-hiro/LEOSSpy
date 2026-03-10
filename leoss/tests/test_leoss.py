from leoss import *
import pytest

def test_CLASS_VECTOR():
    a = Vector(1,2,3)
    b = Vector(1,2,3)
    c = Vector(4,5,6)
    
    ## repr and str check
    assert str(a) == "Vector(1, 2, 3)"

    ## getitem check
    assert a.x == a[0] == 1
    assert a.y == a[1] == 2
    assert a.z == a[2] == 3
    with pytest.raises(IndexError):
        a[3]

    ## equal vectors check
    assert a == b
    assert a != c
    assert a.__eq__(1) is NotImplemented

    ## magnitude check
    assert a.mag() == a.magnitude() == math.sqrt(14)
    assert a.mag2() == a.mag()*a.mag() == 14

    ## sum check
    assert a.sum() == 1+2+3 == a.x + a.y + a.z

    ## len check
    assert len(a) == 3

    ## iter check, can also use for loop over Vector()
    a1, a2, a3 = a
    assert a1 == a.x
    assert a2 == a.y
    assert a3 == a.z
    assert list(a) == [a1, a2, a3]
    
    tmp = Vector()
    ## normalize zero check
    with pytest.raises(ZeroDivisionError, match="zero vector"):
        tmp.normalize()

    ## copy check
    tmp.copy(a)
    assert tmp == a == b
    assert tmp != c

    ## set check
    tmp.set(c.x, c.y, c.z)
    assert tmp != a
    assert tmp != b
    assert tmp == c

    ## __add__ check, this does not overwrite 
    assert a + b == Vector(2,4,6)
    assert a == b

    ## __sub__ check, this does not overwrite
    assert a - b == Vector(0,0,0)
    assert a == b

    ## __mul__ check, this does not overwrite
    assert a * 2 == Vector(2,4,6)
    assert a == b

    ## __rmul__ check, this does not overwrite
    assert 2 * a == Vector(2,4,6)
    assert a == b

    ## __truediv__ check, this does not overwrite
    assert a / 2 == Vector(0.5,1,1.5)
    assert a == b
    with pytest.raises(ZeroDivisionError):
        a / 0

    ## __neg__ check, this does not overwrite
    assert -a == Vector(-1,-2,-3)
    assert a == b

    ## add() check, this overwrites vector
    assert a.add(b) == Vector(2,4,6)
    assert a != b

    ## sub() check, this overwrites vector
    assert a.sub(b) == Vector(1,2,3)
    assert a == b

    ## scale() check, this overwrites vector
    assert a.scale(2) == Vector(2,4,6)
    assert a != b
    assert a.scale(0.5) == Vector(1,2,3)
    assert a == b

    ## negate() check, this overwrites vector
    assert a.negate() == Vector(-1,-2,-3)
    assert a != b
    assert a.negate() == b

    ## normalize() check, this overwrites vector
    assert a.normalize() == Vector(1/math.sqrt(14),2/math.sqrt(14),3/math.sqrt(14))
    assert a != b
    assert a.mag() == 1

    ## type error checks
    with pytest.raises(TypeError):
        a.add(1)

    with pytest.raises(TypeError):
        a.sub("x")

    with pytest.raises(TypeError):
        a.scale(Vector(1,2,3))

    with pytest.raises(TypeError):
        tmp.copy(5)

def test_CLASS_VECTORS_PRODUCTS():
    X = Vector(2,0,0)
    Y = Vector(0,2,0)
    Z = Vector(0,0,2)
    tmp = Vector()

    # dot: basis vectors
    assert X.dot(X) == 4
    assert X.dot(Y) == 0
    assert X.dot(Z) == 0
    assert Y.dot(X) == 0
    assert Y.dot(Y) == 4
    assert Y.dot(Z) == 0
    assert Z.dot(X) == 0
    assert Z.dot(Y) == 0
    assert Z.dot(Z) == 4

    # dot: general case
    a = Vector(1,2,3)
    b = Vector(4,5,6)
    assert a.dot(b) == 32
    assert b.dot(a) == 32

    # dot should not mutate
    assert a == Vector(1,2,3)
    assert b == Vector(4,5,6)

    # type checks
    with pytest.raises(TypeError):
        X.dot(1)
    with pytest.raises(TypeError):
        X.cross(1)

    X.normalize()
    Y.normalize()
    Z.normalize()

    assert X.mag() == pytest.approx(1.0)
    assert Y.mag() == pytest.approx(1.0)
    assert Z.mag() == pytest.approx(1.0)

    assert X != Y
    assert Y != Z
    assert X != Z

    # basis-vector cross products
    assert tmp.copy(X).cross(Y) == Z
    assert tmp.copy(Y).cross(Z) == X
    assert tmp.copy(Z).cross(X) == Y
    assert tmp.copy(Y).cross(X) == -Z
    assert tmp.copy(Z).cross(Y) == -X
    assert tmp.copy(X).cross(Z) == -Y

    # self cross
    assert tmp.copy(X).cross(X) == Vector(0,0,0)

    # general cross product
    a = Vector(1,2,3)
    b = Vector(4,5,6)
    c = tmp.copy(a).cross(b)
    assert c == Vector(-3,6,-3)

    # cross result orthogonal to both inputs
    assert c.dot(a) == pytest.approx(0.0)
    assert c.dot(b) == pytest.approx(0.0)

    # cross should not mutate other
    assert b == Vector(4,5,6)