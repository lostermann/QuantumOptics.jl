using Base.Test
using QuantumOptics

srand(0)

b1 = NLevelBasis(4)
b2 = SpinBasis(1//2)
b3 = FockBasis(2)

b = b1⊗b2⊗b3

function test_op_equal(op1, op2)
    @test_approx_eq_eps 0. tracedistance_general(full(op1), full(op2)) 1e-11
end


# Test general rules for dense operators
# ======================================

op1 = DenseOperator(b1, rand(Complex128, length(b1), length(b1)))
op2 = DenseOperator(b2, rand(Complex128, length(b2), length(b2)))
op3 = DenseOperator(b3, rand(Complex128, length(b3), length(b3)))
op123 = op1 ⊗ op2 ⊗ op3
I2 = identityoperator(DenseOperator, b2)

op1I3 = 0.3*op1 ⊗ I2 ⊗ op3
op1I3_ = 0.3*LazyTensor(b, [1,3], [op1,op3])

test_op_equal(ptrace(op1I3, 3), ptrace(op1I3_, 3))
test_op_equal(ptrace(op1I3, 2), ptrace(op1I3_, 2))
test_op_equal(ptrace(op1I3, 1), ptrace(op1I3_, 1))


test_op_equal(op1⊗op2*trace(op3), ptrace(op123, 3))
test_op_equal(op1⊗op3*trace(op2), ptrace(op123, 2))
test_op_equal(op2⊗op3*trace(op1), ptrace(op123, 1))

test_op_equal(op1*trace(op2)*trace(op3), ptrace(op123, [2,3]))
test_op_equal(op2*trace(op1)*trace(op3), ptrace(op123, [1,3]))
test_op_equal(op3*trace(op1)*trace(op2), ptrace(op123, [1,2]))

@test_approx_eq_eps 0. abs(trace(op1)*trace(op2)*trace(op3) - ptrace(op123, [1,2,3])) 1e-15


# Compare partial traces of other operators to dense operators
# ============================================================

# SparseOperator
op = DenseOperator(b, rand(Complex128, length(b), length(b)))

test_op_equal(ptrace(sparse(op), 3), ptrace(op, 3))
test_op_equal(ptrace(sparse(op), 2), ptrace(op, 2))
test_op_equal(ptrace(sparse(op), 1), ptrace(op, 1))

test_op_equal(ptrace(sparse(op), [2,3]), ptrace(op, [2,3]))
test_op_equal(ptrace(sparse(op), [1,3]), ptrace(op, [1,3]))
test_op_equal(ptrace(sparse(op), [1,2]), ptrace(op, [1,2]))

@test ptrace(sparse(op), [1,2,3]) == ptrace(op, [1,2,3])

# LazyTensor
op123_ = LazyTensor(b, Dict(zip([1, 2, 3], [op1, op2, op3])))

test_op_equal(ptrace(op123, 3), ptrace(op123_, 3))
test_op_equal(ptrace(op123, 2), ptrace(op123_, 2))
test_op_equal(ptrace(op123, 1), ptrace(op123_, 1))

test_op_equal(ptrace(op123, [2,3]), ptrace(op123_, [2,3]))
test_op_equal(ptrace(op123, [1,3]), ptrace(op123_, [1,3]))
test_op_equal(ptrace(op123, [1,2]), ptrace(op123_, [1,2]))

@test_approx_eq_eps 0. abs(trace(op1)*trace(op2)*trace(op3) - ptrace(op123_, [1,2,3])) 1e-15

I2 = identityoperator(DenseOperator, b2)
op1I3 = 0.3*op1 ⊗ I2 ⊗ op3
op1I3_ = 0.3*LazyTensor(b, [1,3], [op1,op3])

test_op_equal(ptrace(op1I3, 3), ptrace(op1I3_, 3))
test_op_equal(ptrace(op1I3, 2), ptrace(op1I3_, 2))
test_op_equal(ptrace(op1I3, 1), ptrace(op1I3_, 1))

test_op_equal(ptrace(op1I3, [2,3]), ptrace(op1I3_, [2,3]))
test_op_equal(ptrace(op1I3, [1,3]), ptrace(op1I3_, [1,3]))
test_op_equal(ptrace(op1I3, [1,2]), ptrace(op1I3_, [1,2]))

@test_approx_eq_eps 0. abs(0.3*trace(op1)*trace(I2)*trace(op3) - ptrace(op1I3_, [1,2,3])) 1e-14

# Lazy Sum
op1 = DenseOperator(b, rand(Complex128, length(b), length(b)))
op2 = DenseOperator(b, rand(Complex128, length(b), length(b)))
op3 = DenseOperator(b, rand(Complex128, length(b), length(b)))

op123 = 0.2*op1 + 0.5*op2 + 0.7*op3
op123_ = LazySum([0.2, 0.5, 0.7], [op1, op2, op3])

test_op_equal(ptrace(op123, 3), ptrace(op123_, 3))
test_op_equal(ptrace(op123, 2), ptrace(op123_, 2))
test_op_equal(ptrace(op123, 1), ptrace(op123_, 1))

test_op_equal(ptrace(op123, [2,3]), ptrace(op123_, [2,3]))
test_op_equal(ptrace(op123, [1,3]), ptrace(op123_, [1,3]))
test_op_equal(ptrace(op123, [1,2]), ptrace(op123_, [1,2]))

@test_approx_eq_eps 0. abs(trace(op123) - ptrace(op123_, [1,2,3])) 1e-14
