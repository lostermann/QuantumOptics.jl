using Base.Test
using QuantumOptics

Nmin = 2
Nmax = 4
N = Nmax - Nmin + 1
basis = FockBasis(Nmin, Nmax)
bra = Bra(basis)
ket = Ket(basis)

@test_throws DimensionMismatch Bra(basis, [1, 2])
@test_approx_eq 0. norm(bra-Bra(basis, zeros(Int, N)))
@test_approx_eq 0. norm(ket-Ket(basis, zeros(Int, N)))
@test_approx_eq 0. bra*ket
@test_throws bases.IncompatibleBases bra*Ket(FockBasis(Nmin, Nmax+1))


bra = Bra(basis, [1im, 0, 1])
ket = Ket(basis, [0, -1im, 1])

@test_approx_eq 0. norm(5*bra - Bra(basis, [5im, 0, 5]))
@test_approx_eq 0. norm(5*ket - Ket(basis, [0, -5im, 5]))
@test_approx_eq 0. norm(5*ket - ket/0.2)
@test_throws bases.IncompatibleBases bra + Bra(FockBasis(Nmin, Nmax+1))
@test_throws bases.IncompatibleBases ket + Ket(FockBasis(Nmin, Nmax+1))

# Norm
basis = FockBasis(0, 1)
bra = Bra(basis, [3im, -4])
ket = Ket(basis, [-4im, 3])
@test_approx_eq 5. norm(bra)
@test_approx_eq 5. norm(ket)

ubra = normalize(bra)
uket = normalize(ket)
@test_approx_eq 5. norm(bra)
@test_approx_eq 5. norm(ket)
@test_approx_eq 1. norm(ubra)
@test_approx_eq 1. norm(uket)

bra_ = normalize!(bra)
ket_ = normalize!(ket)
@test bra_ === bra
@test ket_ === ket
@test_approx_eq 1. norm(bra)
@test_approx_eq 1. norm(ket)


# Partial Trace
basis = FockBasis(0, 1)
a = normalize(Bra(basis, [1im, 0]))
b = normalize(Bra(basis, [1, 2]))
c = normalize(Bra(CompositeBasis(basis, basis), [1im, 2im, 0, 0]))
@test_approx_eq 0. norm(tensor(a, b) - c)
@test_approx_eq_eps 0. tracedistance(operators.ptrace(c, 1), tensor(dagger(b), b)) 1e-15
@test_approx_eq_eps 0. tracedistance(operators.ptrace(c, 2), tensor(dagger(a), a)) 1e-15


# Test permutating systems
b1 = NLevelBasis(2)
b2 = SpinBasis(1//2)
b3 = FockBasis(2)

srand(0)
psi1 = normalize(Ket(b1, rand(Complex128, length(b1))))
psi2 = normalize(Ket(b2, rand(Complex128, length(b2))))
psi3 = normalize(Ket(b3, rand(Complex128, length(b3))))

psi123 = psi1 ⊗ psi2 ⊗ psi3
psi213 = psi2 ⊗ psi1 ⊗ psi3

c = dagger(psi213)*permutesystems(psi123, [2,1,3])

@test_approx_eq_eps 1. c 1e-5
