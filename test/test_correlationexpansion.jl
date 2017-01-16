using Base.Test
using QuantumOptics

mask = correlationexpansion.indices2mask(3, [1,2])
@test mask == (true, true, false)
indices = correlationexpansion.mask2indices(mask)
@test indices == [1,2]

S1 = correlationexpansion.correlationmasks(3, 1)
S2 = correlationexpansion.correlationmasks(3, 2)
S3 = correlationexpansion.correlationmasks(3, 3)
@test S2 == Set([(true, true, false), (true, false, true), (false, true, true)])
@test S3 == Set([(true, true, true)])

b1 = FockBasis(2)
b2 = SpinBasis(1//2)
b3 = NLevelBasis(4)
b = tensor(b1, b2, b3)

rho = correlationexpansion.ApproximateOperator(b, b, S2 ∪ S3)
for s in S1
    @test s ∈ keys(rho.operators[1])
end

psi1 = normalize(fockstate(b1, 1))
psi2 = normalize(spinup(b2))
psi3 = normalize(nlevelstate(b3, 1))
psi = psi1 ⊗ psi2 ⊗ psi3

rho1 = psi1 ⊗ dagger(psi1)
rho2 = psi2 ⊗ dagger(psi2)
rho3 = psi3 ⊗ dagger(psi3)
rho = psi ⊗ dagger(psi)

x = correlationexpansion.ApproximateOperator(rho, S2)
s1 = (true, false, false)
# x1 = x.operators[1][s1]
# s2 = (false, true, false)
# x2 = x.operators[1][s2]
# s3 = (false, false, true)
# x3 = x.operators[1][s3]

# @test_approx_eq_eps 0. tracedistance(rho1, x1) 1e-10
# @test_approx_eq_eps 0. tracedistance(rho2, x2) 1e-10
# @test_approx_eq_eps 0. tracedistance(rho3, x3) 1e-10

function f(x)
    x = map(Int32, ceil(real(x)))
    for i=1:size(x,1)
        for j=1:size(x,2)
            print(x[i,j], " ")
        end
        println()
    end
    println()
end

# 3 2 4

# A = (rho2 ⊗ rho3 ⊗ rho1).data
# A = reshape(A, 3, 4, 2, 3, 4, 2)
# A = ipermutedims(A, [3, 1, 2, 6, 4, 5])
# A = reshape(A, 24, 24)

# B = (rho1 ⊗ rho2 ⊗ rho3).data
# f(A)
# f(B)


# rho_ = correlationexpansion.correlationoperator(x, s1)
rho_ = correlationexpansion.full(x)
# println(rho.basis_l.shape, rho.basis_r.shape)
# println(tracedistance(rho_, rho))

f(rho.data)
println()
f(rho_.data)