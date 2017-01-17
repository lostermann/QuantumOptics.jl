module correlationexpansion

using Combinatorics
using ..bases
# using ..states
using ..operators
# using ..operators_lazy
# using ..ode_dopri

# import Base: *, full
# import ..operators

typealias CorrelationMask{N} NTuple{N, Bool}

indices2mask(N::Int, indices::Vector{Int}) = CorrelationMask(tuple([(i in indices) for i=1:N]...))
mask2indices{N}(mask::CorrelationMask{N}) = Int[i for i=1:N if mask[i]]

complement(N::Int, indices::Vector{Int}) = Int[i for i=1:N if i ∉ indices]
complement{N}(mask::CorrelationMask{N}) = tuple([! x for x in mask]...)

correlationindices(N::Int, order::Int) = Set(combinations(1:N, order))
correlationmasks(N::Int, order::Int) = Set(indices2mask(N, indices) for indices in correlationindices(N, order))
correlationmasks{N}(S::Set{CorrelationMask{N}}, order::Int) = Set(s for s in S if sum(s)==order)

"""
An operator using only certain correlations.

It stores all subsystem density operators
:math:`\\rho^{(\\alpha)}` and correlation operators :math:`\\rho^{s_n}`
in the *operators* field. The layout of this field is the following:

* operators: (D_1, ..., D_N)
* D_n: Dict(s_n->rho^{s_n})
* s_n are CorrelationMask where exactly n subsystems are included.
"""
type ApproximateOperator{N} <: Operator
    basis_l::CompositeBasis
    basis_r::CompositeBasis
    operators::NTuple{N, Operator}
    correlations::Dict{CorrelationMask{N}, Operator}

    function ApproximateOperator{N}(basis_l::CompositeBasis, basis_r::CompositeBasis,
                operators::NTuple{N, DenseOperator},
                correlations::Dict{CorrelationMask{N}, DenseOperator})
        @assert N == length(basis_l.bases) == length(basis_r.bases)
        for i=1:N
            @assert operators[i].basis_l == basis_l.bases[i]
            @assert operators[i].basis_r == basis_r.bases[i]
        end
        for (mask, op) in correlations
            @assert sum(mask) > 1
            @assert op.basis_l == tensor(basis_l.bases[[mask...]]...)
            @assert op.basis_r == tensor(basis_r.bases[[mask...]]...)
        end
        new(basis_l, basis_r, operators, correlations)
    end
end

function ApproximateOperator{N}(basis_l::CompositeBasis, basis_r::CompositeBasis, S::Set{CorrelationMask{N}})
    operators = ([DenseOperator(basis_l.bases[i], basis_r.bases[i]) for i=1:N]...)
    correlations = Dict{CorrelationMask{N}, DenseOperator}()
    for mask in S
        @assert sum(mask) > 1
        correlations[mask] = tensor(operators[[mask...]]...)
    end
    ApproximateOperator{N}(basis_l, basis_r, operators, correlations)
end




ApproximateOperator{N}(basis::CompositeBasis, S::Set{CorrelationMask{N}}) = ApproximateOperator(basis, basis, S)

maskdiff{N}(x::CorrelationMask{N}, y::CorrelationMask{N}) = ([x[i] && !y[i] for i =1:N]...)


function ApproximateOperator{N}(rho::DenseOperator, S::Set{CorrelationMask{N}})
    operators = ([ptrace(rho, complement(N, [i])) for i=1:N]...)
    correlations = Dict{CorrelationMask{N}, DenseOperator}()
    for k=2:N
        for s_k in correlationmasks(S, k)
            σ_sk = ptrace(rho, mask2indices(complement(s_k)))
            σ_sk -= tensor(operators[[s_k...]]...)
            for s_n in keys(correlations)
                if mask2indices(s_n) ⊆ mask2indices(s_k)
                    println("s_n: ", mask2indices(s_n))
                    println("s_k: ", mask2indices(s_k))
                    s_x = maskdiff(complement(s_n), complement(s_k))
                    println("s_x: ", mask2indices(s_x))
                    ρ_sx = tensor(operators[[s_x...]]...)
                    σ_sn = correlations[s_n]
                    op = σ_sn ⊗ ρ_sx  # subsystems in wrong order
                    perm = sortperm([mask2indices(s_n); mask2indices(s_x)])
                    println("perm: ", perm)
                    σ_sk -= permutesystems(op, perm)
                end
            end
            correlations[s_k] = σ_sk
        end
    end
    ApproximateOperator{N}(rho.basis_l, rho.basis_r, operators, correlations)
end

# function productoperator{N}(op::ApproximateOperator{N})
#     tensor([op.operators[1][indices2mask(N, [i])] for i=1:N]...)
# end

# function correlationoperator{N}(op::ApproximateOperator{N}, s::CorrelationMask{N})
#     indices = mask2indices(s)
#     # println("indices: ", indices)
#     complement_indices = mask2indices(complement(s))
#     # println("compl indices: ", complement_indices)
#     op_compl = tensor([op.operators[1][indices2mask(N, [i])] for i=complement_indices]...)
#     x = (op_compl ⊗ op.operators[sum(s)][s])
#     # println("x.basis_l shape: ", x.basis_l.shape)
#     # println("x.basis_r shape: ", x.basis_r.shape)
#     # println("reshape: ", [reverse(x.basis_l.shape); reverse(x.basis_r.shape)])
#     data = reshape(x.data, [reverse(x.basis_l.shape); reverse(x.basis_r.shape)]...)
#     # println(size(data))
#     perm = [complement_indices; indices; complement_indices+N; indices+N]
#     # println("permutation", perm)
#     data = permutedims(data, [complement_indices; indices; complement_indices+N; indices+N])
#     # println(size(data))
#     DenseOperator(op.basis_l, op.basis_r, reshape(data, length(op.basis_l), length(op.basis_r)))
# end

# function full{N}(op::ApproximateOperator{N})
#     result = DenseOperator(op.basis_l, op.basis_r)
#     for n=2:N
#         for s in keys(op.operators[n])
#             result += correlationoperator(op, s)
#         end
#     end
#     result
# end

end