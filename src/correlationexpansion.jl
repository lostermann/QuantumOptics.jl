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

correlationindices(N, order) = Set(combinations(1:N, order))
correlationmasks(N, order) = Set(indices2mask(N, indices) for indices in correlationindices(N, order))

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
    operators::NTuple{N, Dict{CorrelationMask{N}, Operator}}

    function ApproximateOperator(basis_l::CompositeBasis, basis_r::CompositeBasis, operators::NTuple{N, Dict{CorrelationMask{N}, Operator}})
        @assert N == length(basis_l.bases) == length(basis_r.bases)
        for n=1:N
            for (indices, op) in operators[n]
                @assert sum(indices)==n
                @assert length(op.basis_l.shape)==n
                @assert length(op.basis_r.shape)==n
                b_l = tensor([basis_l.bases[i] for i=1:N if indices[i]]...)
                b_r = tensor([basis_r.bases[i] for i=1:N if indices[i]]...)
                @assert b_l == op.basis_l
                @assert b_r == op.basis_r
            end
        end
        new(basis_l, basis_r, operators)
    end
end

function ApproximateOperator{N}(basis_l::CompositeBasis, basis_r::CompositeBasis, S::Set{CorrelationMask{N}})
    operators = tuple([Dict{CorrelationMask{N}, Operator}() for i=1:N]...)
    S_1 = correlationmasks(N, 1)
    for s in S ∪ S_1
        indices = mask2indices(s)
        op = tensor([DenseOperator(basis_l.bases[i], basis_r.bases[i]) for i in indices]...)
        operators[sum(s)][s] = op
    end
    ApproximateOperator{N}(basis_l, basis_r, operators)
end

ApproximateOperator{N}(basis::CompositeBasis, S::Set{CorrelationMask{N}}) = ApproximateOperator(basis, basis, S)

function ApproximateOperator{N}(op::DenseOperator, S::Set{CorrelationMask{N}})
    operators = tuple([Dict{CorrelationMask{N}, Operator}() for i=1:N]...)
    S_1 = correlationmasks(N, 1)
    for s in S ∪ S_1
        operators[sum(s)][s] = ptrace(op, mask2indices(complement(s)))
    end
    ApproximateOperator{N}(op.basis_l, op.basis_r, operators)
end

function productoperator{N}(op::ApproximateOperator{N})
    tensor([op.operators[1][indices2mask(N, [i])] for i=1:N]...)
end

function correlationoperator{N}(op::ApproximateOperator{N}, s::CorrelationMask{N})
    indices = mask2indices(s)
    # println("indices: ", indices)
    complement_indices = mask2indices(complement(s))
    # println("compl indices: ", complement_indices)
    op_compl = tensor([op.operators[1][indices2mask(N, [i])] for i=complement_indices]...)
    x = (op_compl ⊗ op.operators[sum(s)][s])
    # println("x.basis_l shape: ", x.basis_l.shape)
    # println("x.basis_r shape: ", x.basis_r.shape)
    # println("reshape: ", [reverse(x.basis_l.shape); reverse(x.basis_r.shape)])
    data = reshape(x.data, [reverse(x.basis_l.shape); reverse(x.basis_r.shape)]...)
    # println(size(data))
    perm = [complement_indices; indices; complement_indices+N; indices+N]
    # println("permutation", perm)
    data = permutedims(data, [complement_indices; indices; complement_indices+N; indices+N])
    # println(size(data))
    DenseOperator(op.basis_l, op.basis_r, reshape(data, length(op.basis_l), length(op.basis_r)))
end

function full{N}(op::ApproximateOperator{N})
    result = DenseOperator(op.basis_l, op.basis_r)
    for n=2:N
        for s in keys(op.operators[n])
            result += correlationoperator(op, s)
        end
    end
    result
end

end