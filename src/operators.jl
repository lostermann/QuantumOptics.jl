module operators

import Base: trace, ==, +, -, *, /
import ..bases: tensor, ptrace, permutesystems
import ..states: dagger, normalize, normalize!

using ..bases, ..states

export Operator,
       dagger, identityoperator,
       trace, ptrace, normalize, normalize!, expect,
       tensor, permutesystems, embed


"""
Abstract base class for all operators.

All deriving operator classes have to define the fields
basis_l and basis_r defining the left and right side bases.

For fast time evolution also at least the function
gemv!(alpha, op::Operator, x::Ket, beta, result::Ket) should be
implemented. Many other generic multiplication functions can be defined in
terms of this function and are provided automatically.
"""
abstract Operator

Base.eltype(x::Operator) = Complex128
Base.zero{T<:Operator}(x::T) = T(x.basis_l, x.basis_r)


# Common error messages
arithmetic_unary_error(funcname, x::Operator) = throw(ArgumentError("$funcname is not defined for this type of operator: $(typeof(x)).\nTry to convert to another operator type first with e.g. full() or sparse()."))
arithmetic_binary_error(funcname, a::Operator, b::Operator) = throw(ArgumentError("$funcname is not defined for this combination of types of operators: $(typeof(a)), $(typeof(b)).\nTry to convert to a common operator type first with e.g. full() or sparse()."))
addnumbererror() = throw(ArgumentError("Can't add or subtract a number and an operator. You probably want 'op + identityoperator(op)*x'."))


# Arithmetic operations
*(a::Operator, b::Operator) = arithmetic_binary_error("Multiplication", a, b)

+(a::Operator, b::Operator) = arithmetic_binary_error("Addition", a, b)
+(a::Number, b::Operator) = addnumbererror()
+(a::Operator, b::Number) = addnumbererror()

-(a::Operator) = arithmetic_unary_error("Negation", a)
-(a::Number, b::Operator) = addnumbererror()
-(a::Operator, b::Number) = addnumbererror()


"""
Hermitian conjugate of the given operator.
"""
dagger(a::Operator) = arithmetic_unary_error("Hermitian conjugate", a)

"""
Identity operator.
"""
identityoperator{T<:Operator}(::Type{T}, b1::Basis, b2::Basis) = throw(ArgumentError("Identity operator not defined for operator type $T."))
identityoperator{T<:Operator}(::Type{T}, b::Basis) = identityoperator(T, b, b)
identityoperator{T<:Operator}(op::T) = identityoperator(T, op.basis_l, op.basis_r)

"""
Trace of given operator.
"""
trace(x::Operator) = arithmetic_unary_error("Trace", x)

"""
Partial trace of the given operator over the specified indices.
"""
ptrace(a::Operator, index::Vector{Int}) = arithmetic_unary_error("Partial trace", a)
ptrace(a::Operator, index::Int) = ptrace(a, Int[index])

"""
Normalized copy of given operator (trace is 1.).
"""
normalize(op::Operator) = op/trace(op)
normalize!(op::Operator) = throw(ArgumentError("$funcname is not defined for this type of operator: $(typeof(x)).\n You may have to fall back to the non-inplace version 'normalize()'."))

"""
Expectation value of the given operator for the specified state(s).
"""
expect(op::Operator, state::Ket) = dagger(state)*(op*state)
expect(op::Operator, state::Operator) = trace(op*state)
expect(op::Operator, states::Vector) = [expect(op, state) for state=states]


"""
Tensor product of operators.
"""
tensor(a::Operator, b::Operator) = arithmetic_binary_error("Tensor product", a, b)

"""
Change the ordering of the subsystems of the given operator.

Arguments
---------
a
    An operator represented in a composite basis.
perm
    Vector defining the new ordering of the subsystems.
"""
permutesystems(a::Operator, perm::Vector{Int}) = arithmetic_unary_error("Permutations of subsystems", a)


"""
Vector of indices that are not in the given vector.
"""
complement(N::Int, indices) = Int[i for i=1:N if i ∉ indices]

"""
Tensor product of operators where missing indices are filled up with identity operators.

Arguments
---------
basis_l
    Left hand side basis of the resulting operator.
basis_r
    Right hand side basis of the resulting operator.
indices
    Indices of the subsystems in which the given operators live.
operators
    Operators defined in the subsystems.
"""
function embed{T<:Operator}(basis_l::CompositeBasis, basis_r::CompositeBasis,
                            indices::Vector{Int}, operators::Vector{T})
    N = length(basis_l.bases)
    @assert length(basis_r.bases) == N
    @assert length(indices) == length(Set(indices)) == length(operators)
    @assert maximum(indices) <= N
    @assert minimum(indices) > 0
    tensor([i ∈ indices ? operators[findfirst(indices, i)] : identityoperator(T, basis_l.bases[i], basis_r.bases[i]) for i=1:N]...)
end
embed(basis_l::CompositeBasis, basis_r::CompositeBasis, index::Int, op::Operator) = embed(basis_l, basis_r, Int[index], [op])
embed(basis::CompositeBasis, index::Int, op::Operator) = embed(basis, basis, Int[index], [op])
embed{T<:Operator}(basis::CompositeBasis, indices::Vector{Int}, operators::Vector{T}) = embed(basis, basis, indices, operators)

"""
Tensor product of operators where all missing indices are identity operators.

Arguments
---------
basis_l
    Left hand side basis of the resulting operator.
basis_r
    Right hand side basis of the resulting operator.
operators
    Dictionary specifying to which subsystems the corresponding operator
    belongs.
"""
function embed{T<:Operator}(basis_l::CompositeBasis, basis_r::CompositeBasis,
                            operators::Dict{Vector{Int}, T})
    @assert length(basis_l.bases) == length(basis_r.bases)
    N = length(basis_l.bases)
    if length(operators) == 0
        return identityoperator(T, basis_l, basis_r)
    end
    indices, operator_list = zip(operators...)
    operator_list = [operator_list...;]
    indices = [indices...;]
    complement_operators = [identityoperator(T, basis_l.bases[i], basis_r.bases[i]) for i in complement(N, indices)]
    op = tensor([operator_list; complement_operators]...)
    perm = sortperm([indices; complement(N, indices)])
    permutesystems(op, perm)
end
embed{T<:Operator}(basis_l::CompositeBasis, basis_r::CompositeBasis, operators::Dict{Int, T}; kwargs...) = embed(basis_l, basis_r, Dict([i]=>op_i for (i, op_i) in operators); kwargs...)
embed{T<:Operator}(basis::CompositeBasis, operators::Dict{Int, T}; kwargs...) = embed(basis, basis, operators; kwargs...)
embed{T<:Operator}(basis::CompositeBasis, operators::Dict{Vector{Int}, T}; kwargs...) = embed(basis, basis, operators; kwargs...)

gemv!() = error("Not Implemented.")
gemm!() = error("Not Implemented.")

# Helper functions to check validity of arguments
function check_ptrace_arguments(a::Operator, indices::Vector{Int})
    @assert length(a.basis_l.shape) == length(a.basis_r.shape)
    S = Set(indices)
    @assert length(S) == length(indices)
    @assert issubset(S, Set(1:length(a.basis_l.shape)))
end
check_samebases(a::Operator, b::Operator) = ((a.basis_l!=b.basis_l) || (a.basis_r!=b.basis_r) ? throw(IncompatibleBases()) : nothing)


end # module
