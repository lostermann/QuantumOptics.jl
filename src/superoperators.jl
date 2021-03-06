module superoperators

import Base: ==, *, /, +, -
import ..operators.check_samebases

using ..bases, ..operators, ..operators_dense, ..operators_sparse

export DenseSuperOperator, SparseSuperOperator, spre, spost, liouvillian

"""
Base class for all super operator classes.

Super operators are bijective mappings from operators given in one specific
basis to operators, possibly given in respect to another, different basis.
To embed super operators in an algebraic framework they are defined with a
left hand basis ``basis_l`` and a right hand basis ``basis_r`` where each of
them again consists of a left and right hand basis.

.. math::

    A_{bl_1,bl_2} &= S_{(bl_1,bl_2)<->(br_1,br_2)} B_{br_1,br_2}
    \\\\
    A_{br_1,br_2} &= B_{bl_1,bl_2} S_{(bl_1,bl_2)<->(br_1,br_2)}
"""
abstract SuperOperator


type DenseSuperOperator <: SuperOperator
    basis_l::Tuple{Basis, Basis}
    basis_r::Tuple{Basis, Basis}
    data::Matrix{Complex128}
    function DenseSuperOperator(basis_l::Tuple{Basis, Basis}, basis_r::Tuple{Basis, Basis}, data::Matrix{Complex128})
        if length(basis_l[1])*length(basis_l[2]) != size(data, 1) || length(basis_r[1])*length(basis_r[2]) != size(data, 2)
            throw(DimensionMismatch())
        end
        new(basis_l, basis_r, data)
    end
end


type SparseSuperOperator <: SuperOperator
    basis_l::Tuple{Basis, Basis}
    basis_r::Tuple{Basis, Basis}
    data::SparseMatrixCSC{Complex128}
    function SparseSuperOperator(basis_l::Tuple{Basis, Basis}, basis_r::Tuple{Basis, Basis}, data::SparseMatrixCSC{Complex128})
        if length(basis_l[1])*length(basis_l[2]) != size(data, 1) || length(basis_r[1])*length(basis_r[2]) != size(data, 2)
            throw(DimensionMismatch())
        end
        new(basis_l, basis_r, data)
    end
end

Base.full(a::SparseSuperOperator) = DenseSuperOperator(a.basis_l, a.basis_r, full(a.data))
Base.full(a::DenseSuperOperator) = deepcopy(a)

=={T<:SuperOperator}(a::T, b::T) = (a.basis_l == b.basis_l) && (a.basis_r == b.basis_r) && (a.data == b.data)

operators.check_samebases(a::SuperOperator, b::SuperOperator) = ((a.basis_l!=b.basis_l) || (a.basis_r!=b.basis_r) ? throw(IncompatibleBases()) : nothing)

function *{T<:SuperOperator}(a::T, b::DenseOperator)
    if a.basis_r[1] != b.basis_l || a.basis_r[2] != b.basis_r
        throw(DimensionMismatch())
    end
    data = a.data*reshape(b.data, length(b.data))
    return DenseOperator(a.basis_l[1], a.basis_l[2], reshape(data, length(a.basis_l[1]), length(a.basis_l[2])))
end

function *{T<:SuperOperator}(a::T, b::T)
    if a.basis_r != b.basis_l
        throw(DimensionMismatch())
    end
    return T(a.basis_l, b.basis_r, a.data*b.data)
end

/{T<:SuperOperator}(a::T, b::Number) = T(a.basis_l, a.basis_r, a.data/complex(b))

+{T<:SuperOperator}(a::T, b::T) = (operators.check_samebases(a, b); T(a.basis_l, a.basis_r, a.data+b.data))

-{T<:SuperOperator}(a::T, b::T) = (operators.check_samebases(a, b); T(a.basis_l, a.basis_r, a.data-b.data))
-{T<:SuperOperator}(a::T) = T(a.basis_l, a.basis_r, -a.data)

"""
Create a super-operator equivalent for right side operator multiplication.

For operators :math:`A`, :math:`B` the relation

.. math::

    \\mathrm{spre}(A) B = A B

holds.
"""
spre(op::DenseOperator) = DenseSuperOperator((op.basis_l, op.basis_r), (op.basis_l, op.basis_r), tensor(identityoperator(op), op).data)
spre(op::SparseOperator) = SparseSuperOperator((op.basis_l, op.basis_r), (op.basis_l, op.basis_r), tensor(identityoperator(op), op).data)

"""
Create a super-operator equivalent for left side operator multiplication.

For operators :math:`A`, :math:`B` the relation

.. math::

    \\mathrm{spost}(A) B = B A

holds.
"""
spost(op::DenseOperator) = DenseSuperOperator((op.basis_l, op.basis_r), (op.basis_l, op.basis_r), kron(transpose(op.data), identityoperator(op).data))
spost(op::SparseOperator) = SparseSuperOperator((op.basis_l, op.basis_r), (op.basis_l, op.basis_r),  kron(transpose(op.data), identityoperator(op).data))


function _check_input(H::Operator, J::Vector, Jdagger::Vector, Gamma::Union{Vector{Float64}, Matrix{Float64}})
    for j=J
        @assert typeof(j) <: Operator
        operators.check_samebases(H, j)
    end
    for j=Jdagger
        @assert typeof(j) <: Operator
        operators.check_samebases(H, j)
    end
    @assert length(J)==length(Jdagger)
    if typeof(Gamma) == Matrix{Float64}
        @assert size(Gamma, 1) == size(Gamma, 2) == length(J)
    elseif typeof(Gamma) == Vector{Float64}
        @assert length(Gamma) == length(J)
    else
        error()
    end
end


"""
Create a super-operator equivalent to the master equation.

The  so that :math:`\\dot \\rho = S \\rho`

The super-operator :math:`S` is defined by

.. math::

    \\dot \\rho = S \\rho = -\\frac{i}{\\hbar} [H,\\rho]
            + 2 J \\rho J^\\dagger - J^\\dagger J \\rho - \\rho J^\\dagger J

Arguments
---------
H
    Hamiltonian
J
    Vector of jump operators

Keyword Arguments
-----------------
Gamma (optional)
    Vector or matrix specifying the coefficients for the jump operators.
Jdagger (optional)
    Vector containing the hermitian conjugates of the jump operators. If they
    are not given they are calculated automatically.
"""
function liouvillian{T<:Operator}(H::T, J::Vector{T};
            Gamma::Union{Vector{Float64}, Matrix{Float64}}=ones(Float64, length(J)),
            Jdagger::Vector{T}=map(dagger, J))
    _check_input(H, J, Jdagger, Gamma)
    L = spre(-1im*H) + spost(1im*H)
    if typeof(Gamma) == Matrix{Float64}
        for i=1:length(J), j=1:length(J)
            jdagger_j = Gamma[i,j]/2*Jdagger[j]*J[i]
            L -= spre(jdagger_j) + spost(jdagger_j)
            L += spre(Gamma[i,j]*J[i]) * spost(Jdagger[j])
        end
    elseif typeof(Gamma) == Vector{Float64}
        for i=1:length(J)
            jdagger_j = Gamma[i]/2*Jdagger[i]*J[i]
            L -= spre(jdagger_j) + spost(jdagger_j)
            L += spre(Gamma[i]*J[i]) * spost(Jdagger[i])
        end
    else
        error()
    end
    return L
end

Base.expm(op::DenseSuperOperator) = DenseSuperOperator(op.basis_l, op.basis_r, expm(op.data))

end # module
