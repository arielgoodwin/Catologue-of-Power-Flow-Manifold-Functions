# imports 
using LinearAlgebra
using Random
using ForwardDiff
using OrdinaryDiffEq
using DifferentialEquations

#### Zeroth-Order Functions: Φ, F

# Power flow function, no (p,q) variables
# Φ(v,θ) on Page 2 of paper
# Input: x = (v,θ), Y the admittance matrix
# Output: Φ(v,θ), vector in R^{2n}
function bigphi(x,Y)
    n = size(Y)[1]
    v = x[1:n]
    θ = x[n+1:2*n]
    G = real(Y)
    B = imag(Y)
    output = zeros(2*n)
    theta = zeros(n,n)
    for i = 1:n 
        for k = 1:n
            theta[i,k] = θ[i]-θ[k]
            output[i] += v[i]*v[k]*(G[i,k]*cos(theta[i,k]) + B[i,k]*sin(theta[i,k]))
        end
    end
    for i = n+1:2*n
        for k = 1:n
            output[i] += v[i-n]*v[k]*(G[i-n,k]*sin(theta[i-n,k])- B[i-n,k]*cos(theta[i-n,k]))
        end
    end
    return output
end

# Power flow function, with (p,q) variables
# F(v,θ,p,q) on Page 1 of the paper
# Input: x = (v,θ,p,q), Y the admittance matrix
# Output: F(v,θ,p,q), vector in R^{2n}
function powerfloweqn(x,Y)
    n = size(Y)[1]
    return bigphi(x[1:2*n],Y) - x[2*n+1:4*n]
end

#### First-Order Functions: DΦ, DF

# helper function, used in DΦ function below. Borrowed from Bolognani & Dorfler formula.
# Input: Matrix A
# Output: A larger matrix
function brak(A)
    return [real(A) -imag(A); imag(A) real(A)]
end

# Jacobian of power flow Φ function
# Using formula from Bolognani & Dorfler on Page 2 of paper
# Input: x = (v,θ), Y the admittance matrix
# Output: Jacobian of Φ at (v,θ), a 2n x 2n matrix
function jac_powerflow_phi(x,Y)
    n = size(Y)[1]
    v = x[1:n]
    θ = x[n+1:2n]
    u = v.*exp.(θ.*im)
    return (brak(conj(Diagonal(Y*u))) + brak(Diagonal(u))*[I zeros(n,n); zeros(n,n) -I]*brak(Y))*[Diagonal(cos.(θ)) -Diagonal(v)*Diagonal(sin.(θ));
    Diagonal(sin.(θ)) Diagonal(v)*Diagonal(cos.(θ))]
end
            
# Jacobian of power flow F function
# Formula in Proof of Prop II.2 in paper
# Input: x = (v,θ), Y the admittance matrix
# Output: Jacobian of F at (v,θ,p,q), a 2n x 4n matrix 
function jac_powerflow_eqn(x,Y)
    n = size(Y)[1]
    return [jac_powerflow_phi(x,Y) -I(2*n)]
end

#### Tangent Space Functions: Basis, Orthogonal Projection

# Produce basis for tangent space at (x,p,q)
# Formula in Prop II.2 in paper
# Input: x = (v,θ), Y the admittance matrix
# Output: Matrix whose 2n columns are basis for tangent space at (x,p,q)
function tangent_basis(x,Y)
    n = size(Y)[1]
    D = jac_powerflow_phi(x,Y)
    return [I(2n); D]
end

# Orthogonally project u in R^{4n} onto tangent space at (x,p,q)
# Formula in Corollary II.1 in paper
# Input: x = (v,θ), u a vector in R^{4n}, Y the admittance matrix
# Output: The orthogonal projection of u onto the appropriate tangent space
function orth_proj_tang(x,u,Y)
    n = size(Y)[1]
    D = jac_powerflow_phi(x,Y)
    M = I(2n) + D'*D
    F = cholesky(M)
    u_1 = u[1:2n]
    u_2 = u[2n+1:4n]
    
    v_1 = F\u_1 + F\(D'*u_2)
    v_2 = D*v_1
    return [v_1;v_2]
end  

#### Second-order Functions: D^2Φ, H_x[u]

# helper function, used in second derivative computation
# Kronecker delta, 1 if x == y, 0 else
δ(x,y) = ==(x,y)


# Computes all second-order partial derivatives of Φ
# Formulas in Prop A.1 of Paper
# Input: x = (v,θ), Y the admittance matrix
# Output: 3D array, where entry (l,j,i) is the partial derivative of Φ_i (i-th row of Φ) wrt l-th and j-th variables,
#         using the convention that v = x[1:n], θ = x[n+1:2n]. Ex: output[:,:,i] is Hessian of Φ_i
function second_der_phi(x,Y)
    n = size(Y)[1]
    G = real(Y)
    B = imag(Y)
    v = x[1:n]
    θ = x[n+1:2n]
    H = zeros(2n,2n,2n)
    for i =  1:n
        # v derivatives
        for j = 1:n
            # v_l v_j derivatives
            for l = 1:n
                H[l,j,i] = δ(i,j)*(G[i,l]*cos(θ[i] - θ[l]) + B[i,l]*sin(θ[i] - θ[l])) + δ(i,l)*(G[i,j]*cos(θ[i] - θ[j]) + B[i,j]*sin(θ[i] - θ[j]))
            end
            # θ_l v_j derivatives
            for l = n+1:2n
                s = 0
                for k = 1:n
                    s += v[k]*(B[i,k]*cos(θ[i] - θ[k]) - G[i,k]*sin(θ[i] - θ[k]))
                end
                H[l,j,i] = v[i]*(δ(i,l-n) - δ(j,l-n))*(B[i,j]*cos(θ[i] - θ[j]) - G[i,j]*sin(θ[i] - θ[j]))- δ(i,j)*v[l-n]*(B[i,l-n]*cos(θ[i] - θ[l-n]) - G[i,l-n]*sin(θ[i] - θ[l-n]))+ δ(i,j)*δ(i,l-n)*s
            end
            
        end
        # θ derivatives
        for j = n+1:2n
            # θ_j v_l derivatives
            for l = 1:n
                s = 0
                for k = 1:n
                    s += v[k]*(B[i,k]*cos(θ[i] - θ[k]) - G[i,k]*sin(θ[i] - θ[k]))
                end
                H[l,j,i] = v[i]*(δ(i,j-n) - δ(l,j-n))*(B[i,l]*cos(θ[i] - θ[l]) - G[i,l]*sin(θ[i] - θ[l]))- δ(i,l)*v[j-n]*(B[i,j-n]*cos(θ[i] - θ[j-n]) - G[i,j-n]*sin(θ[i] - θ[j-n]))+ δ(i,l)*δ(i,j-n)*s
                
            end
            # θ_j θ_l derivatives
            for l = n+1:2n
                s = 0
                for k = 1:n
                    s += v[k]*(G[i,k]*cos(θ[i] - θ[k]) + B[i,k]*sin(θ[i] - θ[k]))
                end
                H[l,j,i] = (δ(i,l-n) - δ(j-n,l-n))*v[i]*v[j-n]*(G[i,j-n]*cos(θ[i] - θ[j-n]) + B[i,j-n]*sin(θ[i] - θ[j-n]))+ δ(i,j-n)*v[i]*v[l-n]*(G[i,l-n]*cos(θ[i] - θ[l-n]) + B[i,l-n]*sin(θ[i] - θ[l-n]))- δ(i,j-n)*δ(i,l-n)*v[i]*s
            end
        end
    end
    
    for i =  1:n
        for j = 1:n
            for l = 1:n
                H[l,j,i+n] = δ(i,j)*(-B[i,l]*cos(θ[i] - θ[l]) + G[i,l]*sin(θ[i] - θ[l])) + δ(i,l)*(-B[i,j]*cos(θ[i] - θ[j]) + G[i,j]*sin(θ[i] - θ[j]))
            end
            for l = n+1:2n
                s = 0
                for k = 1:n
                    s += v[k]*(G[i,k]*cos(θ[i] - θ[k]) + B[i,k]*sin(θ[i] - θ[k]))
                end
                H[l,j,i+n] = v[i]*(δ(i,l-n) - δ(j,l-n))*(G[i,j]*cos(θ[i] - θ[j]) + B[i,j]*sin(θ[i] - θ[j]))- δ(i,j)*v[l-n]*(G[i,l-n]*cos(θ[i] - θ[l-n]) + B[i,l-n]*sin(θ[i] - θ[l-n]))+ δ(i,j)*δ(i,l-n)*s
            end
        end
        
        for j = n+1:2n
            for l = 1:n
                s = 0
                for k = 1:n
                    s += v[k]*(G[i,k]*cos(θ[i] - θ[k]) + B[i,k]*sin(θ[i] - θ[k]))
                end
                H[l,j,i+n] = v[i]*(δ(i,j-n) - δ(l,j-n))*(G[i,l]*cos(θ[i] - θ[l]) + B[i,l]*sin(θ[i] - θ[l]))- δ(i,l)*v[j-n]*(G[i,j-n]*cos(θ[i] - θ[j-n]) + B[i,j-n]*sin(θ[i] - θ[j-n]))+ δ(i,l)*δ(i,j-n)*s
            end
            for l = n+1:2n
                s = 0
                for k = 1:n
                    s += v[k]*(-B[i,k]*cos(θ[i] - θ[k]) + G[i,k]*sin(θ[i] - θ[k]))
                end
                H[l,j,i+n] = (δ(i,l-n) - δ(j-n,l-n))*v[i]*v[j-n]*(-B[i,j-n]*cos(θ[i] - θ[j-n]) + G[i,j-n]*sin(θ[i] - θ[j-n]))+ δ(i,j-n)*v[i]*v[l-n]*(-B[i,l-n]*cos(θ[i] - θ[l-n]) + G[i,l-n]*sin(θ[i] - θ[l-n]))- δ(i,j-n)*δ(i,l-n)*v[i]*s
            end
        end
    end
    return H
end

# Computes the matrix of "stacked" Hessians
# Formula in Theorem II.2 in paper
# Input: Point x = (v,θ) in R^{2n}, u in R^{4n}, Y the admittance matrix
# Output: 2n x 4n matrix
function hess_stack(x,u,Y)
    n = size(Y)[1]
    HX = zeros(2n,4n)
    H = second_der_phi(x,Y)
    
    for i = 1:2n
        F_i = [H[:,:,i] zeros(2n,2n); zeros(2n,2n) zeros(2n,2n)] 
        HX[i,:] = (F_i*u)'
    end
    
    return HX
end

#### Metric Functions: g, Dg, Γ

# Computes the posdef matrix g(v,θ) for the Riemannian metric on R^{2n} coming from the power flow manifold
# Formula on Page 3 of paper
# Input: x = (v,θ), Y the admittance matrix
# Output: A matrix
function powerflow_metric(x,Y)
    n = size(Y)[1]
    D = jac_powerflow_phi(x,Y)
    return I(2*n) + D'*D
end

# Computes all the first-order partial derivatives of the metric g (i.e., ∂_k g_{ij})
# Formula on Page 3 of paper
# Input: x = (v,θ), Y the admittance matrix
# Output: 3D array, where entry [i,j,k] is ∂_k g_{ij}
function metric_deriv(x,Y)
    n = size(Y)[1]
    H = second_der_phi(x,Y)
    D = jac_powerflow_phi(x,Y)
    Dg = zeros(2n,2n,2n)
    
    for k = 1:2n
        for i = 1:2n
            for j = 1:2n
                Dg[i,j,k] = H[k,i,:]'*D[:,j] + D[:,i]'*H[k,j,:]
            end
        end
    end
    
    return Dg
end

# Computes all the Christoffel symbols Γ_{ij}^k
# Formula on Page 3 of paper
# Input: x = (v,θ), Y the admittance matrix
# Output: 3D array, where entry [i,j,k] is Γ_{ij}^k
function christoffel(x,Y)
    n = size(Y)[1]
    Dg = metric_deriv(x,Y)
    g = powerflow_metric(x,Y)
    F = cholesky(g)
    Γ = zeros(2n,2n,2n)
    
    for i = 1:2n
        for j = 1:2n
            x_ij = Dg[j,:,i]
            y_ij = Dg[i,:,j]
            z_ij = Dg[i,j,:]
            Γ[i,j,:] = (1/2)*(F\(x_ij + y_ij - z_ij))
        end
    end
    
    return Γ
end

#### Geodesic Functions: Q, exp

# Computes the righthand side of the ODE system for the geodesic equations
# Formula Q on Page 3 of paper
# Input: (x,y) in R^{4n}, x are variables, y are dummy variables to make the system 1st order, Y the admittance matrix
# Output: A vector in R^{4n}
function Q(x,y,Y)
    n = size(Y)[1]
    Γ = christoffel(x,Y)
    out = zeros(4n)
    out[1:2n] = y
    
    for k = 1:2n
        s = 0
        for i = 1:2n
            for j = 1:2n
                s -= y[i]*y[j]*Γ[i,j,k]
            end
        end
        out[2n+k] = s
    end
    
    return out
end

# Function for use in ODE solver; note it modifies its arguments
# Format based on DifferentialEquations syntax
# Input: dz (vector for storing RHS of ODE system output), z (variables), p (parameters, namely admittance Y), t (for solver)
# Output: None, modifies its arguments internally
function geodesic!(dz, z, p, t)
    Y = p
    n = size(Y)[1]
    x = z[1:2n]
    y = z[2n+1:4n]
    dz[1:2n] = z[2n+1:4n]
    u = Q(x,y,Y)
    dz[2n+1:4n] = u[2n+1:4n]
end

# Computes the exponential map with initial point u0, initial velocity v0
# Works by solving the geodesic equations implemented by the geodesic! function
# Input: u0, v0 initial position and velocity in R^{2n}, respectively, Y the admittance matrix
# Output: exp_{u0}(v0), final position and velocity obtained after travelling for 1 time unit along geodesic
function exp_map(u0,v0,Y)
    tspan = (0.0, 1.0)
    prob = ODEProblem(geodesic!, [u0;v0], tspan, Y)
    sol = solve(prob,reltol=1e-8,verbose=false)
    
    return sol
    # not quite complete, should figure out how to return this data in a convenient/silent way
end

#### Curvature Functions: Second Fundamental Form, Weingarten, Curvature Tensor, Sectional Curvature, Scalar Curvature

# Computes the second fundamental form
# Formula in Corollary II.2 in paper
# Input: x = (v,θ), u,v tangent vectors to the PFM, Y the admittance matrix
# Output: II(u,v), a normal vector the PFM
function sff(x,u,v,Y)
    Hu = hess_stack(x,u,Y)
    z = Hu*v
    D = jac_powerflow_eqn(x,Y)
    F = cholesky(D*D')
    w = F\z
    return -D'*w
end

# Computes the Weingarten map 
# Formula in Corollary II.2 in paper
# Input: x = (v,θ), u tangent vector, w normal vector, Y the admittance matrix
# Output: W(u,w), a tangent vector to the PFM
function wein(x,u,w,Y)
    Hu = hess_stack(x,u,Y)
    D = jac_powerflow_eqn(x,Y)
    F = cholesky(D*D')
    z = D*w
    a = F\z
    return -orth_proj_tang(x,Hu'*a,Y)
end

# Computes the Riemann curvature tensor
# Formula in Corollary II.3 in paper
# Input: x = (v,θ), u,v,w,z tangent vectors at x, Y the admittance matrix
# Output: The scalar Rm_x(u,v,w,z)
function curv_tensor(x,u,v,w,z,Y)
    Hu = hess_stack(x,u,Y)
    Hv = hess_stack(x,v,Y)
    D = jac_powerflow_eqn(x,Y)
    F = cholesky(D*D')
    c = Hv*w
    d = F\c
    f = Hv*z
    r = F\f
    return z'*(Hu)'d - w'*(Hu)'*r
end

# Computes the sectional curvature 
# Formula on Page 5 of the paper
# Input: x = (v,θ), u,v tangent vectors at x, Y the admittance matrix
# Output: The scalar K_x(u,v)
function sect_curv(x,u,v,Y)
    denom = norm(u)^2*norm(v)^2 - u'*v
    return curv_tensor(x,u,v,v,u,Y)/denom
end

# Computes the scalar curvature
# Formula on Page 5 of paper
# Input: x = (v,θ), Y the admittance matrix
# Output: The scalar S(x)
function scalar_curv(x,Y)
    n = size(Y)[1]
    B = tangent_basis(x,Y)
    Q = qr(B).Q
    S = 0
    
    for i = 1:2n
        for j = 1:2n
            if i != j
                S += sect_curv(x,Q[:,i],Q[:,j],Y)
            end
        end
    end
    
    return S
end
