module ThinWalledBeamColumn

using FornbergFiniteDiff, LinearAlgebra, NLsolve


export solve



struct Dof

    u::Array{Int64}
    v::Array{Int64}
    ϕ::Array{Int64}
 
 end
 
 struct Inputs   
 
    z::Array{Float64}
    A::Array{Float64}
    Ix::Array{Float64}
    Iy::Array{Float64}
    Io::Array{Float64}
    J::Array{Float64}
    Cw::Array{Float64}
    E::Array{Float64}
    G::Array{Float64}
    ax::Array{Float64} #distance from shear center to qy applied load location
    ay::Array{Float64}
    kx::Array{Float64}
    ky::Array{Float64}
    kϕ::Array{Float64}
    hx::Array{Float64} #distance from centroid to ky spring location
    hy::Array{Float64}
 
    qx::Array{Float64}
    qy::Array{Float64}
    P::Array{Float64}
 
    end_boundary_conditions::Array{String} 
    supports::Vector{Tuple{Float64, String, String, String}} 
 
 end
 
 struct Equations
 
    K::Matrix{Float64}
    F::Array{Float64}
    Kff::Matrix{Float64}
    Ff::Array{Float64}
    free_dof::Dof
    free_dof_global::Array{Int64}
 
 end
 
 struct Outputs
 
    u::Array{Float64}
    v::Array{Float64}
    ϕ::Array{Float64}
 
 end
 
 struct Model
 
    inputs::Inputs
    equations::Equations
    outputs::Outputs
 
 end
 
 
 
 
 function calculate_derivative_operators(dz)

   z = [0.0; cumsum(dz)]
   num_nodes = length(z)

   #Calculate the 4th derivative operator.
   Azzzz = zeros(Float64, (num_nodes, num_nodes))
   nth_derivative = 4
   
   for i=3:num_nodes-2
   
      x = z[i-2:i+2]
      x0 = x[3]
      stencil = FornbergFiniteDiff.calculate_weights(nth_derivative, x0, x)
      Azzzz[i, i-2:i+2] = stencil

   end


   #Calculate the 2nd derivative operator.
   Azz = zeros(Float64, (num_nodes, num_nodes))
   nth_derivative = 2
   
   for i=3:num_nodes-2
   
      x = z[i-2:i+2]
      x0 = x[3]
      stencil = FornbergFiniteDiff.calculate_weights(nth_derivative, x0, x)
      Azz[i, i-2:i+2] = stencil

   end

   return Azzzz, Azz

end

 
 function calculate_boundary_stencils(bc_flag, h, nth_derivative)
 
    #Calculate boundary conditions stencils without ghost nodes using
    #Jorge M. Souza, "Boundary Conditions in the Finite Difference Method"
    #https://cimec.org.ar/ojs/index.php/mc/article/download/2662/2607
 
 
    taylor_coeffs =  [1; h; h^2/2; h^3/6; h^4/24]
 
    RHS = zeros(Float64, 5)
    #row1*[A;B;C;D;E]*u2
    #row2*[A;B;C;D;E]*u2'
    #row3*[A;B;C;D;E]*u2''
    #row4*[A;B;C;D;E]*u2'''
    #row5*[A;B;C;D;E]*u2''''
 
    if bc_flag == "simply-supported" #simply supported end
 
       LHS = [1 1 1 1 0
          -1 0 1 2 0
          1 0 1 4 2
          -1 0 1 8 -6
          1 0 1 16 12]
 
       RHS[nth_derivative + 1] = (1/taylor_coeffs[nth_derivative + 1])
       boundary_stencil = LHS \ RHS
       boundary_stencil = ((boundary_stencil[1:4]),(zeros(4)))  #since u''=0
 
    elseif bc_flag == "fixed" #fixed end
 
       LHS = [1 1 1 1 0
          -1 0 1 2 1
          1 0 1 4 -2
          -1 0 1 8 3
          1 0 1 16 -4]
 
       RHS[nth_derivative+1] = (1/taylor_coeffs[nth_derivative + 1])
       boundary_stencil = LHS\RHS
       boundary_stencil = ((boundary_stencil[1:4]),(zeros(4)))  #since u'=0
 
    elseif bc_flag == "free" #free end
                 #u'' u'''
       LHS = [1 1 1  0   0
            0 1 2  0   0
            0 1 4  0.5 0
            0 1 8  0   6
            0 1 16 0  0]
 
       RHS[nth_derivative+1] = (1/taylor_coeffs[nth_derivative+1])
       boundary_stencil1 = LHS\RHS  #at free end
       boundary_stencil1 = boundary_stencil1[1:3]   #u''=u'''=0
 
       # use simply supported BC to find stencil at one node in from free end
       LHS = [1 1 1 1 0
          -1 0 1 2 0
          1 0 1 4 2
          -1 0 1 8 -6
          1 0 1 16 12]
 
       RHS[nth_derivative+1] = (1/taylor_coeffs[nth_derivative+1])
       boundary_stencil2 = LHS\RHS #at one node over from free end
       boundary_stencil2 = boundary_stencil2[1:4]
       boundary_stencil = ((boundary_stencil1), (boundary_stencil2))  #two stencils are calculated
 
    end
 
    return boundary_stencil
 
 end
 
 function apply_end_boundary_conditions(A, end_boundary_conditions, nth_derivative, dz)
 
    #Consider the left end boundary condition.  Swap out interior stencil for boundary stencil.
    h = dz[1]
    bc_flag = end_boundary_conditions[1]
    boundary_stencil = calculate_boundary_stencils(bc_flag, h, nth_derivative)
 
    A[1,:] .= 0.0
    A[2,:] .= 0.0
 
    if (bc_flag == "simply-supported") | (bc_flag == "fixed")   #make this cleaner, combine
       A[2,1:length(boundary_stencil[1])] = boundary_stencil[1]
    else
       A[1,1:length(boundary_stencil[1])] = boundary_stencil[1]
       A[2,1:length(boundary_stencil[2])] = boundary_stencil[2]
    end
 
    #Consider the right end boundary condition.
    h = dz[end]
    bc_flag = end_boundary_conditions[2]
    boundary_stencil = calculate_boundary_stencils(bc_flag,h,nth_derivative)
 
    A[end,:] .= 0.0
    A[end-1,:] .= 0.0
 
    if (bc_flag == "simply-supported") | (bc_flag == "fixed")
       A[end-1,(end-(length(boundary_stencil[1])-1)):end] = reverse(boundary_stencil[1])
    else
       A[end,end-(length(boundary_stencil[1])-1):end] = reverse(boundary_stencil[1])
       A[end-1,end-(length(boundary_stencil[2])-1):end] = reverse(boundary_stencil[2])
    end
 
    return A
 
 end
 
 
 function define_fixed_dof(z, support_location, support_fixity)
 
    if support_fixity == "fixed"
 
       fixed_dof = findall(x->x≈support_location, z)
 
    else
 
       fixed_dof = []
 
    end
 
    return fixed_dof
 
 end
 
 
 function governing_equations(z, A, Ix, Iy, Io, J, Cw, E, G, ax, ay, kx, ky, kϕ, hx, hy, qx, qy, P, end_boundary_conditions, supports)
 
     #The equilbrium equations come from Plaut and Moen (2020) https://www.sciencedirect.com/science/article/abs/pii/S0263823120307758 without imperfection terms and with additional beam terms from Plaut and Moen(2019) https://cloud.aisc.org/SSRC/2020/43PlautandMoen2020SSRC.pdf.
 
    #Define the number of nodes.
    num_nodes = length(z)
 
    #Calculate the derivative operators.
    dz = diff(z) 
    Azzzz,Azz = calculate_derivative_operators(dz) 
 
    #Apply left and right end boundary condition stencils to derivative operators.
    nth_derivative = 4
    Azzzz = apply_end_boundary_conditions(Azzzz, end_boundary_conditions, nth_derivative, dz)
 
    nth_derivative = 2
    Azz = apply_end_boundary_conditions(Azz, end_boundary_conditions, nth_derivative, dz)
 
    #Build identity matrix for ODE operations.
    AI = Matrix(1.0I, num_nodes, num_nodes)
 
    #Build operator matrix that doesn't update with load.
 
    #LHS first.
    A11 = zeros(Float64, num_nodes, num_nodes)
    A12 = zeros(Float64, num_nodes, num_nodes)
    A13 = zeros(Float64, num_nodes, num_nodes)
    A21 = zeros(Float64, num_nodes, num_nodes)
    A22 = zeros(Float64, num_nodes, num_nodes)
    A23 = zeros(Float64, num_nodes, num_nodes)
    A31 = zeros(Float64, num_nodes, num_nodes)
    A32 = zeros(Float64, num_nodes, num_nodes)
    A33 = zeros(Float64, num_nodes, num_nodes)
 
    #Calculate operator quantities on LHS  AU=B.
    for i = 1:num_nodes
       A11[i,:] = E .* Iy .* Azzzz[i,:] .+ P .* Azz[i,:] .+ kx .* AI[i,:]
       A13[i,:] = -kx .* hy .* AI[i,:]
       A22[i,:] = E .* Ix .* Azzzz[i,:] .+ P .* Azz[i,:] .+ ky .* AI[i,:]
       A23[i,:] = ky .* hx .* AI[i,:]
       A31[i,:] = -kx .* hy .* AI[i,:]
       A32[i,:] = ky .* hx .* AI[i,:]
       A33[i,:] = E .* Cw .* Azzzz[i,:] .-(G .* J .- (P .* Io ./ A)).*Azz[i,:] .+kx .* hy.^2 .*AI[i,:] .+ ky .* hx.^2 .*AI[i,:] .+ kϕ .* AI[i,:] .+  qx .* ax .* AI[i,:] - .+ qy .* ay .* AI[i,:]
    end
 
    #Calculate RHS of AU=B.
    B1 = qx
    B2 = qy
    B3 = qx .* ay .+ qy .* ax
 
    #Reduce problem to free dof.
 
    fixed_dof_u = Array{Int64}(undef, 1)
    fixed_dof_v = Array{Int64}(undef, 1)
    fixed_dof_ϕ = Array{Int64}(undef, 1)
 
    for i = 1:length(supports)
 
       if i == 1
 
          fixed_dof_u = define_fixed_dof(z, supports[i][1], supports[i][2])
          fixed_dof_v = define_fixed_dof(z, supports[i][1], supports[i][3])
          fixed_dof_ϕ = define_fixed_dof(z, supports[i][1], supports[i][4])
 
       else
 
          new_fixed_dof_u = define_fixed_dof(z, supports[i][1], supports[i][2])
          new_fixed_dof_v = define_fixed_dof(z, supports[i][1], supports[i][3])
          new_fixed_dof_ϕ = define_fixed_dof(z, supports[i][1], supports[i][4])
          
          fixed_dof_u = [fixed_dof_u; new_fixed_dof_u]
          fixed_dof_v = [fixed_dof_v; new_fixed_dof_v]
          fixed_dof_ϕ = [fixed_dof_ϕ; new_fixed_dof_ϕ]
 
       end
 
    end
 
    #Define free dof. 
    free_dof = Dof(setdiff(1:num_nodes,fixed_dof_u), setdiff(1:num_nodes,fixed_dof_v), setdiff(1:num_nodes,fixed_dof_ϕ))
 
    #Assemble global K.
    K = [A11 A12 A13;
    A21 A22 A23;
    A31 A32 A33]
 
    #Partition stiffness matrix.
    free_dof_global = [free_dof.u; free_dof.v .+ num_nodes; free_dof.ϕ .+ 2*num_nodes]
 
    Kff = K[free_dof_global, free_dof_global]
 
    #Define external force vector.
    F = [B1; B2; B3]
 
    Ff = F[free_dof_global]
 
    equations = Equations(K, F, Kff, Ff, free_dof, free_dof_global)
 
    return equations
 
 end
 
 function residual!(R, U, K, F)
 
    for i=1:length(F)
 
       R[i] = transpose(K[i,:]) * U - F[i]
 
    end
 
    return R
 
 end
 
 
 function solve(z, A, Ix, Iy, Io, J, Cw, E, G, ax, ay, kx, ky, kϕ, hx, hy, qx, qy, P, end_boundary_conditions, supports)
 
    #Define inputs.
    inputs = Inputs(z, A, Ix, Iy, Io, J, Cw, E, G, ax, ay, kx, ky, kϕ, hx, hy, qx, qy, P, end_boundary_conditions, supports)
 
    # end_boundary_conditions::Array{String} 
    # supports::Vector{Tuple{Float64, String, String, String}} 
 
    #Set up solution matrices from governing equations.
    equations = governing_equations(z, A, Ix, Iy, Io, J, Cw, E, G, ax, ay, kx, ky, kϕ, hx, hy, qx, qy, P, end_boundary_conditions, supports)
 
    #Define the number of nodes along the beam.
    num_nodes = length(z)
 
    #Define the deformation vectors.
    u = zeros(Float64, num_nodes)
    v = zeros(Float64, num_nodes)
    ϕ = zeros(Float64, num_nodes)
 
    #Define the deformation initial guess for the nonlinear solver.
    deformation_guess = equations.Kff \ equations.Ff
 
    #Solve for the beam deformations.
    solution = nlsolve((R,U) ->residual!(R, U, equations.Kff, equations.Ff), deformation_guess)
 
    #Pull the displacements and twist from the solution results.
    u[equations.free_dof.u] = solution.zero[1:length(equations.free_dof.u)]
    v[equations.free_dof.v] = solution.zero[length(equations.free_dof.u)+1:(length(equations.free_dof.u) + length(equations.free_dof.v))]
    ϕ[equations.free_dof.ϕ] = solution.zero[(length(equations.free_dof.u) + length(equations.free_dof.v))+1:(length(equations.free_dof.u) + length(equations.free_dof.v) + length(equations.free_dof.ϕ))]
 
    outputs = Outputs(u, v, ϕ)
 
    model = Model(inputs, equations, outputs)
 
    return model
 
 end


end # module
