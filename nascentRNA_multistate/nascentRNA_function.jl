using BlockArrays
using LinearAlgebra
using ExponentialUtilities


function W_matrix(m,n,u,v,l1,l2)
    N = m + n
    W = zeros(N,N)
    k1 = u*m; k0=v*n
    W[1,1] = -k1; W[1,2] = k1
    [W[i,i+1] = k1 for i in 1:m]
    [W[i,i] = -k1 for i in 1:m]
    [W[i,i+1] = k0 for i in (m+1):(N-1)]
    [W[i,i] = -k0 for i in (m+1):N]
    W[N,1] = k0

    l1
    l2
    Lambda = [ones(m)*l1; ones(n)*l2]
    A = diagm(Lambda)

    E = ones(N,1)
    W0 = copy(W)
    W0[:,N] = ones(N)
    Z0 = zeros(N,1)
    Z0[N] = 1
    Pi = inv(W0')*Z0
    Pi = Pi
    return([W,A,Pi])
end

function FSP_steady(m,n,T,u,v,l1,l2,N0)

    W,A,Pi= W_matrix(m,n,u,v,l1,l2)

    N = N0
    K = m+n
    R = repeat([K],N)
    B = BlockArray(undef_blocks, Matrix{Float64}, R, R)

    O_matrix = zeros(K,K)

    for i in 1:N
        for j in 1:N
            B[Block(i,j)] = O_matrix
        end
    end

    for i in 1:N
        B[Block(i,i)] = W - A
    end

    for i in 1:(N-1)
        B[Block(i+1,i)] = A
    end

    B_array = Array(B)
    P0 = zeros(N*K,1)
    P0[1:K] = ones(K,1)

    T
    P_exp = exponential!(B_array*T)*P0


    P = zeros(N,1)
    for i in 0:(N-1)
        P[i+1] = dot(Pi[:,1],P_exp[(i*K+1):(i*K+K)])
    end

    return(P)
end



function FSP_steady_infer(T,p)
    m = convert(Int,round(p[1]));n = convert(Int,round(p[2]));
    u = exp(p[3]); v = exp(p[4]); l1 = exp(p[5]); l2 = 0;
    N0 = convert(Int,p[6])
    W,A,Pi= W_matrix(m,n,u,v,l1,l2)

    N = N0
    K = m+n
    R = repeat([K],N)
    B = BlockArray(undef_blocks, Matrix{Float64}, R, R)

    O_matrix = zeros(K,K)

    for i in 1:N
        for j in 1:N
            B[Block(i,j)] = O_matrix
        end
    end

    for i in 1:N
        B[Block(i,i)] = W - A
    end

    for i in 1:(N-1)
        B[Block(i+1,i)] = A
    end

    B_array = Array(B)
    P0 = zeros(N*K,1)
    P0[1:K] = ones(K,1)

    T
    P_exp = exponential!(B_array*T)*P0


    P = zeros(N,1)
    for i in 0:(N-1)
        P[i+1] = dot(Pi[:,1],P_exp[(i*K+1):(i*K+K)])
    end

    return(P)
end


function nascentRNA_noise(m,n,T,u,v)
    #m=1; n=5
    N = m + n
    W = zeros(N,N)
    k1 = u*m; k0=v*n
    [W[i,i+1] = k1 for i in 1:m]
    [W[i,i] = -k1 for i in 1:m]
    [W[i,i+1] = k0 for i in (m+1):(N-1)]
    [W[i,i] = -k0 for i in (m+1):N]
    W[N,1] = k0

    E = ones(N,1)
    W0 = copy(W)
    W0[:,N] = ones(N)
    Z0 = zeros(N,1)
    Z0[N] = 1

    l1 = 20
    l2 = 0
    Lambda = [ones(m)*l1; ones(n)*l2]
    A = diagm(Lambda)
    #T = 1
    ##
    Pi = inv(W0')*Z0
    Pi = Pi'

    IV = inv(W+E*Pi)
    M = Pi*A*E*T
    FF = 1 .+ 2*((Pi*A*E)^2 - Pi*A*IV*A*E)./(Pi*A*E) .+
    Pi*A*(exponential!(W*T)- diagm(ones(N)))*IV^2*A*E*2/(Pi*A*E*T)
    fano = FF[1]
    noise = fano/M[1]
    Mean = M[1]
    return([fano,noise,Mean])
end
