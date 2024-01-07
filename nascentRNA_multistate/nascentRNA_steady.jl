
# transition matrix for promter switching
using LinearAlgebra
using InverseLaplace
using Plots

##
function multi_step_model(m,n,N0,T)
      #m = 1; n = 1
      N = m + n
      W = zeros(N,N)
      k1 = 1*m; k0=1*n
      W[1,1] = -k1; W[1,2] = k1
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
      l2 = 4
      Lambda = [ones(m)*l1; ones(n)*l2]
      A = diagm(Lambda)

      ##
      Pi = inv(W0')*Z0
      Pi = Pi'
      U = W - A

      function Laplace_distri(n,s)
            I0 = inv(s*diagm(ones(N))-U)
            LP = Pi*(I0*A)^n*I0*E
            return(LP[1])
      end

      #N0 = 40
      prob = zeros(N0)
      for i in 1:N0
            print(i)
            PLS(s) = Laplace_distri(i-1,s)
            ft = GWR(s -> PLS(s),20)
            prob[i] = ft(T)
      end

      return(prob)
end

##
N0 = 60
T = 1
prob = multi_step_model(1,5,N0,T)
bins = collect(0:N0-1)
Plots.plot(bins, prob,size = (500,400),xlims = [0,40], dpi=600, linewidth = 4,
grid=0,framestyle=:box,label = "L=1",labelfontsize=14,tickfontsize = 12,
legendfontsize=12,
xlabel = "Nascent RNA number",ylabel = "Probability")


# transition matrix for muli-OFF and multi-On promter switching
##


T = 1
N0 = 40
prob1 = multi_step_model(1,1,N0,T)
prob2 = multi_step_model(5,1,N0,T)
prob3 = multi_step_model(10,1,N0,T)

bins = collect(0:N0-1);
Plots.plot(bins, prob1,size = (500,400),xlims = [0,40], dpi=600, linewidth = 4,
grid=0,framestyle=:box,label = "L=1",labelfontsize=14,tickfontsize = 12,
legendfontsize=12,
xlabel = "Nascent RNA number",ylabel = "Probability")

Plots.plot!(bins, prob2,label = "L=5",linewidth = 4,
legendfontsize=12,tickfontsize = 12)

Plots.plot!(bins, prob3,label = "L=10",linewidth = 4,
legendfontsize=12,tickfontsize = 12)

Plots.savefig("figure1/multi_on_1.png")

##
prob1 = multi_step_model(1,1,N0)
prob2 = multi_step_model(1,5,N0)
prob3 = multi_step_model(1,10,N0)

bins = collect(0:N0-1);
Plots.plot(bins, prob1,size = (500,400),xlims = [0,40], dpi=600, linewidth = 4,
grid=0,framestyle=:box,label = "L=1",labelfontsize=14,tickfontsize = 12,
legendfontsize=12,
xlabel = "Nascent RNA number",ylabel = "Probability")

Plots.plot!(bins, prob2,label = "L=5",linewidth = 4,
legendfontsize=12,tickfontsize = 12)

Plots.plot!(bins, prob3,label = "L=10",linewidth = 4,
legendfontsize=12,tickfontsize = 12)

Plots.savefig("figure1/multi_off_1.png")

##

function multi_step_model(m,n,N0)
      #m = 1; n = 1
      N = m + n
      W = zeros(N,N)
      k1 = 2*m; k0=4*n
      W[1,1] = -k1; W[1,2] = k1
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

      ##
      Pi = inv(W0')*Z0
      Pi = Pi'
      U = W - A

      function Laplace_distri(n,s)
            I0 = inv(s*diagm(ones(N))-U)
            LP = Pi*(I0*A)^n*I0*E
            return(LP[1])
      end

      #N0 = 40
      prob = zeros(N0)
      for i in 1:N0
            print(i)
            PLS(s) = Laplace_distri(i-1,s)
            ft = GWR(s -> PLS(s),20)
            prob[i] = ft(1)
      end

      return(prob)
end

N0 = 40
prob1 = multi_step_model(1,1,N0)
prob2 = multi_step_model(5,1,N0)
prob3 = multi_step_model(10,1,N0)

bins = collect(0:N0-1);
Plots.plot(bins, prob1,size = (500,400),xlims = [0,40], dpi=600, linewidth = 4,
grid=0,framestyle=:box,label = "L=1",labelfontsize=14,tickfontsize = 12,
legendfontsize=12,
xlabel = "Nascent RNA number",ylabel = "Probability")

Plots.plot!(bins, prob2,label = "L=5",linewidth = 4,
legendfontsize=12,tickfontsize = 12)

Plots.plot!(bins, prob3,label = "L=10",linewidth = 4,
legendfontsize=12,tickfontsize = 12)

Plots.savefig("figure1/multi_on_2.png")

##
prob1 = multi_step_model(1,1,N0)
prob2 = multi_step_model(1,5,N0)
prob3 = multi_step_model(1,10,N0)

bins = collect(0:N0-1);
Plots.plot(bins, prob1,size = (500,400),xlims = [0,40], dpi=600, linewidth = 4,
grid=0,framestyle=:box,label = "L=1",labelfontsize=14,tickfontsize = 12,
legendfontsize=12,
xlabel = "Nascent RNA number",ylabel = "Probability")

Plots.plot!(bins, prob2,label = "L=5",linewidth = 4,
legendfontsize=12,tickfontsize = 12)

Plots.plot!(bins, prob3,label = "L=10",linewidth = 4,
legendfontsize=12,tickfontsize = 12)

Plots.savefig("figure1/multi_off_2.png")
