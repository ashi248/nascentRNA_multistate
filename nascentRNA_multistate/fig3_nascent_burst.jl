using BlockArrays
using LinearAlgebra
using ExponentialUtilities

include("nascentRNA_function.jl")
N0=30

u = 3; v = 1; l1 = 20; l2 = 0
prob1 = FSP_steady(1,1,1,u,v,l1,l2,N0)
prob2 = FSP_steady(1,2,1,u,v,l1,l2,N0)
prob3 = FSP_steady(1,10,1,u,v,l1,l2,N0)

bins = collect(0:N0-1)
Plots.plot(bins, prob1,size = (500,400),xlims = [0,15], dpi=600, linewidth = 4,
grid=0,framestyle=:box,label = "L1=1,L2=1",labelfontsize=18,tickfontsize = 14,
legendfontsize=14, color = :red,
xlabel = "Nascent RNA",ylabel = "Probability")

Plots.plot!(bins, prob2, linewidth = 4,
grid=0,framestyle=:box,label = "L1=1,L2=2",
color = :blue)

Plots.plot!(bins, prob3, linewidth = 4,
grid=0,framestyle=:box,label = "L1=1,L2=10",
color = :green3)

Plots.savefig("figure1/nascent_burst1.png")


L0 = 10
P0 = zeros(L0)
P1 = zeros(L0)

for i in 1:L0
    prob1 = FSP_steady(1,i,1,u,v,l1,l2,N0)
    P0[i] = prob1[1]
    P1[i] = prob1[2]
end

bins = 1:L0
Plots.plot(bins, P0,size = (500,400), dpi=600, linewidth = 4,
grid=0,framestyle=:box,label = "P(n=0)",labelfontsize=18,tickfontsize = 14,
legendfontsize=14, markershape = :circle,markersize = 9,color = :blue,
xlabel = "L2",ylabel = "Probability")

Plots.plot!(bins, P1, linewidth = 4,
grid=0,framestyle=:box,label = "P(n=1)",
markershape = :circle,markersize = 9,color = :red)

Plots.savefig("figure1/nascent_burst1_P0.png")

##

u = 3; v = 1; l1 = 20; l2 = 0
prob1 = FSP_steady(1,1,1,u,v,l1,l2,N0)
prob2 = FSP_steady(2,1,1,u,v,l1,l2,N0)
prob3 = FSP_steady(10,1,1,u,v,l1,l2,N0)

bins = collect(0:N0-1)
Plots.plot(bins, prob1,size = (500,400),xlims = [0,15], dpi=600, linewidth = 4,
grid=0,framestyle=:box,label = "L1=1,L2=1",labelfontsize=18,tickfontsize = 14,
legendfontsize=14, color = :red,
xlabel = "Nascent RNA",ylabel = "Probability")

Plots.plot!(bins, prob2, linewidth = 4,
grid=0,framestyle=:box,label = "L1=2,L2=1",
color = :blue)

Plots.plot!(bins, prob3,linewidth = 4,
grid=0,framestyle=:box,label = "L1=10,L2=1",
color = :green3)

Plots.savefig("figure1/nascent_burst2.png")

##

L0 = 10
P0 = zeros(L0)
P1 = zeros(L0)

for i in 1:L0
    prob1 = FSP_steady(i,1,1,u,v,l1,l2,N0)
    P0[i] = prob1[1]
    P1[i] = prob1[2]
end

bins = 1:L0
Plots.plot(bins, P0,size = (500,400), dpi=600, linewidth = 4,ylims = [0.0,0.5],
grid=0,framestyle=:box,label = "P(n=0)",labelfontsize=18,tickfontsize = 14,
legendfontsize=14,  markershape = :circle,markersize = 9,
xlabel = "L1",ylabel = "Probability",color = :blue)

Plots.plot!(bins, P1, linewidth = 4,
grid=0,framestyle=:box,label = "P(n=1)",
markershape = :circle,markersize = 9,color = :red)

Plots.savefig("figure1/nascent_burst2_P0.png")
