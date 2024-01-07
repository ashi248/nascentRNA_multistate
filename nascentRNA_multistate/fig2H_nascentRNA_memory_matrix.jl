using BlockArrays
using LinearAlgebra
using ExponentialUtilities
using Plots

include("nascentRNA_function.jl")

function FPS_plot(m,n)
    N0=60
    u = 0.8; v = 1.5; l1 = 40; l2 = 0
    prob = FSP_steady(m,n,1,u,v,l1,l2,N0)

    bins = collect(0:N0-1)
    fig = plot(bins, prob,size = (500,400),
    xlims = [0,60],dpi=1000, legend = :none,color = :blue,
    grid=0,fillrange = 0, fillalpha = 0.25,
    framestyle=:box)
    return(fig)
end

p1 = FPS_plot(1,1)
p2 = FPS_plot(1,5)
p3 = FPS_plot(1,15)

p4 = FPS_plot(5,1)
p5 = FPS_plot(5,5)
p6 = FPS_plot(5,15)

p7 = FPS_plot(15,1)
p8 = FPS_plot(15,5)
p9 = FPS_plot(15,15)

plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,layout=9,linewidth=3)

Plots.savefig("figure1/memo_to_distri_matrix_plot.png")

##


#
# N0=60
# m = 2; n = 30
# u = 0.8; v = 1.5; l1 = 40; l2 = 0
# prob = FSP_steady(m,n,1,u,v,l1,l2,N0)
# bins = collect(0:N0-1)
#
#
# fig = plot(bins, prob,size = (500,400),
# xlims = [0,60],dpi=1000, linewidth = 4,
# grid=0,framestyle=:box)
#
# bimodal_strength(m,n)
#
# ##
# using LaTeXStrings
#
# function bimodal_strength(m,n,u=1/2,v = 1/0.7,l1 = 40,l2 = 0)
#     N0=60
#     #u; v; l1 = 40; l2 = 2
#     prob = FSP_steady(m,n,1,u,v,l1,l2,N0)
#     bins = collect(0:N0-1)
#
#     diff1 = sign.(diff(vec(prob)))
#     Lmax = diff(diff1) .== -2
#     L0= ifelse(diff1[1]<0,1,0)
#     Lmax = [L0;Lmax;0]
#     peak = prob[Lmax .== 1]
#
#     Lmin = diff(diff1) .== 2
#     Lmin = [0;Lmin;0]
#     low = prob[Lmin .== 1]
#     if length(low) == 0
#         low = minimum(peak)
#     end
#     k = (minimum(peak) - low[1])/maximum(peak)
#     return(k)
# end
#
#
# M = 1:10
# n = 30;
# #
# for i in 1:10
#     print(i)
#     m = M[i]
#     k[i] = bimodal_strength(m,n,1/1.7,1/0.7,40,0)
# end
#
# plot(1:10, k,size = (500,400),dpi=600, linewidth = 4,label = L"\tau_{on}=1.7",
# grid=0,framestyle=:box,markershape=:circle,markersize = 6,
# legendfontsize=14,tickfontsize = 12,labelfontsize=14,
# xlabel = "L1",ylabel = "Bimodality strength",)
#
#
# #
# for i in 1:10
#     print(i)
#     m = M[i]
#     k[i] = bimodal_strength(m,n,1/2,1/0.7,40,0)
# end
#
# plot!(1:10, k,size = (500,400),dpi=600, linewidth = 4,label = L"\tau_{on}=2",
# grid=0,framestyle=:box,markershape=:circle,markersize = 6,
# legendfontsize=14,tickfontsize = 12,labelfontsize=14,)
#
# Plots.savefig("figure1/bimodal_strength1.png")
#
#
#
#
# ##
#
# M = 1:10
# n = 1;
# k1 = zeros(10)
# for i in 1:10
#     print(i)
#     m = M[i]
#     k[i] = bimodal_strength(m,n,1/1.7,1/0.7,40,0)
# end
#
# plot(1:10, k,size = (500,400),dpi=600, linewidth = 4,label = L"\tau_{on}=1.7",
# grid=0,framestyle=:box,markershape=:circle,markersize = 6,
# legendfontsize=14,tickfontsize = 12,labelfontsize=14,
# xlabel = "L1",ylabel = "Bimodality strength",ylim = [0.35,0.55])
#
# for i in 1:10
#     print(i)
#     m = M[i]
#     k[i] = bimodal_strength(m,n,1/2,1/0.7,40,0)
# end
#
# plot!(1:10, k,size = (500,400),dpi=600, linewidth = 4,label = L"\tau_{on}=2",
# grid=0,framestyle=:box,markershape=:circle,markersize = 6,
# legendfontsize=14,tickfontsize = 12,labelfontsize=14,
# xlabel = "L2",ylabel = "Bimodality strength",)
#
# Plots.savefig("figure1/bimodal_strength2.png")
