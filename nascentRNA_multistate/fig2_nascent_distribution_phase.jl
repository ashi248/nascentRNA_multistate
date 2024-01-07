
include("nascentRNA_function.jl")

uu = 10 .^(collect(range(log10(0.1),log10(10),300)))
vv = 10 .^(collect(range(log10(0.1),log10(10),300)))

FF = zeros(length(uu),length(vv))
m = 5; n=1; T=1
for i in 1:length(uu)
    print(i)
    for j in 1:length(vv)
        u = uu[i]; v = vv[j]
        FF[i,j] = nascentRNA_noise(m,n,T,u,v)[1]
    end
end

using Plots
pyplot()
Plots.heatmap(uu,vv, FF',c =:viridis,xlabel = "u",
ylabel = "v",yscale =:log, xscale =:log,framestyle=:box,
xlims = [0.1,10],ylims = [0.1,10],labelfontsize=14,dpi=600,
tickfontsize = 12,legendfontsize = 14,
colorbar_title = "Fanor factor",colorbar_titlefontsize = 14,
colorbar_tickfontsize=12)


include("find_peak.jl")
type = zeros(length(uu),length(vv))
l1 =20;l2 = 0; N0=40

for i in 1:length(uu)
    print(i)
    for j in 1:length(vv)
        u = uu[i]; v = vv[j]
        probs = FSP_steady(m,n,T,u,v,l1,l2,N0)
        type[i,j] = find_peak_nascent(probs)
    end
end


using Plots
pyplot()

Plots.contour!(uu,vv, type',xlabel = "u",color=[:white],
ylabel = "v",yscale =:log, xscale =:log,framestyle=:box,
xlims = [0.1,10],ylims = [0.1,10],labelfontsize=18,dpi=600,
tickfontsize = 14,legendfontsize = 14,linewidth=2,
colorbar_title = "Fanor factor",colorbar_titlefontsize = 14,
colorbar_tickfontsize=12)


Plots.savefig("figure1/phase_diagram_m5.png")


##

include("nascentRNA_function.jl")
N0=40

u = 0.5; v = 1.5; l1 = 20; l2 = 0
prob1 = FSP_steady(1,1,1,u,v,l1,l2,N0)

N0 = 40
bins = collect(0:N0-1)
Plots.plot(bins, prob1,size = (500,400),xlims = [0,40], dpi=600, linewidth = 4,
grid=0,framestyle=:box,label = "L1=1,L2=1",labelfontsize=18,tickfontsize = 14,
fillrange = 0, fillalpha = 0.25,legend = :none,color = :blue,
xlabel = "Nascent RNA",ylabel = "Probability")

Plots.savefig("figure1/phase_diagram_regime3.png")
##

u = 12; v = 2; l1 = 20; l2 = 0
prob1 = FSP_steady(1,1,1,u,v,l1,l2,N0)

N0 = 40
bins = collect(0:N0-1)
Plots.plot(bins, prob1,size = (500,400),xlims = [0,20], dpi=600, linewidth = 4,
grid=0,framestyle=:box,label = "L1=1,L2=1",labelfontsize=18,tickfontsize = 14,
fillrange = 0, fillalpha = 0.25,legend = :none,color = :blue,
xlabel = "Nascent RNA",ylabel = "Probability")

Plots.savefig("figure1/phase_diagram_regime1.png")

##

u = 1; v = 10; l1 = 20; l2 = 0
prob1 = FSP_steady(1,1,1,u,v,l1,l2,N0)

N0 = 40
bins = collect(0:N0-1)
Plots.plot(bins, prob1,size = (500,400),xlims = [0,40], dpi=600, linewidth = 4,
grid=0,framestyle=:box,label = "L1=1,L2=1",labelfontsize=18,tickfontsize = 14,
legendfontsize=12, fillrange = 0, fillalpha = 0.25,legend = :none,color = :blue,
xlabel = "Nascent RNA",ylabel = "Probability")

Plots.savefig("figure1/phase_diagram_regime2.png")

##

u = 0.8; v = 1.2; l1 = 20; l2 = 0
prob1 = FSP_steady(1,5,1,u,v,l1,l2,N0)

N0 = 40
bins = collect(0:N0-1)
Plots.plot(bins, prob1,size = (500,400),xlims = [0,40], dpi=600, linewidth = 4,
grid=0,framestyle=:box,label = "L1=1,L2=1",labelfontsize=18,tickfontsize = 14,
legendfontsize=12, fillrange = 0, fillalpha = 0.25,legend = :none,color = :blue,
xlabel = "Nascent RNA",ylabel = "Probability")

Plots.savefig("figure1/phase_diagram_regime4.png")
