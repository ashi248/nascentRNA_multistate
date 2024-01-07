##using  LinearAlgebra
using BAT, DensityInterface, IntervalSets
using ValueShapes
using BlackBoxOptim
using Plots
using Statistics
include("nascentRNA_function.jl")

using XLSX
using DataFrames
df = DataFrame(XLSX.readtable("data/HIV-1.xlsx","Sheet1"))
df1 = Int.(round.(df))


function Log_Likelihood_nascent(data,p)
      id = data[:,1] .+1
      N = maximum(id)
      p0 = [p;N]
      T = 1;
      prob = FSP_steady_infer(T,p0)
      prob = abs.(prob)
      tot_loss = -sum(log.(prob[id]))
      return (tot_loss)
end




##
cost_function(p) = Log_Likelihood_nascent(df1, p)
bound1 = Tuple{Float64, Float64}[(1, 5),(1, 5),(log(0.1), log(10)),(log(0.1), log(10)),(log(150), log(300))]
result = bboptimize(cost_function;SearchRange = bound1,MaxSteps = 500)
results1 =  result.archive_output.best_candidate


m = Int(round(results1[1]))
n = Int(round(results1[2]))
a1 = exp(results1[3])
a2 = exp(results1[4])
a3 = exp(results1[5])

N0 = maximum(df1[:,1]) + 1
nas_freq = FSP_steady(m,n,1,a1,a2,a3,0,N0)


histogram(df1[:,1],bins=0:1:160,normalize=:probability,
color = "skyblue",labels="data",labelfontsize=14,dpi=600,legendfontsize = 12,
ylims = [0,0.02],
tickfontsize = 12,grid=0,size = (500,400))

bins = collect(0:N0-1)
Plots.plot!(bins, nas_freq,size = (500,400),dpi=600, linewidth = 5,color = :red,
grid=0,framestyle=:box,label = "cyclic model",labelfontsize=16,tickfontsize = 14,
legendfontsize=12,
xlabel = "Nascent RNA",ylabel = "Probability")


##
bound1 = Tuple{Float64, Float64}[(1, 1),(1, 1),(log(0.1), log(10)),(log(0.1), log(10)),(log(100), log(300))]
result = bboptimize(cost_function;SearchRange = bound1,MaxSteps = 2000)
results2 =  result.archive_output.best_candidate

m = Int(round(results2[1]))
n = Int(round(results2[2]))
a1 = exp(results2[3])
a2 = exp(results2[4])
a3 = exp(results2[5])

N0 = maximum(df1[:,1]) + 1
nas_freq = FSP_steady(m,n,1,a1,a2,a3,0,N0)

#
bins = collect(0:N0-1)
Plots.plot!(bins, nas_freq,size = (500,400),dpi=600, linewidth = 5,color = :blue,
grid=0,framestyle=:box,label = "2-state model",labelfontsize=18,tickfontsize = 12,ls = :dash,
legendfontsize=14,
xlabel = "Nascent RNA",ylabel = "Probability")

Plots.savefig("figure1/HIV-2_infer.png")

##

using StatsBase

fs = sort(countmap(df1[:,1]));
bar(collect(fs.keys),cumsum(fs.vals) ./sum(fs.vals),color = "skyblue",
legend = :none,bar_edges = true, bar_width = 1)


#
m = Int(round(results1[1]))
n = Int(round(results1[2]))
a1 = exp(results1[3])
a2 = exp(results1[4])
a3 = exp(results1[5])

N0 = maximum(df1[:,1]) + 1
nas_freq = FSP_steady(m,n,1,a1,a2,a3,0,N0)
nas_freq = nas_freq[:,1]

bins = collect(0:N0-1)
Plots.plot!(bins, cumsum(nas_freq),size = (400,400),dpi=600, linewidth = 5,color = :red,
grid=0,framestyle=:box,xlims = [0,120],labelfontsize=14,tickfontsize = 12,
xlabel = " ",ylabel = "Cumlative probability")


m = Int(round(results2[1]))
n = Int(round(results2[2]))
a1 = exp(results2[3])
a2 = exp(results2[4])
a3 = exp(results2[5])

N0 = maximum(df1[:,1]) + 1
nas_freq = FSP_steady(m,n,1,a1,a2,a3,0,N0)
nas_freq = nas_freq[:,1]

bins = collect(0:N0-1)
Plots.plot!(bins, cumsum(nas_freq),linewidth = 5,color = :blue,
grid=0,framestyle=:box,ls = :dash)


Plots.savefig("figure1/HIV-2_infer_cumulative.png")

##
# confidence interval
function Log_Likelihood_nascent(data,p)
      id = data .+1
      N = maximum(id)
      p0 = [p;N]
      T = 1;
      prob = FSP_steady_infer(T,p0)
      prob = abs.(prob)
      tot_loss = -sum(log.(prob[id]))
      return (tot_loss)
end

params = zeros(100,5)

for k in 1:100
      print(k)
      df_sample = sample(df1[:,1],1412)
      cost_function(p) = Log_Likelihood_nascent(df_sample , p)
      bound1 = Tuple{Float64, Float64}[(1, 5),(1, 5),(log(0.1), log(10)),(log(0.1), log(10)),(log(200), log(300))]
      result = bboptimize(cost_function;SearchRange = bound1,MaxSteps = 500)
      results1 =  result.archive_output.best_candidate
      m = Int(round(results1[1]))
      n = Int(round(results1[2]))
      a1 = exp(results1[3])
      a2 = exp(results1[4])
      a3 = exp(results1[5])
      params[k,:] = [m,n,a1,a2,a3]
end

save_object("data/infer_nascent_HIV.jld2",params)

load_object("data/infer_nascent_HIV.jld2")

fs = sort(countmap(params[:,1]))
fig1 = bar(collect(fs.keys),fs.vals ./sum(fs.vals),color = "yellow", xlims = [0.5,5.5],
legend = :none,bar_edges = true, bar_width = 0.5,ylabel = "Probability",
xlabel = "L1",labelfontsize=16,tickfontsize = 14,legendfontsize=12,
grid=0,framestyle=:box)


fs = sort(countmap(params[:,2]))
fig2 = bar(collect(fs.keys),fs.vals ./sum(fs.vals),color = "yellow", xlims = [0.5,5.5],
legend = :none,bar_edges = true, bar_width = 0.5,ylabel = "Probability",
xlabel = "L2",labelfontsize=16,tickfontsize = 14,legendfontsize=12,
grid=0,framestyle=:box)

plot(fig1,fig2,layout = (2, 1),legend = false, size = (500,400),
grid = false, fontfamily="Helvetica")

Plots.savefig("figure1/nascent_HIV_1.png")

# param1 = [params[:,1];params[:,2]]
# boxplot(repeat(["L1","L2"],inner=100),param1,label = "",
# color = :yellow,linewidth=2,size = (500,400),ylabel = "Estimated value",
# labelfontsize=16,tickfontsize = 14,legendfontsize=12,
# grid=0,framestyle=:box)
#
#
# Plots.savefig("figure1/nascent_HIV_1.png")


param1 = [params[:,3];params[:,4];params[:,5]]
boxplot(repeat(["u","v","k1"],inner=100),param1,yscale = :log,
label = "",color = :yellow,linewidth=2,size = (500,400),ylabel = "Estimated value",
labelfontsize=18,tickfontsize = 14,legendfontsize=14,
grid=0,framestyle=:box,fontfamily="Helvetica")


Plots.savefig("figure1/nascent_HIV_2.png")

##
#error bar
load_object("data/infer_nascent_HIV.jld2")
p3 = params[:,3]; p4 = params[:,4]; p5 = params[:,5]

a = ["u","v","k1"]
med_p = [median(p3);median(p4);median(p5)]
min_p = [quantile(p3,0.25);quantile(p4,0.25);quantile(p5,0.25)]
max_p = [quantile(p3,0.975);quantile(p4,0.975);quantile(p5,0.975)]

err1 = med_p .-min_p
err2 = max_p .-med_p

scatter(a, med_p,yerror=(err1,err2),msw = 2, ms=7,color = :yellow,
yscale = :log,xlim = [0,3],label = "Median",
linewidth=2,size = (500,400),labelfontsize=18,tickfontsize = 16,legendfontsize=14,
grid=0,framestyle=:box,ylabel = "Estimated value",legend = (0.1,0.9))


Plots.savefig("figure1/nascent_HIV_2_error.png")
