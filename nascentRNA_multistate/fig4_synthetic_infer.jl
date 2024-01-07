## SSA result
using DelaySSAToolkit
using Random, Distributions
using StatsBase
using Interpolations
using Plots
using JLD2
using BlackBoxOptim
include("nascentRNA_function.jl")


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
function generate_data_mature(u = 1, v = 1, l1 = 20, l2 = 0)
    #u = 1; v = 1; k1 = 20; k2 = 0;
    m = 3;n = 3
    rates = [m*u,m*u,m*u,n*v,n*v,n*v,l1,l1,l1,l2,l2,l2]
    reactant_stoich = [[1=>1],[2=>1],[3=>1],[4=>1],[5=>1],[6=>1],
    [1=>1],[2=>1],[3=>1],[4=>1],[5=>1],[6=>1]]
    net_stoich = [[1=>-1,2=>1],[2=>-1,3=>1],[3=>-1,4=>1],[4=>-1,5=>1],
    [5=>-1,6=>1],[6=>-1,1=>1],[7=>1],[7=>1],[7=>1],[7=>1],[7=>1],[7=>1]]
    mass_jump = DelaySSAToolkit.MassActionJump(rates, reactant_stoich, net_stoich; scale_rates =false)
    jumpset = DelaySSAToolkit.JumpSet((),(),nothing,mass_jump)

    u0 = zeros(7)
    u0[1] = 1
    de_chan0 = [[]]
    tf = 200.
    tspan = (0,tf)
    dprob = DiscreteProblem(u0, tspan)
    delay_trigger_affect! = function (integrator, rng)
        τ=1
        append!(integrator.de_chan[1], τ)
    end
    delay_trigger = Dict(7=>delay_trigger_affect!,8=>delay_trigger_affect!,
    9=>delay_trigger_affect!,10=>delay_trigger_affect!,
    11=>delay_trigger_affect!,12=>delay_trigger_affect!)

    delay_complete = Dict(1=>[7=>-1])
    delay_interrupt = Dict()
    delayjumpset = DelayJumpSet(delay_trigger,delay_complete,delay_interrupt)

    djprob = DelayJumpProblem(dprob, DelayRejection(), jumpset, delayjumpset,
     de_chan0, save_positions=(true,true))
    sol1 = solve(djprob, SSAStepper())
    return(sol1)
end

N = 10000
tt = collect(range(0,100,101))

state = 1
u = 3; v = 1; l1=20; l2 = 0
mRNA_expr = zeros(N)
for i in 1:N
  print(i)
  jsol = generate_data_mature(u,v,l1,l2)
  nodes = (jsol.t,)
  mRNA = jsol[7,:]
  mRNA_itp = Interpolations.interpolate(nodes,mRNA, Gridded(Constant{Previous}()))
  mRNA0 = mRNA_itp(tt)
  mRNA_expr[i] = mRNA0[length(mRNA0)]
end



params = zeros(200,5)
for k = 1:200
    print(k)
    mRNA_sample = sample(mRNA_expr[1:2000],2000, replace = true)
    df1 = Int.(round.(mRNA_sample))
    cost_function(p) = Log_Likelihood_nascent(df1, p)
    bound1 = Tuple{Float64, Float64}[(1, 5),(1, 5),(log(0.1), log(10)),(log(0.1), log(10)),(log(1), log(100))]
    result = bboptimize(cost_function;SearchRange = bound1,MaxSteps = 5000)
    results1 =  result.archive_output.best_candidate
    m = Int(round(results1[1]))
    n = Int(round(results1[2]))
    a1 = exp(results1[3])
    a2 = exp(results1[4])
    a3 = exp(results1[5])
    params[k,:] = [m,n,a1,a2,a3]
end

save_object("data/infer_nascent.jld2",params)

params2 = load_object("data/infer_nascent.jld2")


fs = sort(countmap(params2[:,1]))
fig1 = bar(collect(fs.keys),fs.vals ./sum(fs.vals),color = "yellow", xlims = [0.5,5.5],
bar_edges = true, bar_width = 0.5,ylabel = "Probability", label = "",
xlabel = "L1",labelfontsize=15,tickfontsize = 12,legendfontsize=12,
grid=0,framestyle=:box,legend = (0.1,0.8))
vline!([3], lw = 5, color = "red",label= "True value")


fs = sort(countmap(params2[:,2]))
fig2 = bar(collect(fs.keys),fs.vals ./sum(fs.vals),color = "yellow", xlims = [0.5,5.5],
bar_edges = true, bar_width = 0.5,ylabel = "Probability",label = "",
xlabel = "L2",labelfontsize=15,tickfontsize = 12,legendfontsize=12,
grid=0,framestyle=:box)
vline!([3],lw = 5, color = "red",label= "")

plot(fig1,fig2,layout = (2, 1),size = (500,400),
grid = false, fontfamily="Helvetica")


Plots.savefig("figure1/nascent_estimated_1.png")



##
param1 = [params2[:,3];params2[:,4];params2[:,5]]
boxplot(repeat(["u","v","k1"],inner=200),param1,yscale = :log,
label = "",color = :yellow,linewidth=2,size = (500,400),ylabel = "Estimated value",
labelfontsize=18,tickfontsize = 14,legendfontsize=12,
grid=0,framestyle=:box,fontfamily="Helvetica")
scatter!([0.5,1.5,2.5],[20,3,1],markersize=8,markercolor = :red,
label = "True value")
Plots.savefig("figure1/nascent_estimated_2.png")


# error bar
params2 = load_object("data/infer_nascent.jld2")
p3 = params2[:,3]; p4 = params2[:,4]; p5 = params2[:,5]

a = ["u","v","k1"]
med_p = [median(p3);median(p4);median(p5)]
min_p = [quantile(p3,0.25);quantile(p4,0.25);quantile(p5,0.25)]
max_p = [quantile(p3,0.975);quantile(p4,0.975);quantile(p5,0.975)]

err1 = med_p .-min_p
err2 = max_p .-med_p

scatter(a, med_p,yerror=(err1,err2),msw = 2, ms=7,color = :yellow,
yscale = :log,xlim = [0,3],label = "Median",
linewidth=2,size = (500,400),labelfontsize=18,tickfontsize = 16,legendfontsize=14,
grid=0,framestyle=:box,ylabel = "Estimated value")

scatter!([0.5,1.5,2.5],[3,1,20],markersize=8,markercolor = :red,
label = "True value",legend = (0.1,0.9))

Plots.savefig("figure1/nascent_estimated_2.png")

##
mRNA_sample = mRNA_expr[1:2000]
df1 = Int.(round.(mRNA_sample))
cost_function(p) = Log_Likelihood_nascent(df1, p)
bound1 = Tuple{Float64, Float64}[(1, 5),(1, 5),(log(0.1), log(10)),(log(0.1), log(10)),(log(1), log(100))]
result = bboptimize(cost_function;SearchRange = bound1,MaxSteps = 10000)
results1 =  result.archive_output.best_candidate
m = Int(round(results1[1]))
n = Int(round(results1[2]))
a1 = exp(results1[3])
a2 = exp(results1[4])
a3 = exp(results1[5])

mRNA_prob = proportionmap(mRNA_sample)
bar(mRNA_prob,label = "Synthetic data",color = "skyblue")

prob1 = FSP_steady(m,n,1,a1,a2,a3,0,30)
N0 = 30
bins = collect(0:N0-1)
Plots.plot!(bins, prob1,size = (500,400),xlims = [0,20], dpi=600, linewidth = 5,
grid=0,framestyle=:box,label = "cyclic model",labelfontsize=18,tickfontsize = 14,
legendfontsize=14, color = :red,
xlabel = "Nascent RNA",ylabel = "Probability")


#--------------------
bound1 = Tuple{Float64, Float64}[(1, 1),(1, 1),(log(0.1), log(10)),(log(0.1), log(10)),(log(1), log(100))]
result = bboptimize(cost_function;SearchRange = bound1,MaxSteps = 10000)
results1 =  result.archive_output.best_candidate

m = Int(round(results1[1]))
n = Int(round(results1[2]))
a1 = exp(results1[3])
a2 = exp(results1[4])
a3 = exp(results1[5])

prob1 = FSP_steady(m,n,1,a1,a2,a3,0,30)
N0 = 30
bins = collect(0:N0-1)
Plots.plot!(bins, prob1,size = (500,400),dpi=600, linewidth = 5,
grid=0,framestyle=:box,label = "2-state model",labelfontsize=18,tickfontsize = 14,
legendfontsize=14, color = :blue,ls = :dash)

Plots.savefig("figure1/nascent_simulation_infer1.png")


##############################
##############################
function generate_data_mature(u = 1, v = 1, l1 = 20, l2 = 0)
    #u = 1; v = 1; k1 = 20; k2 = 0;
    m = 2;n=2
    rates = [m*u,m*u,n*v,n*v,l1,l1,l2,l2]
    reactant_stoich = [[1=>1],[2=>1],[3=>1],[4=>1],
    [1=>1],[2=>1],[3=>1],[4=>1]]
    net_stoich = [[1=>-1,2=>1],[2=>-1,3=>1],[3=>-1,4=>1],[4=>-1,1=>1],
    [5=>1],[5=>1],[5=>1],[5=>1]]
    mass_jump = DelaySSAToolkit.MassActionJump(rates, reactant_stoich, net_stoich; scale_rates =false)
    jumpset = DelaySSAToolkit.JumpSet((),(),nothing,mass_jump)

    u0 = zeros(5)
    u0[1] = 1
    de_chan0 = [[]]
    tf = 200.
    tspan = (0,tf)
    dprob = DiscreteProblem(u0, tspan)
    delay_trigger_affect! = function (integrator, rng)
        τ=1
        append!(integrator.de_chan[1], τ)
    end
    delay_trigger = Dict(5=>delay_trigger_affect!,6=>delay_trigger_affect!,
    7=>delay_trigger_affect!,8=>delay_trigger_affect!)
    delay_complete = Dict(1=>[5=>-1])
    delay_interrupt = Dict()
    delayjumpset = DelayJumpSet(delay_trigger,delay_complete,delay_interrupt)

    djprob = DelayJumpProblem(dprob, DelayRejection(), jumpset, delayjumpset,
     de_chan0, save_positions=(true,true))
    sol1 = solve(djprob, SSAStepper())
    return(sol1)
end

N = 6000
tt = collect(range(0,100,101))

# state = 1
state = 1
u = 3; v = 1; l1=20; l2 = 0
mRNA_expr = zeros(N)
for i in 1:N
  print(i)
  jsol = generate_data_mature(u,v,l1,l2)
  nodes = (jsol.t,)
  mRNA = jsol[5,:]
  mRNA_itp = Interpolations.interpolate(nodes,mRNA, Gridded(Constant{Previous}()))
  mRNA0 = mRNA_itp(tt)
  mRNA_expr[i] = mRNA0[length(mRNA0)]
end


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

df1 = Int.(round.(mRNA_expr))
cost_function(p) = Log_Likelihood_nascent(df1, p)


bound1 = Tuple{Float64, Float64}[(1, 5),(1, 5),(log(0.1), log(10)),(log(0.1), log(10)),(log(1), log(100))]
result = bboptimize(cost_function;SearchRange = bound1,MaxSteps = 20000)
results2 =  result.archive_output.best_candidate


# 5-element Vector{Float64}:
#   2.356947341035986
#   1.7877094703333136
#   1.1351533635457198
#  -0.000900093379879678
#   3.044539581585897

m = Int(round(results2[1]))
n = Int(round(results2[2]))
a1 = exp(results2[3])
a2 = exp(results2[4])
a3 = exp(results2[5])


mRNA_prob = proportionmap(mRNA_expr)
bar(mRNA_prob,label = "Synthetic data",color = "skyblue")

N0 = 20
prob1 = FSP_steady(m,n,1,a1,a2,a3,0,N0)

bins = collect(0:N0-1)
Plots.plot!(bins, prob1,size = (500,400),xlims = [0,N0], dpi=600, linewidth = 3,
grid=0,framestyle=:box,label = "Model fitting",labelfontsize=16,tickfontsize = 12,
legendfontsize=12, color = :red,
xlabel = "Nascent RNA number",ylabel = "Probability")

Plots.savefig("figure1/nascent_simulation_infer2.png")

############################
##############################
function generate_data_mature(u = 1, v = 1, l1 = 20, l2 = 0)
    #u = 1; v = 1; k1 = 20; k2 = 0;
    m = 1;n=3
    rates = [m*u,n*u,n*v,n*v,l1,l2,l2,l2]
    reactant_stoich = [[1=>1],[2=>1],[3=>1],[4=>1],
    [1=>1],[2=>1],[3=>1],[4=>1]]
    net_stoich = [[1=>-1,2=>1],[2=>-1,3=>1],[3=>-1,4=>1],[4=>-1,1=>1],
    [5=>1],[5=>1],[5=>1],[5=>1]]
    mass_jump = DelaySSAToolkit.MassActionJump(rates, reactant_stoich, net_stoich; scale_rates =false)
    jumpset = DelaySSAToolkit.JumpSet((),(),nothing,mass_jump)

    u0 = zeros(5)
    u0[1] = 1
    de_chan0 = [[]]
    tf = 200.
    tspan = (0,tf)
    dprob = DiscreteProblem(u0, tspan)
    delay_trigger_affect! = function (integrator, rng)
        τ=1
        append!(integrator.de_chan[1], τ)
    end
    delay_trigger = Dict(5=>delay_trigger_affect!,6=>delay_trigger_affect!,
    7=>delay_trigger_affect!,8=>delay_trigger_affect!)
    delay_complete = Dict(1=>[5=>-1])
    delay_interrupt = Dict()
    delayjumpset = DelayJumpSet(delay_trigger,delay_complete,delay_interrupt)

    djprob = DelayJumpProblem(dprob, DelayRejection(), jumpset, delayjumpset,
     de_chan0, save_positions=(true,true))
    sol1 = solve(djprob, SSAStepper())
    return(sol1)
end

N = 6000
tt = collect(range(0,100,101))

# state = 1
state = 1
u = 1; v = 1; l1=20; l2 = 0
mRNA_expr = zeros(N)
for i in 1:N
  print(i)
  jsol = generate_data_mature(u,v,l1,l2)
  nodes = (jsol.t,)
  mRNA = jsol[5,:]
  mRNA_itp = Interpolations.interpolate(nodes,mRNA, Gridded(Constant{Previous}()))
  mRNA0 = mRNA_itp(tt)
  mRNA_expr[i] = mRNA0[length(mRNA0)]
end


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

df1 = Int.(round.(mRNA_expr))
cost_function(p) = Log_Likelihood_nascent(df1, p)


bound1 = Tuple{Float64, Float64}[(1, 5),(1, 5),(log(0.1), log(10)),(log(0.1), log(10)),(log(1), log(100))]
result = bboptimize(cost_function;SearchRange = bound1,MaxSteps = 20000)
results3 =  result.archive_output.best_candidate


# 5-element Vector{Float64}:
#  1.0058733715199748
#  3.0946449972324412
#  0.0174359956876318
#  0.00169714083164312
#  2.9941989392552797

m = Int(round(results3[1]))
n = Int(round(results3[2]))
a1 = exp(results3[3])
a2 = exp(results3[4])
a3 = exp(results3[5])


mRNA_prob = proportionmap(mRNA_expr)
bar(mRNA_prob,label = "Synthetic data",color = "skyblue")

N0 = 30
prob1 = FSP_steady(m,n,1,a1,a2,a3,0,N0)

bins = collect(0:N0-1)
Plots.plot!(bins, prob1,size = (500,400),xlims = [0,N0], dpi=600, linewidth = 3,
grid=0,framestyle=:box,label = "Model fitting",labelfontsize=15,tickfontsize = 12,
legendfontsize=12, color = :red,
xlabel = "Nascent RNA number",ylabel = "Probability")

Plots.savefig("figure1/nascent_simulation_infer3.png")
