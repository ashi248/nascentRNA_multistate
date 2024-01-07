## SSA result
using DelaySSAToolkit
using Random, Distributions
using StatsBase
using Interpolations
using Plots
using JLD2

function generate_data_mature(u = 1, v = 1, l1 = 20, l2 = 0)
    #u = 1; v = 1; k1 = 20; k2 = 0;
    rates = [2*u,2*u,2*v,2*v,l1,l1,l2,l2]
    reactant_stoich = [[1=>1],[2=>1],[3=>1],[4=>1],
    [1=>1],[2=>1],[3=>1],[4=>1]]
    net_stoich = [[1=>-1,2=>1],[2=>-1,3=>1],[3=>-1,4=>1],[4=>-1,1=>1],
    [5=>1],[5=>1],[5=>1],[5=>1]]
    mass_jump = DelaySSAToolkit.MassActionJump(rates, reactant_stoich, net_stoich; scale_rates =false)
    jumpset = DelaySSAToolkit.JumpSet((),(),nothing,mass_jump)

    u0 = zeros(5)
    u0[1] = 1
    de_chan0 = [[]]
    tf = 200000.
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

jsol = generate_data_mature()

T = 200000
tt = collect(range(0,T,2*T+1))
mRNA = jsol[5,:]
nodes = (jsol.t,)
mRNA_itp = Interpolations.interpolate(nodes,mRNA, Gridded(Constant{Previous}()))
mRNA_expr = mRNA_itp(tt)[10000:2*T+1]
mRNA_prob = proportionmap(mRNA_expr)


u = 1; v = 1; l1 = 20; l2 = 0
prob1 = FSP_steady(2,2,1,u,v,l1,l2,40)

N0 = 40
bins = collect(0:N0-1)
Plots.plot(bins, prob1,size = (500,400),xlims = [0,40], dpi=600, linewidth = 5,
grid=0,framestyle=:box,label = "Theoretical",labelfontsize=18,tickfontsize = 14,
legendfontsize=14, color = :blue,
xlabel = "Nascent RNA number",ylabel = "Probability")
scatter!(mRNA_prob,shape = [:circle],markersize=8,label = "SSA")

Plots.savefig("figure1/nascent_simulation1.png")
##


u = 0.7; v = 1.5; l1 = 20; l2 = 1
prob1 = FSP_steady(2,2,1,u,v,l1,l2,40)

N0 = 40
bins = collect(0:N0-1)
Plots.plot(bins, prob1,size = (500,400),xlims = [0,40], dpi=600, linewidth = 5,
grid=0,framestyle=:box,label = "Theoretical",labelfontsize=18,tickfontsize = 14,
legendfontsize=14,color = :blue,
xlabel = "Nascent RNA number",ylabel = "Probability")



T = 200000
jsol = generate_data_mature(u,v,l1,l2)


tt = collect(range(0,T,2*T+1))
mRNA = jsol[5,:]
nodes = (jsol.t,)
mRNA_itp = Interpolations.interpolate(nodes,mRNA, Gridded(Constant{Previous}()))
mRNA_expr = mRNA_itp(tt)[100000:2*T+1]
mRNA_prob = proportionmap(mRNA_expr)
scatter!(mRNA_prob,shape = [:circle],markersize=8,label = "SSA")

Plots.savefig("figure1/nascent_simulation2.png")

##

function generate_data_mature(u = 2, v = 1, l1 = 20, l2 = 0)
    #u = 1; v = 1; k1 = 20; k2 = 0;
    rates = [1*u,3*v,3*v,3*v,l1,l2,l2,l2]
    reactant_stoich = [[1=>1],[2=>1],[3=>1],[4=>1],
    [1=>1],[2=>1],[3=>1],[4=>1]]
    net_stoich = [[1=>-1,2=>1],[2=>-1,3=>1],[3=>-1,4=>1],[4=>-1,1=>1],
    [5=>1],[5=>1],[5=>1],[5=>1]]
    mass_jump = DelaySSAToolkit.MassActionJump(rates, reactant_stoich, net_stoich; scale_rates =false)
    jumpset = DelaySSAToolkit.JumpSet((),(),nothing,mass_jump)

    u0 = zeros(5)
    u0[1] = 1
    de_chan0 = [[]]
    tf = 200000.
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

jsol = generate_data_mature()

T = 200000
tt = collect(range(0,T,2*T+1))
mRNA = jsol[5,:]
nodes = (jsol.t,)
mRNA_itp = Interpolations.interpolate(nodes,mRNA, Gridded(Constant{Previous}()))
mRNA_expr = mRNA_itp(tt)[10000:2*T+1]
mRNA_prob = proportionmap(mRNA_expr)


u = 2; v = 1; l1 = 20; l2 = 0
prob1 = FSP_steady(1,3,1,u,v,l1,l2,40)

N0 = 40
bins = collect(0:N0-1)
Plots.plot(bins, prob1,size = (500,400),xlims = [0,40], dpi=600, linewidth = 5,
grid=0,framestyle=:box,label = "Theoretical",labelfontsize=18,tickfontsize = 14,
legendfontsize=14,color = :blue,
xlabel = "Nascent RNA number",ylabel = "Probability")
scatter!(mRNA_prob,shape = [:circle],markersize=8,label = "SSA")

Plots.savefig("figure1/nascent_simulation3.png")
