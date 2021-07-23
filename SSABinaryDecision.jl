module BinaryDecSSA

    """
    Need to more properly document and add ability to plot/output m intstead of N₊.
    """

    export BinDecMod, Ensemble, EnsProb, CentralMoments

    using Distributions, Parameters, StatsBase, LinearAlgebra, Colors

    """
    Define the Ising model struct.
    """
    @with_kw struct BinDecMod
        N::Int64 = 100
        F::Float64
        J::Float64
        α::Float64 = 0.0
        β::Float64 = 1.0
        γ::Real = 1.0 # timescale
        fixedIC::Bool = true
        p₀::Float64 = 0.5 # initial prob of spin up
        @assert α <= 1 && α >= 0 # α ∈ [0,1]
        @assert β > 0
        @assert p₀ <= 1 && p₀ >= 0 # p₀ ∈ [0,1]
    end

    """
    Define the struct for an ensemble of sims.
    """
    @with_kw struct Ensemble
        BD::BinDecMod
        e_size::Int64 = 1 # size of ensemble
        τₘ::Real = 1E3 # max system time
        Δτ::Real = 1.0 # time spacing of storage in trajectory
        sims = SSA(e_size, τₘ, Δτ, BD)
        @assert e_size >= 1
        @assert τₘ > 0
        @assert Δτ > 0 && Δτ < τₘ
    end

    """
    Calculation of the gain.
    """
    function Gain(Sᵢ::Int64, n::Int64, BD::BinDecMod)
        @unpack α, F, J, N = BD;
        m = (2*n-N)/N; # order parameter.
        G = -2*Sᵢ*(F + J*m*(1+α))+(2*J*(1+α))/N;
        return G
    end

    """
    Propensities.
    """
    function props(n::Int64, BD::BinDecMod)
        @unpack γ, N, β = BD;
        G = [Gain(-1, n, BD), Gain(1, n, BD)] # (-1→1,1→-1)
        props = zeros(2) # propensity vector.
        props[1] = γ*(N-n)*(1+exp(-β*G[1]))^-1 # f₊
        props[2] = γ*n*(1+exp(-β*G[2]))^-1 # f₋
        return props
    end

    """
    Mean-field fully connected SSA: species is n. Does much faster mf SSA sims.
    """
    function SSA(ens_its::Int64, τₘ::Real, Δτ::Real, BD::BinDecMod)
        @unpack p₀, fixedIC, N = BD;
        # create the storage array for the ensembles
        ens_store = Array{Array{Int64}}(undef, ens_its);
        # assign the initial condition - either fixed or binomially drawn.
        if fixedIC == true
            n = ceil(Int64, N*p₀)
        else
            n = rand(Binomial(N,p₀),1)[1]
        end
        # define the sim times for storage of n
        times = convert(Array{Float64,1},LinRange(τₘ,0.0,floor(Int,τₘ/Δτ)+1))
        # loop over for each sim in the ensemble
        for it in 1:ens_its
            # n at t=0 is either fixed or binomially drawn.
            fixedIC == true ? n = ceil(Int64, N*p₀) : n = rand(Binomial(N,p₀),1)[1]
            # initialise the state vector and sys time
            sim_times = copy(times)
            store = zeros(length(times));
            τ = 0.0;
            m = 1; # counter for updating storage
            while τ < τₘ
                ps = props(n, BD); # get propensities
                f = sum(ps);
                r1, r2 = rand(2); # collect 2 ind rand nums ∈ [0,1]
                u = (1/f)*log(1/r1); # waiting time until next reaction
                k = findfirst(x -> x>=r2*f,cumsum(ps)); # find fired reaction
                # update state vector storage
                while τ+u >= sim_times[length(sim_times)]
                    store[m] = copy(n)
                    pop!(sim_times)
                    m += 1
                    if length(sim_times) == 0
                        break
                    end
                end
                # update the state variable
                k == 1 ? n += 1 : n -= 1;
                # update the time.
                τ += u;
            end
            ens_store[it] = store;
        end
        return (reverse(times),ens_store)
    end

    """
    Get the central moments from the ensemble
    """
    function CentralMoments(ens::Ensemble,n::Int64)
        ns = convert(Matrix{Float64},transpose(cat(ens.sims[2]...,dims=2)))
        store = Array{Float64,1}(undef, length(ns[1,:]))
        if n == 1
            for t in 1:length(ns[1,:])
                store[t] = mean(ns[:,t])
            end
        else
            for t in 1:length(ns[1,:])
                store[t] = moment(ns[:,t],n)
            end
        end
        # need to sort out times
        @unpack Δτ = ens;
        times = LinRange(0:Δτ:length(ns[1,:])-1);
        return (times, store)
    end

    """
    Get prob distribution at all times for an ensemble
    """
    function EnsProb(ens::Ensemble, T::Int64)
        ns = convert(Matrix{Float64},transpose(cat(ens.sims[2]...,dims=2)))
        mod_bins = LinRange(minimum(ns[:,T])-0.5:1:maximum(ns[:,T])+0.5);
        mid_pts = LinRange(minimum(ns[:,T]):1:maximum(ns[:,T]));
        bin_vals = normalize(fit(Histogram, ns[:,T], mod_bins), mode=:probability).weights;
        return (mid_pts, bin_vals)
    end

end # module end
