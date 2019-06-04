using Turing, StatsPlots, MCMCChains

function infiere(θ_true)
    # define el modelo
    @model GaussianMixtureModel(x) = begin
        a = 10.0
        b = 100.0
        N = length(x)
        p ~ Normal(0,b)
        k = Vector{Int}(undef, N)
        for i = 1:N
            k[i] ~ DiscreteUniform(0,1)
            x[i] ~ Normal(p*k[i], k[i]*1+(1-k[i])*a)
        end
        return k
    end

    # genera la data para inferir el parámetro. Se generan muchos valores para la visualización
    n = 100000
    xdata = [(rand()<0.5) ? (rand(Normal(θ_true,1))) : (rand(Normal(0,10.0))) for i = 1:n]

    # sampling usando solo 200 valores de la data. Un aumento en este valor ralentiza considerablemente el proceso
    iter = 100
    datatamanho = 200
    c1 = sample(GaussianMixtureModel(xdata[1:datatamanho]), SMC(iter))

    # extraer los parámetros
    esta = []
    for i = 1:iter
        push!(esta,get(c1,:p).p[i])
    end

    # genera gráficos
    plot(xdata, seriestype = :density, xticks = -40:5:40, fill=0, α=0.2, w = 2, c = :red, label = "Data")
    # plot!(Normal(θ_true,1),xlims = (-x,x),fill=0, α=0.2, w = 2, c = :blue)
    # plot!(Normal(0,10.0),xlims = (-x,x),fill=0, α=0.2, w = 2, c = :blue)
    vline!([mean(esta)], label = "Inferred = $(round(mean(esta),digits=2))", w = 3,c = :green)
    vline!([θ_true], label = "True Mean = $(θ_true)", w =3, c = :red)
end
