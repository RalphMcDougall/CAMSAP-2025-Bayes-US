module ClassicalUS

using Statistics

frac(x) = x - floor(x)
M(f, lambda) = 2 * lambda * (frac(0.5 * f / lambda + 0.5) - 0.5)

S(s) = cumsum([0; s])

Delta(y, N) = N == 0 ? y : Delta(diff(y), N - 1)
LambdaRound(f, lambda) = 2 * lambda * round(f / (2 * lambda))

N_lb(lambda, beta, T, Omega) = ceil((log(lambda) - log(beta)) / log(T * Omega * exp(1)))

function noisy_US(y, gts, N, lambda, T, Omega)
    g0 = ceil(maximum(abs.(gts)))
    beta = ceil(g0 / (2 * lambda)) * 2 * lambda
    J::Int = round(6 * beta / lambda)

    Delta_y = Delta(y, N)
    Delta_eps = M.(Delta_y, lambda) - Delta_y
    s_next = Delta_eps

    for _ in 0:(N - 2)
        S2 = S(S(s_next))
        s_next = S(s_next)

        s_next = LambdaRound.(s_next, lambda)

        kappa = floor((S2[1] - S2[J + 1]) / (12 * beta) + 0.5)

        s_next .+= 2 * lambda * kappa
    end
    gamma = S(s_next) + y[:]
    # Return the result with the mean as close as possible to 0.
    return gamma .- LambdaRound(mean(gamma), lambda)
end

# This provides an upper bound on the sample period, T, in terms of the signal bandwidth, Omega 
US_theorem_bound(Omega) = 1 / (2 * Omega * exp(1))
function US_theorem_bound_noisy(Omega, beta, lambda, eta_max)
    return 1 / (2^alpha(beta, lambda, eta_max) * Omega * exp(1))
end

alpha(beta, lambda, eta_max) = ceil(log(2 * beta / lambda) / log(4 * eta_max / lambda))

function N_noisy(lambda, beta, T, Omega)
    return -ceil((log(lambda) - log(beta) - 1) / log(T * Omega * exp(1)))
end
eta_upper_bound(lambda, beta, alpha) = 0.25 * lambda * (2 * beta / lambda)^(-1 / alpha)

end # module