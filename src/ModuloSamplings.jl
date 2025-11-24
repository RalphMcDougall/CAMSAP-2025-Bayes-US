module ModuloSampling

using Random,
    Distributions,
    LinearAlgebra,
    LogExpFunctions,
    Plots,
    MatrixEquations,
    FastGaussQuadrature,
    ProgressMeter

function K1(a, D, Lambda, eps)
    #println(a, " ", D, " ", Lambda, " ", eps)
    R1 = exp(-2 * Lambda * (-a + 3 * Lambda - D / 2))
    #println(R1)
    val = floor(
        ((a + D / 2) + sqrt((a + D / 2)^2 - 2 * log(eps * (1 - R1) / 2))) / (2 * Lambda)
    )
    #println("")
    return val > 0 ? val : 0
end
function K2(a, D, Lambda, eps)
    R2 = exp(-2 * Lambda * (a + 3 * Lambda - D / 2))
    val = floor(
        (-(a - D / 2) + sqrt((a + D / 2)^2 - 2 * log(eps * (1 - R2) / 2))) / (2 * Lambda)
    )
    return val > 0 ? val : 0
end

function bounds(a, D, Lambda, eps)
    if a >= 0
        return K1(a, D, Lambda, eps), K2(a, D, Lambda, eps)
    else
        return K2(a, D, Lambda, eps), K1(a, D, Lambda, eps)
    end
end

struct SSM
    A::Matrix{Float64}
    Q::Matrix{Float64}
    H::Matrix{Float64}
    R::Matrix{Float64}
end

function process_power(A::Matrix{Float64}, Q::Matrix{Float64}, H::Matrix{Float64})
    return (H * lyapd(A, Q) * H')[1]
end

obs_dim(ssm::SSM) = size(ssm.H)[1]
state_dim(ssm::SSM) = size(ssm.H)[2]

struct PFParams
    num_particles::Integer
    min_particle_factor::Float64
    lambda_min::Float64
    lambda_max::Float64
    adc_bits::Float64
    ssm::SSM
    eps::T where {T<:AbstractFloat}
end

Lambda(pf_params::PFParams) = pf_params.lambda_max - pf_params.lambda_min

num_bins(adc_bits) = 2^adc_bits
vec2mat(v) = reshape(v, (length(v), 1))

function integrate(f::Function, a::Real, b::Real)
    x, w = gausslegendre(10)
    x = 0.5 * (b - a) .* x .+ 0.5 * (a + b)
    w *= (b - a) / 2

    return sum(wi * f(xi) for (wi, xi) in zip(w, x))
end

function continuous_to_discrete(ssm::SSM, sample_time::Float64)
    new_A = exp(ssm.A * sample_time)

    fq(t) = exp(ssm.A * (sample_time - t)) * ssm.Q * exp(ssm.A * (sample_time - t))'
    new_Q = integrate(fq, 0, sample_time)
    new_H = ssm.H
    new_R = ssm.R
    return ssm(new_A, new_Q, new_H, new_R)
end

function prod(ma, Pa, mb, Pb)
    invPa = inv(Pa)
    invPb = inv(Pb)
    invPc = invPa + invPb
    Pc = inv(invPc)
    mc = Pc * (invPa * ma + invPb * mb)
    return mc, Pc, ma, mb, Pa + Pb
end

function ssm_output_m(ssm::SSM, m)
    return ssm.H * m
end
function ssm_output_P(ssm::SSM, P)
    return ssm.H * P * ssm.H'
end

function ssm_pred(ssm::SSM, m, P)
    A, Q = ssm.A, ssm.Q
    return A * m, A * P * A' + Q
end

struct ParticleState
    v::Float64
    mx::Vector{Float64}
    Px::Matrix{Float64}
    ssm::SSM
end

function initialise_pf(pf_params::PFParams)
    ssm = pf_params.ssm
    initial_var = lyapd(ssm.A, ssm.Q)

    particles::Vector{ParticleState} = [
        ParticleState(
            0.0,
            vec(zeros((state_dim(ssm), 1))),
            initial_var * Matrix(I, state_dim(ssm), state_dim(ssm)),
            ssm,
        ) for _ in 1:(pf_params.num_particles)
    ]
    log_weights = zeros(pf_params.num_particles) .- log(pf_params.num_particles)

    return particles, log_weights
end

function kalman_update(
    ssm::SSM, m::Vector{Float64}, P::Matrix{Float64}, observation::Vector{Float64}
)
    m_pred, P_pred = ssm_pred(ssm, m, P)

    H, R = ssm.H, ssm.R

    v = observation - H * m_pred
    S = H * P_pred * H' + R
    K = P_pred * H' * inv(S)
    new_m = m_pred + K * v
    new_P = P_pred - K * S * K'

    return new_m, new_P
end

function proposal(
    particle::ParticleState, observation::Vector{Float64}, pf_params::PFParams
)
    (; v, mx, Px, ssm) = particle
    H, R = ssm.H, ssm.R

    pred_mx, pred_Px = ssm_pred(ssm, mx, Px)
    pred_mv, pred_Pv = H * pred_mx, H * pred_Px * H' + R

    pred_v_dist = Normal(pred_mv[1], sqrt(pred_Pv[1]))

    bin_width = Lambda(pf_params) / num_bins(pf_params.adc_bits)

    L = pf_params.lambda_max / sqrt(pred_Pv[1])
    a = (observation[1] - pred_mv[1]) / sqrt(pred_Pv[1])
    offset_cnt = 0
    while a >= L
        a -= 2 * L
        offset_cnt += 1
    end
    while a < -L
        a += 2 * L
        offset_cnt -= 1
    end
    D = bin_width / sqrt(pred_Pv[1])
    K1, K2 = bounds(a, D, L, pf_params.eps)

    bin_centres = a .+ ((-K1):K2) * (2 * L)

    bin_mins = bin_centres .- 0.5 * D
    bin_maxs = bin_centres .+ 0.5 * D
    log_bin_weights = zeros(length(bin_mins))

    dist = Normal(0, 1)
    ind = 1
    for (bin_min, bin_max) in zip(bin_mins, bin_maxs)
        log_bin_weights[ind] = log_cdf_diff(dist, bin_min, bin_max)
        ind += 1
    end

    pre_max = maximum(log_bin_weights)
    if !isfinite(pre_max)
        error("Check this!")
        new_v = Inf
    else
        log_bin_weights .-= reduce(logaddexp, log_bin_weights)

        ind = rand(Categorical(exp.(log_bin_weights)))

        bin_min = bin_mins[ind]
        bin_max = bin_maxs[ind]

        min_bin_cdf = cdf(dist, bin_min)
        max_bin_cdf = cdf(dist, bin_max)
        u = if min_bin_cdf < max_bin_cdf
            rand(Uniform(min_bin_cdf, max_bin_cdf))
        else
            min_bin_cdf
        end

        new_v = pred_mv[1] + sqrt(pred_Pv[1]) * quantile(dist, u)
    end
    new_mx, new_Px = kalman_update(ssm, mx, Px, [new_v;])

    return ParticleState(new_v, new_mx, new_Px, ssm)
end

function log_cdf_diff(dist, min, max)
    return logdiffcdf(dist, max, min)
end

function log_weight_update(
    new_particle::ParticleState,
    old_particle::ParticleState,
    observation::Float64,
    pf_params::PFParams,
)
    (; v, mx, Px, ssm) = old_particle
    H, R = ssm.H, ssm.R

    pred_mx, pred_Px = ssm_pred(ssm, mx, Px)
    pred_mv, pred_Pv = H * pred_mx, H * pred_Px * H' + R
    pred_v_dist = Normal(pred_mv[1], sqrt(pred_Pv[1]))

    bin_width = Lambda(pf_params) / num_bins(pf_params.adc_bits)

    L = pf_params.lambda_max / sqrt(pred_Pv[1])
    a = (observation[1] - pred_mv[1]) / sqrt(pred_Pv[1])
    offset_cnt = 0
    while a >= L
        a -= 2 * L
        offset_cnt += 1
    end
    while a < -L
        a += 2 * L
        offset_cnt -= 1
    end
    if a < -L || a >= L
        error("Check modulo operation")
    end
    D = bin_width / sqrt(pred_Pv[1])
    K1, K2 = bounds(a, D, L, pf_params.eps)

    bin_centres = a .+ (((-K1):K2) * (2 * L))

    bin_mins = bin_centres .- 0.5 * D
    bin_maxs = bin_centres .+ 0.5 * D

    log_bin_weights = zeros(length(bin_mins))

    dist = Normal(0, 1)
    ind = 1
    for (bin_min, bin_max) in zip(bin_mins, bin_maxs)
        log_bin_weights[ind] = log_cdf_diff(dist, bin_min, bin_max)
        ind += 1
    end

    val = reduce(logaddexp, log_bin_weights)
    return (isnan(val) || !isfinite(val)) ? -1E12 : val
end

log_ess(log_weights::Vector{Float64}) = -reduce(logaddexp, log_weights * 2)

function update(
    particles::Vector{ParticleState},
    log_weights::Vector{Float64},
    observation::Vector{Float64},
    pf_params::PFParams,
)
    num_particles, min_particle_factor = pf_params.num_particles,
    pf_params.min_particle_factor

    new_particles = [proposal(p, observation, pf_params) for p in particles]
    new_log_weights =
        log_weights + [
            log_weight_update(new_particles[i], particles[i], observation[1], pf_params) for
            i in 1:num_particles
        ]
    new_log_weights .-= reduce(logaddexp, new_log_weights)

    if log_ess(new_log_weights) < log(min_particle_factor * num_particles)
        new_particle_inds = rand(Categorical(exp.(log_weights)), num_particles)
        new_particles = particles[new_particle_inds]
        new_log_weights = zeros(num_particles) .- log(num_particles)
    end

    return new_particles, new_log_weights
end

function adc_bin(
    x::T where {T<:AbstractFloat},
    max_val::T where {T<:AbstractFloat},
    min_val::T where {T<:AbstractFloat},
    bits::Real,
)
    bin_size = (max_val - min_val) / num_bins(bits)
    return bin_size * floor(x / bin_size + 0.5)
end

function adc_foo(
    x::T where {T<:AbstractFloat},
    lambda_min::T where {T<:AbstractFloat},
    lambda_max::T where {T<:AbstractFloat},
    bits::Real,
)
    Lambda = lambda_max - lambda_min
    temp = x
    while temp >= lambda_max
        temp -= Lambda
    end
    while temp < lambda_min
        temp += Lambda
    end

    return adc_bin(temp, lambda_max, lambda_min, bits)
end

function run_pf(
    observations::Vector{Float64},
    sample_rate::Float64,
    pf_params::PFParams,
    H_smoothing::Matrix{Float64},
)
    particles, log_weights = initialise_pf(pf_params)

    num_samples = length(observations)

    x_m = zeros((num_samples, pf_params.num_particles))
    x_P = zeros((num_samples, pf_params.num_particles))
    weights_hist = zeros((num_samples, pf_params.num_particles))

    x_m_smooth = zeros((num_samples, pf_params.num_particles))
    x_P_smooth = zeros((num_samples, pf_params.num_particles))

    @showprogress for current_t in 1:num_samples
        particles, log_weights = update(
            particles, log_weights, [observations[current_t];], pf_params
        )

        for ind in 1:(pf_params.num_particles)
            p = particles[ind]
            x_m[current_t, ind] = (pf_params.ssm.H * p.mx)[1]
            x_P[current_t, ind] = (pf_params.ssm.H * p.Px * pf_params.ssm.H')[1]
            weights_hist[current_t, :] = exp.(log_weights)

            x_m_smooth[current_t, ind] = (H_smoothing * p.mx)[1]
            x_P_smooth[current_t, ind] = (H_smoothing * p.Px * H_smoothing')[1]
        end
    end
    return x_m, x_P, weights_hist, x_m_smooth, x_P_smooth
end

end # module