using CSV,
    DataFrames,
    Plots,
    Statistics,
    Distributions,
    DSP,
    MatrixEquations,
    LinearAlgebra,
    Random,
    LaTeXStrings,
    Crayons,
    ProgressMeter,
    Printf;

include("../src/ModuloSamplings.jl");
include("../src/ClassicalUS.jl")

SEPERATOR = "-------------------------"
function log_info(txt::String)
    return println("\n", Crayon(; foreground=:yellow), txt, Crayon(; reset=true), "\n")
end

BLACK = colorant"#000000"
BLUE = colorant"#030aa7" # XKCD: "Cobalt blue"
RED = colorant"#9a0200" # XKCD: "Deep red"
GREEN = colorant"#02590f" # XKCD: "Deep green"
GREY = colorant"#929591" # XKCD: "Grey"

LINE_WIDTH = 2;
SECONDARY_LINEWIDTH = 1
LEGEND_FONT_SIZE = 10;

SCATTER_SIZE = 2
GAUSSIAN_CONFIDENCE_95 = 1.96

mse(x, y) = sqrt((x - y)' * (x - y)) / length(y)

function loss(y, sample_rate, params)
    c, a1, f1, phi1, a2, f2, phi2 = params

    err = 0

    c_der = 0

    a1_der = 0
    f1_der = 0
    phi1_der = 0

    a2_der = 0
    f2_der = 0
    phi2_der = 0

    for t in 1:length(y)
        total_term =
            y[t] - c - a1 * sin(2 * pi * f1 * t / sample_rate + phi1) -
            a2 * sin(2 * pi * f2 * t / sample_rate + phi2)
        err += total_term^2

        c_der += 2 * total_term * (-1)

        a1_der += 2 * total_term * (-sin(2 * pi * f1 * t / sample_rate + phi1))
        f1_der +=
            2 *
            total_term *
            (-a1 * 2 * pi * t / sample_rate * cos(2 * pi * f1 * t / sample_rate + phi1))
        phi1_der += 2 * total_term * (-a1 * cos(2 * pi * f1 * t / sample_rate + phi1))

        a2_der += 2 * total_term * (-sin(2 * pi * f2 * t / sample_rate + phi2))
        f2_der +=
            2 *
            total_term *
            (-a2 * 2 * pi * t / sample_rate * cos(2 * pi * f2 * t / sample_rate + phi2))
        phi2_der += 2 * total_term * (-a2 * cos(2 * pi * f2 * t / sample_rate + phi2))
    end

    err /= length(y)
    ders = [c_der, a1_der, f1_der, phi1_der, a2_der, f2_der, phi2_der] / length(y)

    return err, ders
end

function draw_samples(ssm::ModuloSampling.SSM, ts)
    initial_var = Matrix(I, ModuloSampling.state_dim(ssm), ModuloSampling.state_dim(ssm))#lyapd(ssm.A, ssm.Q);
    #println(det(initial_var))
    x = rand(MultivariateNormal(zeros((ModuloSampling.state_dim(ssm))), initial_var))

    samples = zeros(length(ts))

    for t in 1:length(ts)
        x =
            ssm.A * x + rand(
                MultivariateNormal(
                    zeros(ModuloSampling.state_dim(ssm)),
                    ssm.Q +
                    1E-6 *
                    Matrix(I, ModuloSampling.state_dim(ssm), ModuloSampling.state_dim(ssm)),
                ),
            )
        samples[t] = (ssm.H * x)[1]
    end

    p = plot(ts, samples)
    return display(p)
end

function mod_sample(x, lambda)
    return if (x < lambda)
        (x >= -lambda ? x : mod_sample(x + 2 * lambda, lambda))
    else
        mod_sample(x - 2 * lambda, lambda)
    end
end;

function display_mse(label, reconstruction, gt, baseline)
    return println(
        Crayon(; foreground=:light_blue),
        label,
        " mMSE:\t",
        Crayon(; reset=true),
        "$(round(mse(reconstruction, gt) * 1000; sigdigits=4)) -> $(round(100 * mse(reconstruction, gt) / mse(baseline, gt); sigdigits=4))%",
    )
end

function display_val(label, val)
    return println(
        Crayon(; foreground=:light_blue),
        "$(label):\t",
        val >= 0 ? " " : "",
        Crayon(; foreground=:white, bold=true),
        @sprintf("%0.4e", val),
        Crayon(; reset=true),
    )
end
function display_mat(label, mat)
    return println(
        Crayon(; foreground=:light_blue),
        "$(label):\t",
        Crayon(; foreground=:white, bold=true),
        repr("text/plain", mat),
        "\n",
        Crayon(; reset=true),
    )
end

function correct_modulo_sampled(
    adc_sampled, modulo_sampled, sample_rate, lambda_max, lambda_min
)
    # The best fit sum of sine waves needs to be determined from the provided ground truth.
    # This is done using a simple gradient-descent approach.
    # The developed model can then be compared against this compensated signal.

    display_val("MAX", maximum(modulo_sampled))
    display_val("MIN", minimum(modulo_sampled))

    c = 0

    a1 = 2
    f1 = 8E3
    phi1 = 0

    a2 = 0.3
    f2 = 1E3
    phi2 = 0

    step = 1E-1
    err = 0
    ders = [Inf, 0, 0, 0, 0, 0, 0]
    params = [c, a1, f1, phi1, a2, f2, phi2]
    while sqrt(ders' * ders) > 5E-5
        err, ders = loss(adc_sampled, sample_rate, params)
        params -= ders * step
    end
    c, a1, f1, phi1, a2, f2, phi2 = params
    println("\nConstant offset")
    display_val("c", c)

    println("")
    println("Sine 1")
    display_val("a1", a1)
    display_val("f1", f1)
    display_val("ϕ1", phi1 / pi)

    println("")
    println("Sine 2")
    display_val("a2", a2)
    display_val("f2", f2)
    display_val("ϕ2", phi2 / pi)
    println("")
    display_val("residual err", err)

    ts = (1:length(adc_sampled)) / sample_rate
    f(t) = c + a1 * sin(2 * pi * f1 * t + phi1) + a2 * sin(2 * pi * f2 * t + phi2)
    gts = f.(ts)

    # Performing modulo sampling on the ADC measurements does not provide the modulo-sampled signal.
    # There may be some constant offset from one to the other because of different hardware parameters.
    # This needs to be corrected for.

    best_const = 0
    best_err = Inf
    for con in lambda_min:1E-4:lambda_max
        err = 0

        for t in eachindex(adc_sampled)
            term = abs(
                (mod_sample(adc_sampled[t], lambda_max) - (modulo_sampled[t] + con)) %
                (2 * lambda_max),
            )
            err += (term < lambda_max ? term : 2 * lambda_max - term)^2
        end

        if err < best_err
            best_err = err
            best_const = con
        end
    end
    modulo_sampled .+= best_const
    display_val("best_offset", best_const)

    display_val("NEW MAX", maximum(modulo_sampled))
    display_val("NEW MIN", minimum(modulo_sampled))
    return modulo_sampled, gts, ts
end

function hardware_reconstruction()
    log_info("$(SEPERATOR)\n<> Running hardware reconstruction experiment...")

    data_path = "data/data.csv"
    if isfile(data_path)
        df = CSV.read(data_path, DataFrame)
    else
        println(
            Crayon(; foreground=:light_red),
            "Unable to locate data file at $(data_path). \nThis file may have been deleted, please download the original file at: \n\n",
            Crayon(; foreground=:white),
            "https://alumni.media.mit.edu/~ayush/USF-RR/_downloads/5fa7ddc47c4e38e1332b5b3555b2f549/8K_code_data.zip",
            Crayon(; foreground=:light_red),
            " \n\nto the `data` folder. Thereafter, run \n\n",
            Crayon(; foreground=:white),
            "python data/mat_to_csv.py",
            Crayon(; foreground=:light_red),
            " \n\nagain from the project root.\n",
        )
        error("Unable to locate file.")
    end
    adc_sampled, modulo_sampled = df[!, "x"], df[!, "y"]

    adc_bits = 6.7
    lambda_max = 0.4
    lambda_min = -0.4
    sample_rate = 100E3

    modulo_sampled, gts, ts = correct_modulo_sampled(
        adc_sampled, modulo_sampled, sample_rate, lambda_max, lambda_min
    )

    gts = gts
    ts = ts[1:length(gts)]

    samples = modulo_sampled[1:length(gts)]

    T = 1 / sample_rate
    Omega = 2 * pi * 8E3
    us_reconstruction = ClassicalUS.noisy_US(samples, gts, 3, lambda_max, T, Omega)

    analog = DSP.Filters.analogfilter(
        DSP.Filters.Lowpass(2 * pi * 10E3), DSP.Filters.Butterworth(5)
    )
    digital = DSP.Filters.bilinear(analog, sample_rate)

    filt_ord = length(DSP.Filters.coefa(digital))

    A = zeros((filt_ord, filt_ord))
    A[1, 1:(end - 1)] = -(DSP.Filters.coefa(digital)[2:end])
    for i in 2:filt_ord
        A[i, i - 1] = 1
    end

    Q = zeros((filt_ord, filt_ord))

    H = zeros((1, filt_ord))
    H[1, :] = DSP.Filters.coefb(digital)

    println("\nSystem matrices:\n$(SEPERATOR)")
    display_mat("A", A)
    display_mat("H", H)

    R = zeros((1, 1))
    Q[1, 1] = 1

    noise_var = 5E-4
    target_power = 2.5E0

    Q[1, 1] = target_power / ModuloSampling.process_power(A, Q, H)
    R[1, 1] = noise_var
    ssm = ModuloSampling.SSM(A, Q, H, R)

    pf_params = ModuloSampling.PFParams(
        2_00, 1E-1, lambda_min, lambda_max, adc_bits, ssm, 1E-4
    )
    ms, Ps, weights_hist, _, _ = ModuloSampling.run_pf(
        samples, sample_rate, pf_params, zeros((1, filt_ord))
    )

    m_sample = zeros(length(samples))
    for t in 1:length(samples)
        m_sample[t] = ms[t, :]' * weights_hist[t, :]
    end

    log_info("$(SEPERATOR)\nErrors\n$(SEPERATOR)")
    restriction = 75:length(gts)
    display_mse(" US-Alg", us_reconstruction, gts, us_reconstruction)
    display_mse("BUS-SSM", m_sample, gts, us_reconstruction)
    display_mse(
        " US-Alg (after burn-in)",
        us_reconstruction[restriction],
        gts[restriction],
        us_reconstruction[restriction],
    )
    display_mse(
        "BUS-SSM (after burn-in)",
        m_sample[restriction],
        gts[restriction],
        us_reconstruction[restriction],
    )
    println(
        Crayon(; foreground=:light_blue),
        "Relative all:     ",
        Crayon(; reset=true),
        round(100 * (mse(m_sample, gts) / mse(us_reconstruction, gts) - 1); sigdigits=4),
        "%",
    )
    println(
        Crayon(; foreground=:light_blue),
        "Relative burn-in: ",
        Crayon(; reset=true),
        round(
            100 * (
                mse(m_sample[restriction], gts[restriction]) /
                mse(us_reconstruction[restriction], gts[restriction]) - 1
            );
            sigdigits=4,
        ),
        "%",
    )
    log_info("$(SEPERATOR)")

    prog = ProgressMeter.ProgressUnknown(; desc="Constructing plots", spinner=true)
    next!(prog)

    max_weights = [(20 / pf_params.num_particles) for t in 1:length(samples)]

    restriction = 1:150

    c = mean(modulo_sampled)
    p_adc = hline(
        [c];
        color=BLACK,
        label="",
        xticks=false,
        grid=false,
        xlims=(-0.05, restriction[end] / sample_rate * 1000),
    )
    hline!([c - lambda_max, c + lambda_max]; label="", color=GREY)
    plot!(
        1000 * ts[restriction],
        samples[restriction];
        color=GREEN,
        seriestype=:steppre,
        label="",
    )
    ylabel!("ADC")
    yticks!(
        [c - lambda_max, c, c + lambda_max], [L"$C - \lambda$", L"$C$", L"$C + \lambda$"]
    )
    title!("Provided samples and signal reconstructions")

    p_recon = hline(
        [0];
        color=BLACK,
        label="",
        grid=false,
        xlims=(-0.05, restriction[end] / sample_rate * 1000),
        ylims=(-2.5, 2.5),
        legend=:topright,
    )
    plot!(1000 * ts[restriction], us_reconstruction[restriction]; color=RED, label="US-Alg")
    plot!(1000 * ts[restriction], m_sample[restriction]; color=BLUE, label="BUS-SSM")
    xlabel!("Time [ms]")
    ylabel!(L"$g_t$")

    p = plot(p_adc, p_recon; layout=(2, 1), link=:x)
    savefig(p, "figs/fig_01_hardware_reconstruction.pdf")
    next!(prog)

    p_errs = hline(
        lambda_max * (-6:2:6);
        color=GREY,
        label="",
        grid=false,
        ylim=(-2.1, 2.1),
        xlims=(-0.05, restriction[end] / sample_rate * 1000),
        legend=:topright,
    )
    scatter!(
        [-1],
        [0];
        label=L"$\mathbf{H}\mathbf{m}_x$",
        markercolor=BLUE,
        markeralpha=1,
        markersize=2,
        markerstrokecolor=BLUE,
    )
    scatter!(
        [-1],
        [0];
        label="95% error",
        markercolor=GREY,
        markeralpha=1,
        markersize=1,
        markerstrokecolor=GREY,
    )
    for ind in 1:(pf_params.num_particles)
        scatter!(
            1000 * ts[restriction],
            ms[restriction, ind] - gts[restriction];
            label="",
            markercolor=BLUE,
            markeralpha=(weights_hist[restriction, ind] ./ max_weights[restriction]),
            markersize=2,
            markerstrokecolor=BLUE,
        )
        scatter!(
            1000 * ts[restriction],
            ms[restriction, ind] + GAUSSIAN_CONFIDENCE_95 * sqrt.(Ps[restriction, ind]) -
            gts[restriction];
            label="",
            linestyle=:dash,
            markercolor=GREY,
            markeralpha=(weights_hist[restriction, ind] ./ max_weights[restriction]),
            markersize=1,
            markerstrokecolor=GREY,
        )
        scatter!(
            1000 * ts[restriction],
            ms[restriction, ind] - GAUSSIAN_CONFIDENCE_95 * sqrt.(Ps[restriction, ind]) -
            gts[restriction];
            label="",
            linestyle=:dash,
            markercolor=GREY,
            markeralpha=(weights_hist[restriction, ind] ./ max_weights[restriction]),
            markersize=1,
            markerstrokecolor=GREY,
        )
    end

    p = plot(p_errs; layout=(1, 1), link=:x)
    xlabel!("Time [ms]")
    ylabel!("Error")
    range = -4:2:4
    yticks!(range * lambda_max, ["$(l)" * L"$\lambda$" for l in range])
    title!("Individual particle errors")
    savefig(p, "figs/fig_02_hardware_particles.pdf")
    next!(prog)
    finish!(prog)
    sleep(0.5)

    return log_info("</> Hardware reconstruction experiment completed.\n$(SEPERATOR)")
end

function noisy_reconstruction()
    log_info("$(SEPERATOR)\n<> Running noisy reconstruction experiment...")

    base_sample_rate = 1E3
    T = 1 / base_sample_rate
    ts = 0:T:1
    f(t) = exp(-20 * (t - 0.5)^2) * sin(2 * pi * 10 * (t - 0.5))
    x = f.(ts)

    # -----
    # Found a small mistake in the code here when producing a presentation.
    # I had originally used 5E-2 as the signal noise variance, but forgot that 
    # Normal(..., ...) takes the standard variation as its argument.
    # The code as here replicates the experiment, although there is a slight 
    # error in the paper (only the listed variance should change, and it 
    # should be stated that the SSM model isn't matched to the process noise.)

    true_noise_var = (5E-2)^2
    model_noise_var = 5E-2

    y = x + rand(Normal(0, sqrt(true_noise_var)), length(ts))
    # -----

    beta = 1.2

    lambda = 0.4
    adc_bits = 3

    z = y
    z_adc = ModuloSampling.adc_foo.(z, -lambda, lambda, adc_bits)
    z_only_adc =
        ModuloSampling.adc_foo.(z, -100.0, 100.0, log2(200 * (2^adc_bits) / (2 * lambda)))
    z_low_adc = ModuloSampling.adc_foo.(z, -1.2, 1.2, adc_bits)

    Omega = 2 * pi * 200
    eta_max = 2 * 2 * lambda / (2^adc_bits)
    display_val("Calculuated US α", ClassicalUS.alpha(beta, lambda, eta_max))
    display_val("US N lower bound", ClassicalUS.N_lb(lambda, beta, T, Omega))
    display_val("US N noisy bound", ClassicalUS.N_noisy(lambda, beta, T, Omega))
    display_val("US T upper bound", ClassicalUS.US_theorem_bound(Omega))
    display_val(
        "US T noisy bound", ClassicalUS.US_theorem_bound_noisy(Omega, beta, lambda, eta_max)
    )
    display_val("US T η bound", ClassicalUS.eta_upper_bound(T, beta, -2))
    display_val("Curr T", T)
    us_reconstruction = ClassicalUS.noisy_US(z_adc, x, 1, lambda, T, Omega)
    display_val("mERR US", 1000 * mse(us_reconstruction, x))

    analog = DSP.Filters.analogfilter(
        DSP.Filters.Lowpass(2 * pi * 2E1), DSP.Filters.Butterworth(8)
    )
    digital = DSP.Filters.bilinear(analog, base_sample_rate)

    filt_ord = length(DSP.Filters.coefa(digital))

    smoothing_range = 5
    A = zeros((filt_ord + smoothing_range, filt_ord + smoothing_range))
    A[1, 1:(filt_ord - 1)] = -(DSP.Filters.coefa(digital)[2:end])
    for i in 2:size(A)[1]
        A[i, i - 1] = 1
    end

    Q = zeros(size(A))

    H = zeros((1, filt_ord + smoothing_range))
    H[1, 1:filt_ord] = DSP.Filters.coefb(digital)
    H_smoothing = zeros((1, filt_ord + smoothing_range))
    H_smoothing[1, (smoothing_range + 1):end] = DSP.Filters.coefb(digital)
    R = zeros((1, 1))
    Q[1, 1] = 1

    desired_power = 1E0

    Q[1, 1] = desired_power / ModuloSampling.process_power(A, Q, H)
    R[1, 1] = model_noise_var
    ssm = ModuloSampling.SSM(A, Q, H, R)

    pf_params = ModuloSampling.PFParams(4_00, 0.1, -lambda, lambda, adc_bits, ssm, 1E-6)
    ms, Ps, weights_hist, ms_smooth, Ps_smooth = ModuloSampling.run_pf(
        z_adc, base_sample_rate, pf_params, H_smoothing
    )
    m_sample = zeros(length(z_adc))
    for t in 1:length(z_adc)
        m_sample[t] = ms[t, :]' * weights_hist[t, :]
    end
    m_smooth = zeros(length(z_adc))
    for t in smoothing_range:length(z_adc)
        m_smooth[t - smoothing_range + 1] = ms_smooth[t, :]' * weights_hist[t, :]
    end

    baseline = y

    log_info("$(SEPERATOR)\nErrors\n$(SEPERATOR)")

    display_mse("LOW ADC", z_low_adc, x, baseline)
    display_mse("ORIG", baseline, x, baseline)
    display_mse("ALL ADC", z_only_adc, x, baseline)
    display_mse("US-Alg", us_reconstruction, x, baseline)
    display_mse("BUS-SSM (filtering)", m_sample, x, baseline)
    display_mse("BUS-SSM (smoothing)", m_smooth, x, baseline)

    log_info("$(SEPERATOR)")

    prog = ProgressMeter.ProgressUnknown(; desc="Constructing plots", spinner=true)
    next!(prog)

    p_samples = plot(
        ts,
        z_adc;
        seriestype=:steppre,
        label=L"Measured $y_t$",
        color=GREEN,
        grid=false,
        xticks=false,
    )
    plot!(ts, us_reconstruction; label="US-Alg", color=RED)
    title!("Noisy signal reconstruction")

    p_recon = scatter(
        ts, y; label=L"Unfolded $v_t$", color=GREY, markerstrokecolor=GREY, markersize=1
    )
    plot!(ts, x; label=L"Ground truth $g_t$", color=BLACK, grid=false)
    plot!(ts, m_smooth; label="BUS-SSM (smooth)", color=BLUE)
    xlabel!("Time [s]")

    p = plot(p_samples, p_recon; link=:x, layout=(2, 1))

    savefig(p, "figs/fig_03_noisy_signal.pdf")
    next!(prog)
    finish!(prog)
    sleep(0.5)
    return log_info("</> Noisy reconstruction experiment completed.\n$(SEPERATOR)")
end

# Program execution starts here

AUTHORS = "MCDOUGALL & GODSILL"
TITLE = "SEQUENTIAL BAYESIAN SIGNAL RECONSTRUCTION FOR UNLIMITED SAMPLING"
CONFERENCE = "CAMSAP 2025"
log_info("$(SEPERATOR)\n$(AUTHORS) - $(TITLE)\n$(CONFERENCE)\n$(SEPERATOR)")

Random.seed!(1234)
hardware_reconstruction()

Random.seed!(1234)
noisy_reconstruction()

log_info("Done.")