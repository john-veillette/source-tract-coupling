using FFTW
using DSP
using ChaosTools

Fs = 48000
times = 0:(1/Fs):1

findnearest(A::AbstractVector,x) = findmin(abs.(A .- x))[2]

function predict(θ, X = u0, times = 0:(1/Fs):1)
    Array(solve(prob, MethodOfSteps(AutoTsit5(Rosenbrock23())), 
                u0 = X, p = θ, 
                saveat = times,
                sensealg = ForwardDiffSensitivity(),
                force_dtmin = true
                )[3,:])
end

function simulate_tract(α, β, γ, L)
    c = 34300 # speed of sound in cm/sec
    τ = 2 * L/c # coupling delay
    predict([α, β, γ, τ])
end

function compute_spectrum(x, fs = Fs)
    # returns the square magnitude spectrum of the input signal
    x .* hanning(length(x))
    F = fft(x) |> fftshift
    psd = abs.(F)
    freqs = FFTW.fftfreq(length(x), fs) |> fftshift
    speech_range = (0 .< freqs) .& (freqs .<= 5000)
    freqs[speech_range], psd[speech_range].^2
end

function compute_fundamental(x, fs = Fs) # return fundamental frequency (F0, .k.a. pitch)
    try
        period = estimate_period(x, :autocorrelation, 0:(1/fs):((length(x) - 1)/fs))
        return 1/period
    catch
        return 0. # not enough periodicity
    end
end

function compute_sci(freqs, psd, f0)
    f0_idx = findnearest(freqs, f0)
    f0_pow = psd[f0_idx]
    msf = sum(psd .* freqs)/sum(psd)
    if f0 != 0.
        sci = msf/f0 # spectral content index
    else
        sci = Inf 
    end
end

function spectrum_diff(freqs, psd1, psd2)
    # area between the dB scale spectrograms 
    sum(abs.(10*log10.(psd1) .- 10*log10.(psd2))) * (freqs[2] - freqs[1])
end

function get_metrics(α, β, γ)
    x = simulate_tract(α, β, γ, 13)
    y = simulate_tract(α, β, γ, 20)
    # compute spectrograms
    freqs, psd_x = compute_spectrum(x)
    freqs, psd_y = compute_spectrum(y)
    # estimate fundamental frequencies
    f0_x = compute_fundamental(x)
    f0_y = compute_fundamental(y)
    # estimate spectral content index
    sci_x = compute_sci(freqs, psd_x, f0_x)
    sci_y = compute_sci(freqs, psd_y, f0_y)
    # and spectral difference 
    delta = spectrum_diff(freqs, psd_x, psd_y)
    f0_x, sci_x, f0_y, sci_y, delta # values we want to save!
end
