using Optim
using Cubature
using Roots
using StatsFuns
using TimerOutputs

"""
TODOs:
    1. optimize pre_process, specifically the pdf support estimation, the pdf integral verification
        (Ideally, it's theoretically guarantee that pdf is well-conditioned (by trucation..), then for example [0, 10] would always cover the pdf support)
    2. Do we need quantiles from each distribution component to help quantile computation? -- try without this first
    3. the narrowest CI computation can be optimized (the equal CI is fine, only involving two quantiles)
"""

"pre-process pdf and cdf, given fixed pdf and cdf at x0, compute estimated support and check if pdf is proper"
function pre_process(x0::Array{T,2}, Fx0::Array{T,2}, pdf::Function, cdf::Function, test_buffer; yshift = 0., verbose=false) where T<:Float64
    # quantbound_fixed(p) = quantbound(x0, Fx0, p)
    pdf_fixed(y) = pdf(x0, Fx0, y, test_buffer = test_buffer)
    cdf_fixed(y) = cdf(x0, Fx0, y, test_buffer = test_buffer)
    # dpdf_fixed(y) = dpdf(x0, Fx0, y)

    support = [1e-4, 10.0]
    
    function support_comp!(pdf, support)
      current = pdf(support[1])
      for i in 1:5
        next = pdf(support[1]/10)
        if next < current
          support[1] /= 10
        else
          break
        end
        current = next
      end
      while pdf(support[2]) > 1e-6
        support[2] *= 1.2
      end
      support[2] += yshift
      # should make sure CDF(support[2]) - CDF(support[1]) > .98 to make 95% CI possible
      INT = cdf_fixed(support[2]) - cdf_fixed(support[1])
      if INT < .98
        @warn "improper pdf: integral = $INT"
      end
      return INT
    end
    INT = support_comp!(pdf_fixed, support)
    if verbose
    end
    return support, INT
end

"wrap up all statistics computation"
# function summary_comp(pdf_fixed::Function, cdf_fixed::Function, dpdf_fixed::Function, quantbound_fixed::Function, support::Array{T,1};
#                        px = .5, confidence_level = .95) where T<:Float64
#     quant_p, error_quant = quantile(cdf_fixed, quantbound, support; p=px)
#     med, error_med = median(cdf_fixed, quantbound, support)
#     mod = mode(pdf_fixed, support)
#     CI_equal, error_CI_eq = credible_interval(cdf_fixed, quantbound, support;
#                                                 mode=:equal, wp = confidence_level)
#     # CI_narrow, error_CI_nr = credible_interval(cdf_fixed, quantbound, support;
#     #                                             mode=:narrow, wp = confidence_level)
#     quantileInfo = (level = px, value = quant_p, error = error_quant)
#     medianInfo = (value = med, error = error_med)
#     CIequalInfo = (equal = CI_equal, error = error_CI_eq)
#     # CInarrowInfo = (equal = CI_narrow, error = error_CI_nr)
#     CInarrowInfo = nothing
#     DistributionInfo = (quantile = quantileInfo, median = medianInfo, mode = mod, CIequal = CIequalInfo, CInarrow = CInarrowInfo)
#     return DistributionInfo
# end



"""
Given pdf, cdf and maybe pdf_deriv,
compute median, quantile, mode, symmetric/narrowest credible interval.
Warning: only for normalized values
"""
function median(cdf::Function, support::Array{T,1}; pdf = nothing, pdf_deriv=nothing, quantile_bound=nothing, verbose=false,
    loose_bound=false, xtol= nothing, ftol = nothing) where T<:Float64
    med, err = quantile(cdf, support; quantile_bound=quantile_bound, verbose=verbose, loose_bound=loose_bound, xtol = xtol, ftol = ftol)
    return med, err, bound
end

"""
Implemented by Modesto Mas, taken from https://mmas.github.io/brent-julia
"""
function brent(f::Function, x0::Number, x1::Number, args::Tuple=();
               xtol::AbstractFloat=1e-6, ftol=100eps(Float64),
               maxiter::Integer=350)
    # print("Calling brent in statistics.jl")
    EPS = eps(Float64)
    y0 = f(x0,args...)
    y1 = f(x1,args...)
    if abs(y0) < abs(y1)
        # Swap lower and upper bounds.
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    end
    x2 = x0
    y2 = y0
    x3 = x2
    bisection = true
    for _ in 1:maxiter
        # x-tolerance.
        if abs(x1-x0) < xtol
            return x1
        end

        # Use inverse quadratic interpolation if f(x0)!=f(x1)!=f(x2)
        # and linear interpolation (secant method) otherwise.
        if abs(y0-y2) > ftol && abs(y1-y2) > ftol
            x = x0*y1*y2/((y0-y1)*(y0-y2)) +
                x1*y0*y2/((y1-y0)*(y1-y2)) +
                x2*y0*y1/((y2-y0)*(y2-y1))
        else
            x = x1 - y1 * (x1-x0)/(y1-y0)
        end

        # Use bisection method if satisfies the conditions.
        delta = abs(2EPS*abs(x1))
        min1 = abs(x-x1)
        min2 = abs(x1-x2)
        min3 = abs(x2-x3)
        if (x < (3x0+x1)/4 && x > x1) ||
           (bisection && min1 >= min2/2) ||
           (!bisection && min1 >= min3/2) ||
           (bisection && min2 < delta) ||
           (!bisection && min3 < delta)
            x = (x0+x1)/2
            bisection = true
        else
            bisection = false
        end

        y = f(x,args...)
        # y-tolerance.
        if abs(y) < ftol
            return x
        end
        x3 = x2
        x2 = x1
        if sign(y0) != sign(y)
            x1 = x
            y1 = y
        else
            x0 = x
            y0 = y
        end
        if abs(y0) < abs(y1)
            # Swap lower and upper bounds.
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        end
    end
    error("Max iteration exceeded")
end


"""
INPUTS:
    - precond: preconditioner for making Brent algorithm converge faster,
               typically an approximation to the inverse of the CDF function
"""
function quantile(cdf::Function, support::Array{T,1}; pdf = nothing,
  pdf_deriv=nothing, p::T=.5, quantile_bound=nothing, verbose=false,
  loose_bound=false, quantile_method = "0order", precond = nothing,
  xtol = nothing, ftol = nothing) where T<:Float64
    # print("Calling quantile function in statistics.jl")

    bound = quantile_bound == nothing ? support : quantile_bound(p)
    to_q = TimerOutput()
    # try
    #   bound = quantbound(p)
    # catch err
    # end
    # if loose_bound
    #   bound = [bound[1]-1.0, bound[2]+1.0]
    # end
    if precond != nothing
        cdf = x -> precond(cdf(x))
        p = precond(p)
    end

    @timeit to_q "quant_time" begin
      try
        global quant
        if quantile_method == "0order"
          #quant = fzero(y0 -> cdf(y0) - p, bound[1], bound[2])
          quant = brent(y0 -> cdf(y0) - p, bound[1], bound[2]) #use Brent's method
        elseif quantile_method == "1order"
          if pdf == nothing
            error("pdf must be provided for first order quantile-finding method")
          end
          init_guess = (bound[1] + bound[2])/2
          quant = find_zero(x -> (cdf(x), cdf(x)/pdf(x)), init_guess, Roots.Newton())
        elseif quantile_method == "brent"
            # print("using brent in quantile in statistics.jl")
            if xtol == nothing && ftol == nothing
                # print("Not using ftol or xtol in statistics/quantile")
                quant = brent(y0 -> cdf(y0) - p, bound[1], bound[2])
            elseif xtol == nothing
                # print("Quantile-finding ftol is $ftol")
                quant = brent(y0 -> cdf(y0) - p, bound[1], bound[2], ftol = ftol)
            elseif ftol == nothing
                # print("Quantile finding xtol is $xtol")
                quant = brent(y0 -> cdf(y0) - p, bound[1], bound[2], xtol = xtol)
            else
                # print("Quantile finding xtol and ftol are $xtol and $ftol")
                quant = brent(y0 -> cdf(y0) - p, bound[1], bound[2], xtol = xtol, ftol = ftol)
            end
            #quant = brent(y0 -> cdf(y0) - p, bound[1], bound[2])
        elseif quantile_method == "fzero"
            # print("using fzero in quantile in statistics.jl")
            if xtol == nothing && ftol == nothing
                # print("Not using ftol or xtol in statistics/quantile")
                quant = fzero(y0 -> cdf(y0) - p, bound[1], bound[2])
            elseif xtol == nothing
                # print("Quantile-finding ftol is $ftol")
                quant = fzero(y0 -> cdf(y0) - p, bound[1], bound[2], ftol = ftol)
            elseif ftol == nothing
                # print("Quantile finding xtol is $xtol")
                quant = fzero(y0 -> cdf(y0) - p, bound[1], bound[2], xtol = xtol)
            else
                # print("Quantile finding xtol and ftol are $xtol and $ftol")
                quant = fzero(y0 -> cdf(y0) - p, bound[1], bound[2], xtol = xtol, ftol = ftol)
            end
        else
          error("quantile finding method not defined: $quantile_method")
        end
      catch e_quant
        @show e_quant, p, bound
      end
    end
    err = abs(p-cdf(quant))/p
    # status = err < 1e-5 ? 1 : 0

    # return quant, err, bound
    timecost = round.(TimerOutputs.time(to_q["quant_time"])/1e9; digits=5)
    if verbose
      # println("Computing quantile, p=$(p), bound = $(round.(bound; digits=3)), quant=$(round(quant; digits=3)), time=$timecost")
        println("Computing quantile, p=$(p), bound = $bound, quant=$quant, time=$timecost")
    end
    p_rounded = nothing
    bound_rounded = nothing
    quant_rounded = nothing
    try
        p_rounded = round(p; digits=3)
        bound_rounded = round.(bound; digits=3)
        quant_rounded = round(quant; digits=3)
        return quant, [p_rounded, bound_rounded, quant_rounded, timecost]
    catch e
        @show p_rounded
        @show bound_rounded
        @show quant_rounded
    end
end

function mode(pdf::Function, support::Array{T,1}; cdf = nothing, pdf_deriv=nothing) where T<:Float64
    # maximize the pdf
    routine = optimize(x -> -pdf(x), support[1], support[2])
    mod = Optim.minimizer(routine)
    return mod
end

function credible_interval(cdf::Function, support::Array{T,1};
                            pdf=nothing, pdf_deriv=nothing, wp::T=.95, mode=:equal,
                            quantile_bound=nothing, verbose=false, loose_bound=false,
                            quantile_method = "0order", xtol = nothing, ftol = nothing) where T<:Float64
    return credible_interval(cdf, support, Val(mode); pdf_deriv=pdf_deriv, wp=wp,
    quantile_bound=quantile_bound, verbose=verbose, loose_bound=loose_bound,
    quantile_method = quantile_method, xtol = xtol, ftol = ftol)
end

function credible_interval(cdf::Function, support::Array{T,1}, ::Val{:equal};
                            pdf=nothing, pdf_deriv=nothing, wp::T=.95,
                            quantile_bound=nothing, verbose=false, loose_bound=false,
                            quantile_method = "0order", xtol = nothing, ftol = nothing) where T<:Float64
    lower_qp = (1 - wp) / 2
    upper_qp = 1 - lower_qp
    lower_quant = lower_qp # random initialization
    quant_info1 = nothing
    quant_info2 = nothing
    try
        lower_quant, quant_info1 = quantile(cdf, support; p=lower_qp,
        quantile_bound=quantile_bound, verbose=verbose, loose_bound=loose_bound,
        quantile_method = quantile_method, xtol = xtol, ftol = ftol)
    catch err
        @show err
        wp = .9
        lower_qp = (1 - wp) / 2
        upper_qp = 1 - lower_qp
        lower_quant, quant_info1 = quantile(cdf, support; p=lower_qp, quantile_bound=quantile_bound,
        verbose=verbose, loose_bound=loose_bound, quantile_method = quantile_method, xtol = xtol, ftol = ftol)
        @warn "Can't compute 95% credible interval, do $(Int(wp*100))% instead."
    end
    upper_quant, quant_info2 = quantile(cdf, support; p=upper_qp,
    quantile_bound=quantile_bound, verbose=verbose, loose_bound=loose_bound,
    quantile_method = quantile_method, xtol = xtol, ftol = ftol)
    err = abs(cdf(upper_quant) -  cdf(lower_quant) - wp)/wp
    # return [lower_quant, upper_quant], wp, err
    quant_info_CI = [[quant_info1, quant_info2]]
    return [lower_quant, upper_quant], quant_info_CI
end

function credible_interval(cdf::Function, support::Array{T,1}, ::Val{:narrow};
                            pdf=nothing, pdf_deriv=nothing, wp::T=.95, quantile_bound=nothing, verbose=false, loose_bound=false) where T<:Float64
  #=
  Brief idea: bisection
    Suppose the target interval is [alpha*, beta*], i.e. integral of pdf on [alpha*, beta*] = wp
    and the corresponding height is pdf(alpha*) = pdf(beta*) = l_height*
    Say l* is the horizontal line intersecting with pdf curve at alpha* and beta*,
    we first find two horizontal l lines sitting on each side of l*
    One is lower with height l_height_low, and one is higher with height l_height_high
    Then we use a bisection way to gradually find the target l*
  =#

  #= Notations:
  [alpha, beta]:
      interval for integration, target credible interval
      s.t. 1) pdf(alpha) = pdf(beta) and 2) integral of pdf from alpha to beta = wp (like 0.95)
  bound:
      the bound of support of the pdf, currently assume this is available
      i.e. pdf(x) = 0 if z > bound
  l_height_low/l_height_high:
      height of the l line that is lower/higher than the target one
  alpha_low/alpha_high, beta_low/beta_high:
      the corresponding interval
  int_low/int_high:
      the corresponding integral value, i.e. integral of pdf on [alpha_low/high, beta_low/high]
  hquadrature:
      function from the package Cubature for numerical integration
      e.g. integral_heightue, error = hquadrature(f, a, b) gives integral of f on [a,b] with some error
      we mostly use this to compute integral of pdf on [alpha, beta] for different alpha and beta
  =#

 # helper functions
  #=
  given a height l_height, find the two intersections s.t. pdf(α) = pdf(β) = l_height
  Input:
    l_height: given height
    α_intvl/β_intvl: interval for root finding for alpha/beta
  Output:
    [α, β]: the two intersections
    int: integral value of pdf on [α, β]
  =#
  function find_αβ(l_height, α_intvl, β_intvl)
    # find α and β within ginve intervals α_intvl and β_intvl respectively
    temp_fun(x) = pdf(x) - l_height
    α = fzero(temp_fun,  α_intvl[1], α_intvl[2])
    β = fzero(temp_fun,  β_intvl[1], β_intvl[2])
    int = hquadrature(x -> pdf(x), α, β)[1]
    return α, β, int
  end


  #=
  Adjust the height in case initial choice of low/high is not proper
    Since we want int_low > wp and int_high < wp to do bisection,
    we have to adjust height if initial int_low < wp or int_high > wp
  Input:
    l: current height
    [α, β]: current interval
    int: current integral value
    MODE: indicates we are adjusting the low line or the high line
  Output:
    According values after adjustment
    s.t. int > wp if MODE == "low"; int < wp else.
 =#
  function adjust(l, α, β, int, MODE)
    # adjust if int < wp
    if MODE == "low"
        while int < wp
            l /= 2
            α, β, int = find_αβ(l, [support[1], α], [β, support[2]])
        end
    else
        while int > wp
            l = (l + pdf(mode_d))/2
            α, β, int = find_αβ(l, [α, mode_d], [mode_d, β])
        end
    end
    return l, α, β, int
  end
  mode_d = mode(pdf, cdf, support)
  l_height_low = pdf(support[1])
  α_low = support[1]
  β_low = fzero(x -> pdf(x) - l_height_low,  mode_d, support[2])
  int_low = hquadrature(x -> pdf(x), α_low, β_low)[1]
  # α_low, β_low, int_low = find_αβ(l_height_low, [support[1], mode_d], [mode_d, support[2]])
  # adjust height if l low is not lower than l* (i.e. int < wp)
  # l_height_low, α_low, β_low, int_low = adjust(l_height_low, α_low, β_low, int_low, "low")
  l_height_high = pdf(mode_d)*0.9
  α_high, β_high, int_high = find_αβ(l_height_high, [α_low, mode_d], [mode_d, β_low])
  # adjust height if l_high is not higher than l* (i.e. int > wp)
  l_height_high, α_high, β_high, int_high = adjust(l_height_high, α_high, β_high, int_high, "high")

  α_mid = 0.
  β_mid = 0.
  l_mid = 0.
  N = 0
  int = int_low
  while !isapprox(int, wp) && N < 50
  l_mid = (l_height_high + l_height_low)/2
  α_mid, β_mid, int = find_αβ(l_mid, [α_low, α_high], [β_high, β_low])
  if int > wp
      l_height_low = l_mid
      α_low = α_mid
      β_low = β_mid
  else
      l_height_high = l_mid
      α_high = α_mid
      β_high = β_mid
  end
  N += 1
  end
  err1 = abs(cdf(β_mid) -  cdf(α_mid) - wp)/wp
  err = max(err1, abs(pdf(α_mid)- pdf(β_mid))/abs(pdf(β_mid)))
  return ([α_mid, β_mid], err)

end


# @doc raw"""
# """
# function map_estimate(btg::BTG)
#     # TODO
# end

# @doc raw"""
# """
# function cross_validate(btg::BTG)
#     # TODO
# end
