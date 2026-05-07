using HTTP, NCDatasets, DataFrames, Dates, CSV, JLD2, StatsBase, ProgressMeter
using Base.Threads
cd("G:/nino/")

# East Sea
        # &north=36.88
        # &west=129.52
        # &east=130.24
        # &south=36.32
# Nino3.4(wrong)
        # &north=5
        # &west=120
        # &east=170
        # &south=-5
# Nino3.4
        # &north=5
        # &west=190
        # &east=240
        # &south=-5


# @showprogress @threads for day in Date(2019, 1, 1):Date(2024, 9, 5)
#     experiment = "https://ncss.hycom.org/thredds/ncss/GLBy0.08/expt_93.0/ts3z/$(Year(day).value)"
#     if day < Date(2015, 12, 30)
#         experiment = "https://ncss.hycom.org/thredds/ncss/GLBv0.08/expt_53.X/data/$(Year(day).value)"
#     elseif Date(2014, 1, 1) ≤ day ≤ Date(2016, 4, 30)
#         experiment = "https://ncss.hycom.org/thredds/ncss/GLBv0.08/expt_56.3"
#     elseif Date(2016, 5, 1) ≤ day ≤ Date(2017, 1, 31)
#         experiment = "https://ncss.hycom.org/thredds/ncss/GLBv0.08/expt_57.2"
#     elseif Date(2017, 2, 1) ≤ day ≤ Date(2017, 5, 31)
#         experiment = "https://ncss.hycom.org/thredds/ncss/GLBv0.08/expt_92.8"
#     elseif Date(2017, 6, 1) ≤ day ≤ Date(2017, 9, 30)
#         experiment = "https://ncss.hycom.org/thredds/ncss/GLBv0.08/expt_57.7"
#     elseif Date(2017, 10, 1) ≤ day ≤ Date(2017, 12, 31)
#         experiment = "https://ncss.hycom.org/thredds/ncss/GLBv0.08/expt_92.9"
#     elseif Date(2018, 1, 1) ≤ day ≤ Date(2018, 12, 31)
#     end
#     try
#         str_day = string(day)
#         ("data_$str_day.nc4" |> isfile) && continue
#         url = replace("$(experiment)?var=water_temp
#         &north=5
#         &west=190
#         &east=240
#         &south=-5
#         &disableProjSubset=on
#         &horizStride=1
#         &time_start=$(day)T00%3A00%3A00Z
#         &time_end=$(day)T21%3A00%3A00Z
#         &timeStride=1
#         &vertCoord=0
#         &accept=netcdf4", '\n' => "", " " => "")
#         # @time response = HTTP.get(url)
#         response = HTTP.get(url)

#         open("data_$str_day.nc4", "w") do f
#             write(f, response.body)
#         end
#     catch e
#         @warn e
#     end
# end

nc4__ = filter(endswith(".nc4"), readdir())
@threads for K in 1:10
    nc4_ = nc4__[K:K:end]
# colnames = ["t"; repeat("i" .* string.(1:10), outer = 15) .* repeat("j" .* string.(1:15), inner = 10)]
# colnames = ["t"; repeat("i" .* string.(1:10), outer = 8) .* repeat("j" .* string.(1:8), inner = 10)]
colnames = ["t"; repeat("i" .* string.(1:626), outer = 251) .* repeat("j" .* string.(1:251), inner = 626)]
# data_ = DataFrame([zeros(length(nc4_)) for _ in eachindex(colnames)], colnames)
data_ = DataFrame([[] for _ in eachindex(colnames)], colnames)
for _ in eachindex(nc4_) push!(data_, zeros(length(colnames))) end
data_.t = Date.(data_.t)
for k in eachindex(nc4_)
    try
        NCDataset(nc4_[k]) do ds
            df = DataFrame([ds["time"][:] reshape(ds["water_temp"][:, :, 1, :], 251*626, :)'], colnames)
            # df = DataFrame([ds["time"][:] reshape(ds["water_temp"][:, :, 1, :], 10*8, :)'], colnames)
            df.t = Date.(df.t)
            data_[k, :] = combine(groupby(df, :t), names(df, Not(:t)) .=> mean .=> names(df, Not(:t)))[1, :]
        end
    catch e
        @warn "$(nc4_[k])"
    end
    CSV.write("data_GLBv0.08_$K.csv", data_)
end
end
tnsr = reshape(Matrix(data_[:, 2:end])', 626, 251, :)

@save "data_tnsr.jld2" tnsr
# @load "data_tnsr.jld2"; tnsr