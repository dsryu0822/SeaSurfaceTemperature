using HTTP, NCDatasets, DataFrames, Dates, CSV, JLD2, StatsBase, ProgressMeter
using Base.Threads
cd("G:/seasurface/nino34u")

# Never put 0 in the `Date` function
@showprogress @threads for day in (Date(2024, 8, 10):Date(2026, 8, 5))
    # experiment = "https://ncss.hycom.org/thredds/ncss/ESPC-D-V02/t3z/$(Year(day).value)?var=water_temp"
    experiment = "https://ncss.hycom.org/thredds/ncss/ESPC-D-V02/u3z/$(Year(day).value)?var=water_u"
    # experiment = "https://ncss.hycom.org/thredds/ncss/ESPC-D-V02/v3z/$(Year(day).value)?var=water_v"
    try
        str_day = string(day)
        ("data_$str_day.nc4" |> isfile) && continue
        url = replace("$(experiment)
        &north=5
        &west=190
        &east=240
        &south=-5
        &disableProjSubset=on
        &horizStride=1
        &time_start=$(day)T00%3A00%3A00Z
        &time_end=$(day)T21%3A00%3A00Z
        &timeStride=1
        &vertCoord=0
        &accept=netcdf4", '\n' => "", " " => "")
        # @time response = HTTP.get(url)
        response = HTTP.get(url)

        open("data_$str_day.nc4", "w") do f
            write(f, response.body)
        end
    catch e
        @warn e
    end
end


nc4_ = filter(endswith(".nc4"), readdir("G:/seasurface/nino34t"))
ds = NCDataset(nc4_[1])
    nX, nY, _, _ = size(ds[key])
    id_missing = findall(ismissing.(ds[key][:, :, 1, 1][:]))
close(ds)
colnames = repeat("i" .* string.(1:nX), outer = nY) .* repeat("j" .* string.(1:nY), inner = nX)

nc4_ = filter(endswith(".nc4"), readdir("G:/seasurface/nino34t", join = true))
key = NCDataset(nc4_[1]) do ds first(keys(ds)) end
tnsrT = Array{Union{Missing, Float64}, 3}(zeros(nY, nX, length(nc4_)))
@showprogress for k in eachindex(nc4_)
    NCDataset(nc4_[k]) do ds
        tnsrT[:, :, k] = mean(ds[key][:, :, 1, :], dims = 3)[:, :]'
    end
end

nc4_ = filter(endswith(".nc4"), readdir("G:/seasurface/nino34u", join = true))
key = NCDataset(nc4_[1]) do ds first(keys(ds)) end
tnsrU = Array{Union{Missing, Float64}, 3}(zeros(nY, nX, length(nc4_)))
@showprogress for k in eachindex(nc4_)
    NCDataset(nc4_[k]) do ds
        tnsrU[:, :, k] = mean(ds[key][:, :, 1, :], dims = 3)[:, :]'
    end
end
nc4_ = filter(endswith(".nc4"), readdir("G:/seasurface/nino34v", join = true))
key = NCDataset(nc4_[1]) do ds first(keys(ds)) end
tnsrV = Array{Union{Missing, Float64}, 3}(zeros(nY, nX, length(nc4_)))
@showprogress for k in eachindex(nc4_)
    NCDataset(nc4_[k]) do ds
        tnsrV[:, :, k] = mean(ds[key][:, :, 1, :], dims = 3)[:, :]'
    end
end


∂x(M) = (circshift(M, (0, -1)) - circshift(M, (0, 1)))
∂y(M) = (circshift(M, (-1, 0)) - circshift(M, (1, 0)))
∂xx(M) = (circshift(M, (0, -1)) - 2M + circshift(M, (0, 1)))
∂yy(M) = (circshift(M, (-1, 0)) - 2M + circshift(M, (1, 0)))

i_ = collect(1:5:nY)
j_ = collect(1:5:nX)
i_i = UnitRange.(i_[1:end-1], (i_ .- 1)[2:end])
j_j = UnitRange.(j_[1:end-1], (j_ .- 1)[2:end])

@time for M = ["T", "U", "V"]
"""
begin
    _$(M) = zeros(length(i_i), length(j_j), length(nc4_))
    _$(M)x = zeros(length(i_i), length(j_j), length(nc4_))
    _$(M)y = zeros(length(i_i), length(j_j), length(nc4_))
    _$(M)xx = zeros(length(i_i), length(j_j), length(nc4_))
    _$(M)yy = zeros(length(i_i), length(j_j), length(nc4_))

    for ijk in eachindex(IndexCartesian(), _$M)
        i, j, k = ijk.I
        matrix = tnsr$(M)[i_i[i], j_j[j], k]
        _$(M)[i, j, k] = mean(skipmissing(matrix))
        _$(M)x[i, j, k] = mean(skipmissing(∂x(matrix)[2:end-1, 2:end-1]))
        _$(M)y[i, j, k] = mean(skipmissing(∂y(matrix)[2:end-1, 2:end-1]))
        _$(M)xx[i, j, k] = mean(skipmissing(∂xx(matrix)[2:end-1, 2:end-1]))
        _$(M)yy[i, j, k] = mean(skipmissing(∂yy(matrix)[2:end-1, 2:end-1]))
    end
    _d$(M) = diff(_$(M), dims = 3)
    _$(M) = _$(M)[:, :, 1:end-1]
    _$(M)x = _$(M)x[:, :, 1:end-1]
    _$(M)y = _$(M)y[:, :, 1:end-1]
    _$(M)xx = _$(M)xx[:, :, 1:end-1]
    _$(M)yy = _$(M)yy[:, :, 1:end-1]
end
""" |> Meta.parse |> eval
end


D = Date.([last(nc4)[6:15] for nc4 in split.(nc4_, "\\")])[1:end-1]
Y = range(-5, 5, length = size(_T, 1)) |> collect
X = range(-170, -120, length = size(_T, 2)) |> collect

# JLD2.@save "G:/seasurface/nino34tuv.jld2" D X Y _T _U _V _dT _dU _dV _Tx _Ty _Txx _Tyy _Ux _Uy _Uxx _Uyy _Vx _Vy _Vxx _Vyy
@time JLD2.@load "G:/seasurface/nino34tuv.jld2"
