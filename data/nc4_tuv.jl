using HTTP, NCDatasets, DataFrames, Dates, CSV, JLD2, StatsBase, ProgressMeter
using Base.Threads
cd("G:/seasurface/nino34t")

@showprogress @threads for day in (Date(2026, 5, 15):Date(2026, 7, 21))
    experiment = "https://ncss.hycom.org/thredds/ncss/ESPC-D-V02/t3z/$(Year(day).value)?var=water_temp"
    # experiment = "https://ncss.hycom.org/thredds/ncss/ESPC-D-V02/u3z/$(Year(day).value)?var=water_u"
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
key = first(keys(ds))
nX, nY, _, _ = size(ds[key])
id_missing = findall(ismissing.(ds[key][:, :, 1, 1][:]))
close(ds)
colnames = repeat("i" .* string.(1:nX), outer = nY) .* repeat("j" .* string.(1:nY), inner = nX)

tnsr = Array{Union{Missing, Float64}, 3}(zeros(nY, nX, length(nc4_)))
@showprogress for k in eachindex(nc4_)
    NCDataset(nc4_[k]) do ds
        tnsr[:, :, k] = mean(ds[key][:, :, 1, :], dims = 3)[:, :]'
    end
end
D = Date.([nc4[6:15] for nc4 in nc4_])
X = [-5:0.04:5...;]
Y = [-170:0.08:-120...;]

i_ = collect(1:10:nY); hY = 1/10
j_ = collect(1: 5:nX); hX = 1/5
i_i = UnitRange.(i_, circshift(i_, -1))[1:(end-1)]
j_j = UnitRange.(j_, circshift(j_, -1))[1:(end-1)]
T = zeros(length(i_i), length(j_j), length(nc4_)); T |> size
Ty = zeros(length(i_i), length(j_j), length(nc4_))
Tx = zeros(length(i_i), length(j_j), length(nc4_))
Tyy = zeros(length(i_i), length(j_j), length(nc4_))
Txx = zeros(length(i_i), length(j_j), length(nc4_))

for ijk in eachindex(IndexCartesian(), T)
    i, j, k = ijk.I
    matrix = tnsr[i_i[i], j_j[j], k]
    T[i, j, k] = mean(skipmissing(matrix))
    Tx[i, j, k] = mean(skipmissing((circshift(matrix, (0, -1)) - circshift(matrix, (0, 1)))[2:end-1, 2:end-1])) / (2hX)
    Ty[i, j, k] = mean(skipmissing((circshift(matrix, (-1, 0)) - circshift(matrix, (1, 0)))[2:end-1, 2:end-1])) / (2hY)
    Txx[i, j, k] = mean(skipmissing((circshift(matrix, (0, -1)) - 2matrix + circshift(matrix, (0, 1)))[2:end-1, 2:end-1])) / hX^2
    Tyy[i, j, k] = mean(skipmissing((circshift(matrix, (-1, 0)) - 2matrix + circshift(matrix, (1, 0)))[2:end-1, 2:end-1])) / hY^2
end

# JLD2.@save "G:/seasurface/nino34tuv.jld2" D X Y T Tx Ty Txx Tyy U Ux Uy Uxx Uyy V Vx Vy Vxx Vyy
@time JLD2.@load "G:/seasurface/nino34tuv.jld2"