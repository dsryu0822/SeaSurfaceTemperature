using HTTP, NCDatasets, DataFrames, Dates, CSV, JLD2, StatsBase
for y = 2019:2024
    url = replace("https://ncss.hycom.org/thredds/ncss/GLBy0.08/expt_93.0/ts3z/$y?var=water_temp
    &north=36.88
    &west=129.52
    &east=130.24
    &south=36.32
    &disableProjSubset=on
    &horizStride=1
    &time_start=$y-01-01T00%3A00%3A00Z
    &time_end=$y-12-31T21%3A00%3A00Z
    &timeStride=1
    &vertCoord=0
    &accept=netcdf4", '\n' => "", " " => "")
    @time response = HTTP.get(url)

    open("data_$y.nc4", "w") do f
        write(f, response.body)
    end
end

# ds  = NCDataset("data_$y.nc4")
# # ds.attrib
# # keys(ds)
# # ds["water_temp"][:, :, 1, :]
# CSV.write("lon.csv", DataFrame(i = 1:10, x = round.(ds["lon"][:], digits = 2)))
# CSV.write("lat.csv", DataFrame(j = 1:15, y = round.(ds["lat"][:], digits = 2)))
# close(ds)

data_ = []
for y = 2019:2024
    ds  = NCDataset("data_$y.nc4")
    push!(data_, DataFrame([ds["time"][:] reshape(ds["water_temp"][:, :, 1, :], 150, :)'], ["t"; repeat("i" .* string.(1:10), outer = 15) .* repeat("j" .* string.(1:15), inner = 10)]))
    close(ds)
end
_data_ = vcat(data_...)
_data_.t = Date.(_data_.t)
_data_ = combine(groupby(_data_, :t), names(_data_, Not(:t)) .=> mean .=> names(_data_, Not(:t)))
CSV.write("data_GLBy0.08_expt_93.0.csv", _data_)
tnsr = reshape(Matrix(_data_[:, 2:end])', 10, 15, :)

@save "data_tnsr.jld2" tnsr
# @load "data_tnsr.jld2"; tnsr
