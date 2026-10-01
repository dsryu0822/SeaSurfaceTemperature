using HTTP, JSON3, NCDatasets, DataFrames, Dates, CSV, JLD2, StatsBase, ProgressMeter
using Base.Threads
const EDL_TOKEN = "eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6ImRzcnl1MDgyMiIsImV4cCI6MTc5MTM1OTQzNywiaWF0IjoxNzg2MTc1NDM3LCJpc3MiOiJodHRwczovL3Vycy5lYXJ0aGRhdGEubmFzYS5nb3YiLCJpZGVudGl0eV9wcm92aWRlciI6ImVkbF9vcHMiLCJhY3IiOiJlZGwiLCJhc3N1cmFuY2VfbGV2ZWwiOjN9.ufHL77aGr9lYMovXlIdSDYXz4IwasxoIDexVPR6UcuHYRyBd5b9XJf8r9tIoHZcp65qu2niUApcw61023lR6ngIpVFwCWYvMt2nzhFqmcL0-oE_pVyXvO9JpZiKZMqxrF6Y6K6Tb-QwbrpyEmrLZP2-FxPW56eJZ3GpfjnJCVgcHFZsm7aXq9V5vClqay9pda1nT-25xPk8_tz5Cpelmwce1ZTejB1YeA73LX2pC8I7ykjNa5NHQyG0ugtkni0LzTa93ETigihWurLCb-Hnu0CuYJMqR8n1bFgxSqG9yLj64FQNnkhWYakp2wdzeo09lBY3-h5sPrGEx2lVrx-Sxtw"
const AUTH_HEADER = ["Authorization" => "Bearer $EDL_TOKEN"]
# const OSCAR25_CID = "C2102959417-POCLOUD" # interim
const OSCAR25_CID = "C2098858642-POCLOUD" # final
# const MUR_CID = "C2036880657-POCLOUD" # 0.25
const MUR_CID = "C1996881146-POCLOUD" # 0.01

function GHRSSTMUR(filename, day)
    cmr = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
    d1 = DateTime(day)
    d2 = d1 + Day(1)

    q = ["collection_concept_id"=>MUR_CID,
         "temporal"=>"$(d1)Z,$(d2)Z",
         "page_size"=>"10"]

    j = JSON3.read(HTTP.get(cmr; query=q, headers=AUTH_HEADER).body)

    tag = Dates.format(day, "yyyymmdd")
    urls = [u.URL for g in j.items for u in g.umm.RelatedUrls
            if u.Type == "GET DATA" && occursin(tag, u.URL)]

    isempty(urls) && error("No MUR granule found for $day")

    r = HTTP.get(urls[1]; headers=AUTH_HEADER)
    tmp = tempname() * ".nc"

    open(tmp, "w") do f
        write(f, r.body)
    end

    ds = NCDataset(tmp)

    lat = ds["lat"][:]
    lon = ds["lon"][:]
    sst0 = ds["analysed_sst"][:, :, 1]

    # OSCAR와 동일한 0.25° grid
    target_lon = collect(-170.0:0.25:-120.0)
    target_lat = collect(-5.0:0.25:5.0)

    ilon = [argmin(abs.(lon .- x)) for x in target_lon]
    ilat = [argmin(abs.(lat .- x)) for x in target_lat]

    # 선택된 격자점에서 일간 SST
    sst = Float64.(sst0[ilon, ilat]) .- 273.15

    close(ds)
    rm(tmp)

    # lon × lat × 1
    sst = reshape(sst, length(target_lon), length(target_lat), 1)

    NCDataset(filename, "c") do out
        defDim(out, "lon", length(target_lon))
        defDim(out, "lat", length(target_lat))
        defDim(out, "time", 1)

        defVar(out, "lon", target_lon, ("lon",))
        defVar(out, "lat", target_lat, ("lat",))
        defVar(out, "time", [d1], ("time",))

        defVar(out, "analysed_sst", sst,
               ("lon", "lat", "time"))

        out["analysed_sst"].attrib["units"] = "degree_Celsius"
    end
end


day_ = setdiff((Date(2020, 1, 1):Date(2026, 1, 16)),
       [Date(d[1:end-4]) for d in readdir("G:/seasurface/podaacT")])
@showprogress for day in day_
    try
        if !isfile("G:/seasurface/podaacT/$day.nc4")
            GHRSSTMUR("G:/seasurface/podaacT/$day.nc4", day)
        else
            @info "Skipping GHRSSTMUR for $day"
        end
    catch e
        @warn "GHRSSTMUR failed for $day: $e"
    end
end


function OSCAR25(filename, day)
    cmr = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
    cid = OSCAR25_CID

    t1 = DateTime(day)
    t2 = t1 + Day(1) - Second(1)

    q = ["collection_concept_id"=>cid,
         "temporal"=>"$(t1)Z,$(t2)Z",
         "page_size"=>"10"]

    j = JSON3.read(HTTP.get(cmr; query=q, headers=AUTH_HEADER).body)

    tag = Dates.format(day, "yyyymmdd")
    urls = [u.URL for g in j.items for u in g.umm.RelatedUrls
            if u.Type == "GET DATA" && occursin(tag, u.URL)]

    isempty(urls) && error("No OSCAR granule found for $day")

    r = HTTP.get(urls[1]; headers=AUTH_HEADER)
    tmp = tempname() * ".nc"

    open(tmp, "w") do f
        write(f, r.body)
    end

    ds = NCDataset(tmp)

    lat = ds["lat"][:]
    lon = ds["lon"][:]

    # Nino3.4: 5°S–5°N, 170°W–120°W
    ilat = findall((-5 .<= lat) .& (lat .<= 5))
    ilon = findall((190 .<= lon) .& (lon .<= 240))

    # 원본: (lat, lon, time)
    u = Array(ds["ug"][ilat, ilon, 1])
    v = Array(ds["vg"][ilat, ilon, 1])

    close(ds)
    rm(tmp)

    # (lat, lon) → (lon, lat) → (lon, lat, 1)
    u = reshape(permutedims(u, (2, 1)), length(ilon), length(ilat), 1)
    v = reshape(permutedims(v, (2, 1)), length(ilon), length(ilat), 1)

    lon_out = lon[ilon] .- 360

    NCDataset(filename, "c") do out
        defDim(out, "lon", length(ilon))
        defDim(out, "lat", length(ilat))
        defDim(out, "time", 1)

        defVar(out, "lon", lon_out, ("lon",))
        defVar(out, "lat", lat[ilat], ("lat",))
        defVar(out, "time", [string(day)], ("time",))

        defVar(out, "u", u, ("lon", "lat", "time"))
        defVar(out, "v", v, ("lon", "lat", "time"))

        out["u"].attrib["units"] = "m s-1"
        out["v"].attrib["units"] = "m s-1"
    end
end

@showprogress for day in (Date(2020, 1, 1):Date(2026, 8, 1))
    try
        if !isfile("G:/seasurface/podaacUV/$day.nc4")
            OSCAR25("G:/seasurface/podaacUV/$day.nc4", day)
        end
    catch e
        @warn "OSCAR25 failed for $day: $e"
    end
end


days = Date.(first.(split.(readdir("G:/seasurface/podaacT"), ".")))
temp = NCDataset("G:/seasurface/podaacT/2020-01-01.nc4")
D = days
X = temp["lon"][:]
Y = temp["lat"][:]
nX = length(X)
nY = length(Y)
close(temp)

nino34T = DataFrame(zeros(length(days), nX*nY), ["k$k" for k in 1:nX*nY])
nino34U, nino34V = deepcopy(nino34T), deepcopy(nino34T)
T = zeros(nY, nX, length(days))
U, V = deepcopy(T), deepcopy(T)
@showprogress for t in eachindex(days)
    day = days[t]
    filenameT = "G:/seasurface/podaacT/$(string(day)).nc4"
    filenameUV = "G:/seasurface/podaacUV/$(string(day)).nc4"

    NCDataset(filenameT) do ds
        T[:, :, t] = ds["analysed_sst"][:, :, 1]'
        # nino34T[t, :] = T[:, :, t]
        nino34T[t, :] = ds["analysed_sst"][:, :, 1][:]'
    end
    NCDataset(filenameUV) do ds
        U[:, :, t] = ds["u"][:, :, 1]'
        V[:, :, t] = ds["v"][:, :, 1]'
        nino34U[t, :] = U[:, :, t]
        nino34V[t, :] = V[:, :, t]
    end
end

time = DataFrame(date = D)
CSV.write("G:/seasurface/nino34T.csv", [time nino34T])
CSV.write("G:/seasurface/nino34U.csv", [time nino34U])
CSV.write("G:/seasurface/nino34V.csv", [time nino34V])


∂x(M) = (circshift(M, (0, -1)) - circshift(M, (0, 1)))
∂y(M) = (circshift(M, (-1, 0)) - circshift(M, (1, 0)))
∂xx(M) = (circshift(M, (0, -1)) - 2M + circshift(M, (0, 1)))
∂yy(M) = (circshift(M, (-1, 0)) - 2M + circshift(M, (1, 0)))

_T, _U, _V = deepcopy(T), deepcopy(U), deepcopy(V)
@time for M = ["T", "U", "V"]
"""
begin
    _d$(M) = diff(_$(M), dims = 3)
    _$(M)x = ∂x(_$(M))[:, :, 1:end-1]
    _$(M)y = ∂y(_$(M))[:, :, 1:end-1]
    _$(M)xx = ∂xx(_$(M))[:, :, 1:end-1]
    _$(M)yy = ∂yy(_$(M))[:, :, 1:end-1]

    _$(M) = _$(M)[:, :, 1:end-1]
end
""" |> Meta.parse |> eval
end

JLD2.@save "G:/seasurface/nino34TUV.jld2" D X Y nX nY _dT _dU _dV _T _U _V _Tx _Ty _Txx _Tyy _Ux _Uy _Uxx _Uyy _Vx _Vy _Vxx _Vyy

