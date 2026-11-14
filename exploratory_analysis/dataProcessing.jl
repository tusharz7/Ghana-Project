data = Array{Float32}(undef, 1376, 385)
read!("data/Depth.data", data)
data = ntoh.(data)  # convert big-endian to native

lon  = ntoh.(reinterpret(Float32, read("data/XC.data"))) |> x -> reshape(x, 1376, 385)
lat  = ntoh.(reinterpret(Float32, read("data/YC.data"))) |> x -> reshape(x, 1376, 385)
rc   = ntoh.(reinterpret(Float32, read("data/RC.data")))   # 100 depth levels (negative meters)
depth = ntoh.(reinterpret(Float32, read("data/Depth.data"))) |> x -> reshape(x, 1376, 385)

using Plots

# mask land as NaN
depth_plot = Float32.(depth')
depth_plot[depth_plot .== 0f0] .= NaN

# non-uniform colormap: half the colors cover 0–500m, half cover 500–4500m
max_depth = 4500f0
frac = 500.0 / max_depth
n    = 256
base = cgrad(:viridis)
cs   = [base[i / (n-1)] for i in 0:n-1]
pos  = vcat(range(0.0, frac, length=n÷2+1),
            range(frac, 1.0, length=n÷2+1)[2:end]) |> collect
cmap = cgrad(cs, pos)

heatmap(lon[:,1], lat[1,:], depth_plot,
    title="Depth (m)",
    color=cmap,
    clims=(0f0, max_depth),
    nan_color=:gray)
xlabel!("Longitude"); ylabel!("Latitude")