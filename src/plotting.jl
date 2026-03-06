module Visualization

using Plots

export plot_results

"""
    plot_results(snapshots, thrust_time, thrust_values, save_times)

Строит графики профилей n_a, n_i, v_z, E_z для сохранённых моментов времени,
а также график изменения силы тяги F_T во времени.
"""
function plot_results(snapshots, thrust_time, thrust_values, save_times)
    if !isempty(snapshots)
        times = sort(collect(keys(snapshots)))
        p1 = plot(title="n_a", xlabel="z")
        p2 = plot(title="n_i", xlabel="z")
        p3 = plot(title="v_z", xlabel="z")
        p4 = plot(title="E_z", xlabel="z")
        colors = palette(:tab10, length(times))
        
        for (idx, t) in enumerate(times)
            z, n_a, n, v_z, E_z = snapshots[t]
            plot!(p1, z, n_a, label="t=$t", color=colors[idx])
            plot!(p2, z, n, label="t=$t", color=colors[idx])
            plot!(p3, z, v_z, label="t=$t", color=colors[idx])
            plot!(p4, z, E_z, label="t=$t", color=colors[idx])
        end
        
        plot(p1, p2, p3, p4, layout=4, size=(800,600))
        savefig("profiles.png")
        display(current())
    end
    
    if !isempty(thrust_time)
        p_thrust = plot(thrust_time, thrust_values, xlabel="t", ylabel="F_T", legend=false)
        savefig("thrust.png")
        display(p_thrust)
    end
end

end # module