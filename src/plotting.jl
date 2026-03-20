module Visualization

using Plots
using Statistics

export plot_results

"""
    plot_results(snapshots, thrust_time, thrust_values, save_times, force_scale=1.0,
                 L_phys=1.0, v_char=1.0, n_char=1.0, t_char=1.0, mi=1.0, e_charge=1.6e-19)

Строит графики профилей и тяги.
- thrust_values: размерная тяга в мН
- thrust_time: время в мс
- Остальные величины пересчитываются в СИ внутри функции
"""
function plot_results(snapshots, thrust_time, thrust_values, save_times, force_scale=1.0;
                      L_phys=1.0, v_char=1.0, n_char=1.0, t_char=1.0, mi=1.0, e_charge=1.6e-19,
                      H_char=0.01172)
    if !isempty(snapshots)
        # Масштабы перевода в СИ
        E_char = mi * v_char^2 / (e_charge * L_phys)  # В/м

        times = sort(collect(keys(snapshots)))
        t_char_ms = t_char * 1000  # для меток времени: безразмерное → мс

        p1 = plot(title="Нейтральная плотность n_a", xlabel="z (м)", ylabel="n_a (10¹⁷ м⁻³)")
        p2 = plot(title="Ионная плотность n_i",      xlabel="z (м)", ylabel="n_i (10¹⁷ м⁻³)")
        p3 = plot(title="Скорость ионов v_z",        xlabel="z (м)", ylabel="v_z (км/с)")
        p4 = plot(title="Электрическое поле E_z",    xlabel="z (м)", ylabel="E_z (В/м)")
        p5 = plot(title="Магнитное поле H*",          xlabel="z (м)", ylabel="H (Тл)")
        colors = palette(:tab10, length(times))

        for (idx, t) in enumerate(times)
            z, n_a, n, v_z, E_z, H_total = snapshots[t]
            t_ms = round(t * t_char_ms, digits=3)
            lbl = "t=$(t_ms) мс"
            z_m   = z   .* L_phys           # безразм. → м
            n_a_p = n_a .* (n_char * 1e-17)  # м⁻³ → единицы 10¹⁷м⁻³
            n_p   = n   .* (n_char * 1e-17)
            vz_p  = v_z .* (v_char * 1e-3)  # м/с → км/с
            Ez_p  = E_z .* E_char            # безразм. → В/м
            H_p   = H_total .* H_char        # безразм. → Тл

            plot!(p1, z_m, n_a_p, label=lbl, color=colors[idx])
            plot!(p2, z_m, n_p,   label=lbl, color=colors[idx])
            plot!(p3, z_m, vz_p,  label=lbl, color=colors[idx])
            plot!(p4, z_m, Ez_p,  label=lbl, color=colors[idx])
            plot!(p5, z_m, H_p,   label=lbl, color=colors[idx])
        end

        plot(p1, p2, p3, p4, p5, layout=(3,2), size=(1100, 900))
        savefig("profiles.png")
        display(current())
    end

    if !isempty(thrust_time)
        p_thrust = plot(thrust_time, thrust_values,
                       xlabel="t (мс)", ylabel="F_T (мН)",
                       legend=false, title="Тяга СПД-70")
        savefig("thrust.png")
        display(p_thrust)

        println("\n=== Статистика тяги ===")
        println("Мин. тяга: $(minimum(thrust_values)) мН")
        println("Макс. тяга: $(maximum(thrust_values)) мН")
        println("Средняя тяга (без первых 5%): $(mean(thrust_values[Int(ceil(0.05*length(thrust_values))):end])) мН")
    end
end

end # module